# SPDX-License-Identifier: Apache-2.0
"""Preserve official1.0.12 routing and cutoff rules under the virtual runtime."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import TurnMetadata
from aiperf.common.timing_runtime import RealTimingRuntime
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.credit.sticky_router import StickyCreditRouter, _StickyEntry
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.branch_orchestrator import (
    BranchOrchestrator,
    ChildJoinEntry,
    PendingBranchJoin,
    PrereqState,
)
from tests.unit.timing.conftest import make_credit
from tests.unit.timing.strategies.test_agentic_replay import _make_strategy


@pytest.mark.asyncio
async def test_spawn_keeps_own_worker_after_parent_entry_closes(benchmark_run) -> None:
    router = StickyCreditRouter(run=benchmark_run, service_id="spawn-1012")
    router._router_client.send_to = AsyncMock()
    router._register_worker("worker-a")
    router._register_worker("worker-b")
    router._sticky_sessions["parent"] = _StickyEntry(worker_id="worker-a")
    child = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="child-conversation",
        x_correlation_id="child",
        parent_correlation_id="parent",
        branch_mode=ConversationBranchMode.SPAWN,
        agent_depth=1,
        turn_index=0,
        num_turns=2,
        issued_at_ns=1,
    )
    await router.send_credit(child)
    assert router._sticky_sessions["child"].worker_id == "worker-a"
    del router._sticky_sessions["parent"]
    router._sticky_sessions["parent"] = _StickyEntry(worker_id="worker-b")
    final = Credit(
        id=2,
        phase=child.phase,
        conversation_id=child.conversation_id,
        x_correlation_id=child.x_correlation_id,
        parent_correlation_id=child.parent_correlation_id,
        branch_mode=child.branch_mode,
        agent_depth=1,
        turn_index=1,
        num_turns=2,
        issued_at_ns=2,
    )
    await router.send_credit(final)
    assert [call.args[0] for call in router._router_client.send_to.call_args_list] == [
        "worker-a",
        "worker-a",
    ]
    assert "child" not in router._sticky_sessions


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.asyncio
async def test_cutoff_releases_parked_child_and_suppresses_parent(virtual) -> None:
    runtime = VirtualTimingRuntime() if virtual else RealTimingRuntime()
    scheduler = LoopScheduler(runtime=runtime)
    issuer = Mock(dispatch_join_turn=AsyncMock(return_value=False))
    branches = BranchOrchestrator(
        conversation_source=SimpleNamespace(dataset_metadata=None),
        credit_issuer=issuer,
        scheduler=scheduler,
    )
    pending = PendingBranchJoin(
        parent_x_correlation_id="parent",
        parent_conversation_id="parent-conversation",
        parent_num_turns=2,
        gated_turn_index=1,
        is_blocked=True,
        replay_deadline_elapsed=False,
        outstanding={"spawn:child": PrereqState(expected=1, registered=True)},
    )
    branches._active_joins["parent"] = pending
    branches._child_to_join["child"] = [
        ChildJoinEntry(
            parent_correlation_id="parent",
            gated_turn_index=1,
            prereq_key="spawn:child",
        )
    ]
    branches._descendant_counts["parent"] = 1
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING, trajectories=[], scheduler=scheduler
    )
    strategy.branch_orchestrator = branches
    turn = TurnToSend(
        conversation_id="child-conversation",
        x_correlation_id="child",
        parent_correlation_id="parent",
        root_correlation_id="parent",
        agent_depth=1,
        turn_index=1,
        num_turns=2,
    )
    dispatch = AsyncMock()
    strategy._schedule_child_turn(10.0, turn, dispatch)
    assert branches._parked_child_turns == {"child"}
    scheduler.cancel_all_pending()
    await branches.expire_replay_deadlines()
    if virtual:
        await runtime.drain_current()
    dispatch.assert_not_awaited()
    issuer.dispatch_join_turn.assert_awaited_once_with(pending)
    assert branches.stats.children_truncated == 1
    assert branches.stats.joins_suppressed == 1
    assert not branches.has_pending_branch_work()
    await strategy._dispatch_parked_child_turn(turn, dispatch)
    await branches.expire_replay_deadlines()
    dispatch.assert_not_awaited()
    issuer.dispatch_join_turn.assert_awaited_once()
    scheduler.cancel_all()
    if virtual:
        await runtime.drain_current()
        runtime.assert_drained()


@pytest.mark.asyncio
async def test_virtual_child_timer_claims_dispatch_once() -> None:
    runtime = VirtualTimingRuntime()
    scheduler = LoopScheduler(runtime=runtime)
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING, trajectories=[], scheduler=scheduler
    )
    branches = BranchOrchestrator(
        conversation_source=SimpleNamespace(dataset_metadata=None),
        credit_issuer=Mock(),
        scheduler=scheduler,
    )
    strategy.branch_orchestrator = branches
    turn = TurnToSend(
        conversation_id="child-conversation",
        x_correlation_id="child",
        root_correlation_id="parent",
        agent_depth=1,
        num_turns=2,
        turn_index=1,
    )
    dispatch = AsyncMock()
    strategy._schedule_child_turn(10.0, turn, dispatch)
    runtime.advance_to(10_000_000_000)
    await runtime.drain_current()
    dispatch.assert_awaited_once_with(turn)
    await strategy._dispatch_parked_child_turn(turn, dispatch)
    dispatch.assert_awaited_once()
    assert not branches._parked_child_turns
    scheduler.cancel_all()
    await runtime.drain_current()
    runtime.assert_drained()


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.asyncio
async def test_immediate_continuation_preserves_credit_with_each_clock(virtual) -> None:
    runtime = VirtualTimingRuntime() if virtual else RealTimingRuntime()
    scheduler = LoopScheduler(runtime=runtime)
    strategy, issuer, _, source = _make_strategy(
        phase=CreditPhase.PROFILING, trajectories=[], scheduler=scheduler
    )
    source.get_next_turn_metadata = Mock(return_value=TurnMetadata(delay_ms=0))
    credit = make_credit(turn=0, num_turns=2, corr_id="same-session")
    await strategy._dispatch_next_turn(credit)
    if virtual:
        await runtime.drain_current()
    issuer.issue_credit.assert_awaited_once()
    sent = issuer.issue_credit.await_args.args[0]
    assert sent.turn_index == 1 and sent.x_correlation_id == "same-session"
    scheduler.cancel_all()
    if virtual:
        await runtime.drain_current()
        runtime.assert_drained()


@pytest.mark.asyncio
async def test_delayed_spawn_dispatched_before_cutoff_keeps_join_until_completion() -> (
    None
):
    runtime = VirtualTimingRuntime()
    scheduler = LoopScheduler(runtime=runtime)
    issuer = Mock(
        dispatch_first_turn=AsyncMock(return_value=True),
        dispatch_join_turn=AsyncMock(return_value=False),
    )
    branches = BranchOrchestrator(
        conversation_source=SimpleNamespace(dataset_metadata=None),
        credit_issuer=issuer,
        scheduler=scheduler,
    )
    pending = PendingBranchJoin(
        parent_x_correlation_id="parent",
        parent_conversation_id="parent-conversation",
        parent_num_turns=2,
        gated_turn_index=1,
        is_blocked=True,
        replay_deadline_elapsed=False,
        outstanding={"spawn:child": PrereqState(expected=1, registered=True)},
    )
    branches._active_joins["parent"] = pending
    branches._child_to_join["child"] = [
        ChildJoinEntry(
            parent_correlation_id="parent", gated_turn_index=1, prereq_key="spawn:child"
        )
    ]
    branches._child_modes["child"] = ConversationBranchMode.SPAWN
    branches._descendant_counts["parent"] = 1
    branches.stats.children_spawned = 1
    child = SimpleNamespace(
        x_correlation_id="child", effective_root_correlation_id="parent"
    )
    branches._start_delayed_first_turn(child, 1000, "parent")
    runtime.advance_to(1_000_000_000)
    await runtime.drain_current()
    issuer.dispatch_first_turn.assert_awaited_once_with(child)
    assert not branches._pending_delayed_children

    # The already-issued child remains active as the phase stops sending.
    runtime.advance_to(2_000_000_000)
    scheduler.cancel_all_pending()
    await branches.expire_replay_deadlines()
    assert branches._descendant_counts["parent"] == 1
    assert pending.outstanding["spawn:child"].expected == 1
    assert not pending.outstanding["spawn:child"].completed
    assert branches.stats.children_spawned == 1
    assert branches.stats.children_truncated == 0
    issuer.dispatch_join_turn.assert_not_awaited()

    await branches.on_child_leaf_reached("child")
    await branches.on_child_leaf_reached("child")
    await runtime.drain_current()
    issuer.dispatch_join_turn.assert_awaited_once_with(pending)
    assert branches.stats.children_completed == 1
    assert not branches._child_to_join
    assert not branches.has_pending_branch_work()
    scheduler.cancel_all()
    await runtime.drain_current()
    runtime.assert_drained()
