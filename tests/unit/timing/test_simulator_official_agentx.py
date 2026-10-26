# SPDX-License-Identifier: Apache-2.0
"""Official AgentX timing policies keep their causality under virtual time."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.dispatch import TurnAdmission
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator
from aiperf.timing.trajectory_source import Trajectory
from tests.unit.timing.strategies.test_agentic_replay import _make_strategy
from tests.unit.timing.test_replay_barrier_coordinator import _credit, _metadata


def warmup_runner(
    runtime: VirtualTimingRuntime,
) -> tuple[PhaseRunner, CreditCallbackHandler, SimpleNamespace]:
    """Create a real return handler with a paused DAG and one outstanding wire."""
    concurrency = Mock()
    branches = Mock(
        has_pending_branch_work=Mock(return_value=True),
        intercept=AsyncMock(return_value=True),
    )
    callback = CreditCallbackHandler(concurrency, branch_orchestrator=branches)
    strategy = SimpleNamespace(
        allows_pending_branch_handoff_after_sending_complete=True,
    )
    config = CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        warmup_requests_per_lane=10,
        grace_period_sec=1800.0,
        concurrency=1,
    )
    runner = PhaseRunner(
        config=config,
        conversation_source=SimpleNamespace(
            dataset_metadata=None,
            trajectories=[Mock()],
            warmup_credit_counts_by_lane=(0,),
        ),
        phase_publisher=Mock(
            publish_progress=AsyncMock(), publish_phase_complete=AsyncMock()
        ),
        credit_router=Mock(
            cancel_all_credits=AsyncMock(),
            wait_for_cancellation_drain=AsyncMock(),
        ),
        concurrency_manager=concurrency,
        callback_handler=callback,
        branch_orchestrator=branches,
        runtime=runtime,
    )
    callback.register_phase(
        phase=CreditPhase.WARMUP,
        phase_index=None,
        progress=runner._progress,
        lifecycle=runner._lifecycle,
        stop_checker=runner._stop_checker,
        strategy=strategy,
    )
    runner._lifecycle.start()
    runner._progress.increment_sent(TurnToSend("trace", "root", 0, 2))
    runner._progress.freeze_sent_counts()
    runner._lifecycle.mark_sending_complete()
    return runner, callback, strategy


@pytest.mark.asyncio
async def test_virtual_warmup_handoff_waits_for_return_processing_without_dag_drain() -> (
    None
):
    runtime = VirtualTimingRuntime()
    runner, callback, _ = warmup_runner(runtime)
    credit = Credit(
        id=0,
        phase=CreditPhase.WARMUP,
        conversation_id="trace",
        x_correlation_id="root",
        turn_index=0,
        num_turns=2,
        issued_at_ns=0,
    )
    handoff = runtime.create_task(runner._wait_for_accelerated_warmup_handoff())
    await runtime.drain_current()
    assert not handoff.done()
    assert runtime.next_deadline_ns == 1800_000_000_000
    runtime.advance_to(17_000_000_000)
    with runtime.activity_scope():
        await callback.on_credit_return("worker", CreditReturn(credit=credit))
    await runtime.drain_current()
    await handoff
    assert runtime.now_ns() == 17_000_000_000
    assert runner._progress.in_flight == 0
    assert runner._branch_orchestrator.has_pending_branch_work()
    runner._branch_orchestrator.expire_replay_deadlines.assert_not_called()
    runner._credit_router.cancel_all_credits.assert_not_awaited()
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_virtual_1800_second_warmup_grace_cancels_and_drains_before_failure() -> (
    None
):
    runtime = VirtualTimingRuntime()
    runner, _, _ = warmup_runner(runtime)
    drained = False

    async def settle(_: float) -> None:
        nonlocal drained
        assert runtime.now_ns() == 1800_000_000_000
        runner._progress.increment_returned(is_final_turn=False, cancelled=True)
        drained = True

    runner._credit_router.wait_for_cancellation_drain.side_effect = settle

    async def wait_handoff() -> None:
        with pytest.raises(TimeoutError, match="before all wire"):
            await runner._wait_for_accelerated_warmup_handoff()

    task = runtime.create_task(wait_handoff())
    await runtime.drain_current()
    runtime.advance_to(1799_000_000_000)
    await runtime.drain_current()
    runner._credit_router.cancel_all_credits.assert_not_awaited()
    assert not task.done()
    runtime.advance_to(1800_000_000_000)
    await runtime.drain_current()
    await task
    assert drained
    runner._credit_router.cancel_all_credits.assert_awaited_once()
    runner._credit_router.wait_for_cancellation_drain.assert_awaited_once()
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_virtual_warmup_cancellation_failure_cannot_force_success() -> None:
    runtime = VirtualTimingRuntime()
    runner, _, _ = warmup_runner(runtime)
    runner._credit_router.wait_for_cancellation_drain.side_effect = TimeoutError
    with (
        runtime.activity_scope(),
        pytest.raises(RuntimeError, match="Cancelled warmup credits did not drain"),
    ):
        await runner._cancel_accelerated_warmup_drain(timeout=1800.0)
    assert not runner._progress.all_credits_returned_event.is_set()
    runner._concurrency_manager.release_stuck_slots.assert_not_called()
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_virtual_trace_idle_cap_advances_only_its_idle_tree() -> None:
    runtime = VirtualTimingRuntime()
    scheduler = LoopScheduler(runtime=runtime)
    barrier = ReplayBarrierCoordinator(
        _metadata(), scheduler=scheduler, root_idle_gap_cap_seconds=300.0
    )
    barrier.activate()
    fired: list[tuple[str, int]] = []

    async def mark(root: str) -> None:
        fired.append((root, runtime.now_ns()))

    scheduler.schedule_later(1000.0, mark("idle"), group_id="idle")
    scheduler.schedule_later(1200.0, mark("busy"), group_id="busy")
    barrier.observe_issued(_credit("a", "busy"))
    barrier.observe_idle_root("idle")
    await runtime.drain_current()
    runtime.advance_to(300_000_000_000)
    await runtime.drain_current()
    assert fired == [("idle", 300_000_000_000)]
    assert runtime.next_deadline_ns == 1200_000_000_000
    await barrier.cancel_pending(notify_refused=False)
    scheduler.cancel_all()
    await runtime.drain_current()
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_virtual_system_idle_cap_waits_for_all_requests_and_advances_replay() -> (
    None
):
    runtime = VirtualTimingRuntime()
    scheduler = LoopScheduler(runtime=runtime)
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING, trajectories=[], scheduler=scheduler
    )
    strategy._runtime = runtime
    strategy._system_idle_gap_cap_seconds = 10.0
    fired = asyncio.Event()

    async def mark() -> None:
        fired.set()

    scheduler.schedule_later(1000.0, mark())
    strategy.enforce_system_idle_cap(1)
    await runtime.drain_current()
    assert runtime.next_deadline_ns == 1000_000_000_000
    strategy.enforce_system_idle_cap(0)
    assert runtime.next_deadline_ns == 10_000_000_000
    runtime.advance_to(10_000_000_000)
    await runtime.drain_current()
    assert fired.is_set()
    strategy._cancel_system_idle_watchdog()
    scheduler.cancel_all()
    await runtime.drain_current()
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_ten_request_lane_quota_counts_spawn_fanout_and_keeps_paused_turns() -> (
    None
):
    runtime = VirtualTimingRuntime()
    scheduler = LoopScheduler(runtime=runtime)
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=[
            Trajectory(conversation_id=f"trace_{lane}", start_turn_index=0)
            for lane in range(2)
        ],
        scheduler=scheduler,
        cache_warmup_requests_per_lane=10,
    )
    strategy._runtime = runtime
    issuer.set_turn_admission = Mock()
    await strategy.setup_phase()
    admit = issuer.set_turn_admission.call_args.args[0]
    roots = [f"root-{lane}" for lane in range(2)]
    strategy._root_to_lane.update({root: lane for lane, root in enumerate(roots)})
    primers = [
        TurnToSend("trace_0", roots[0], 0, 30),
        TurnToSend(
            "trace_0::primer",
            "primer-child",
            0,
            30,
            agent_depth=1,
            root_correlation_id=roots[0],
        ),
        TurnToSend("trace_1", roots[1], 0, 30),
    ]
    strategy._baseline_warmup_turns.update(
        (turn.x_correlation_id, turn.turn_index) for turn in primers
    )
    assert all(admit(turn) is TurnAdmission.ADMIT for turn in primers)
    assert not strategy._cache_warmup_requests_by_lane
    for index in range(10):
        for lane, root in enumerate(roots):
            child = index % 3 != 0
            turn = TurnToSend(
                f"trace_{lane}::child-{index}" if child else f"trace_{lane}",
                f"{root}-child-{index}" if child else root,
                0 if child else index + 1,
                30,
                agent_depth=int(child),
                parent_correlation_id=root if child else None,
                root_correlation_id=root if child else None,
                branch_mode=ConversationBranchMode.SPAWN,
            )
            assert admit(turn) is TurnAdmission.ADMIT
    assert dict(strategy._cache_warmup_requests_by_lane) == {0: 10, 1: 10}
    issuer.replay_gate.pause_releases.assert_called_once_with()
    for lane, root in enumerate(roots):
        for depth in (0, 1):
            turn = TurnToSend(
                f"trace_{lane}::paused" if depth else f"trace_{lane}",
                f"{root}-paused" if depth else root,
                1,
                30,
                agent_depth=depth,
                parent_correlation_id=root if depth else None,
                root_correlation_id=root if depth else None,
                branch_mode=ConversationBranchMode.SPAWN,
            )
            assert admit(turn) is TurnAdmission.DEFER
            assert admit(turn) is TurnAdmission.DEFER
    states = strategy._build_handoff_states()
    assert [len(states[lane]) for lane in range(2)] == [2, 2]
    for lane, root in enumerate(roots):
        assert {
            (state.x_correlation_id, state.agent_depth) for state in states[lane]
        } == {
            (root, 0),
            (f"{root}-paused", 1),
        }
        assert all(state.next_turn_index == 1 for state in states[lane])
    assert runtime.now_ns() == 0
    runtime.assert_drained()
