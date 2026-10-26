# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root sampling completion must not truncate an immediate child turn."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import TurnMetadata
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.credit.dispatch import ChildDispatchResult
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.request_rate import RequestRateStrategy
from tests.unit.timing.test_branch_orchestrator_dispatch_offset import (
    _mk_conv,
    _mk_credit,
    _mk_harness,
    _parent_conv,
    _spawn_branch,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delay_ms,disposition,issued,truncated",
    [
        param(None, ChildDispatchResult.ISSUED, 1, 0, id="absent-delay"),
        param(0.0, ChildDispatchResult.ISSUED, 1, 0, id="zero-delay"),
        param(250.0, ChildDispatchResult.ISSUED, 0, 1, id="positive-delay-cutoff"),
        param(0.0, ChildDispatchResult.REJECTED, 1, 1, id="zero-delay-refused"),
    ],
)
async def test_child_return_immediately_before_root_sending_cutoff(
    delay_ms: float | None,
    disposition: ChildDispatchResult,
    issued: int,
    truncated: int,
) -> None:
    runtime = VirtualTimingRuntime()
    scheduler = LoopScheduler(runtime=runtime)
    parent = _parent_conv(
        [_spawn_branch("branch", ["kid"], start_timestamp_ms=None, is_background=True)]
    )
    turns = [TurnMetadata(), TurnMetadata(delay_ms=delay_ms)]
    orchestrator, source, issuer = _mk_harness(
        [parent, _mk_conv("kid", turns)], scheduler=scheduler
    )
    source.get_next_turn_metadata.return_value = turns[1]
    issuer.dispatch_child_turn = AsyncMock(return_value=disposition)
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        expected_num_sessions=1,
    )
    strategy = RequestRateStrategy(
        config=config,
        conversation_source=source,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        branch_orchestrator=orchestrator,
    )
    await orchestrator.intercept(_mk_credit("parent", "P", 0))
    assert orchestrator.stats.children_spawned == 1
    child = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="kid",
        x_correlation_id="corr-kid",
        turn_index=0,
        num_turns=2,
        issued_at_ns=0,
        agent_depth=1,
        parent_correlation_id="P",
    )
    with runtime.activity_scope():
        await strategy.handle_credit_return(child)
        # The final parent issuance wakes PhaseRunner before queued tasks run.
        scheduler.cancel_all_pending()
        await orchestrator.expire_replay_deadlines()
    await runtime.drain_current()
    assert issuer.dispatch_child_turn.await_count == issued
    if issued and disposition is ChildDispatchResult.ISSUED:
        await orchestrator.on_child_leaf_reached("corr-kid")
    assert orchestrator.stats.children_completed == 1 - truncated
    assert orchestrator.stats.children_truncated == truncated
    assert not orchestrator.has_pending_branch_work()
    assert scheduler.is_idle()
