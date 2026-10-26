# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real replay watchdogs must drain after the last profiling return."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import DatasetMetadata
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator, ReplayIssueGate
from tests.unit.timing.phase.test_runner import (
    MockStrategy,
    cfg,
    make_runner,
    mock_callback,
    mock_cancel_policy,
    mock_conc_mgr,
)

pytestmark = pytest.mark.asyncio

PROFILE_START_NS = 34_100_080_933
SENDING_END_NS = 3_634_100_080_933
REQUESTS_END_NS = 3_634_958_039_561
WATCHDOG_NS = 3_934_958_039_561


@pytest.fixture
def runtime() -> VirtualTimingRuntime:
    return VirtualTimingRuntime()


@pytest.fixture
def credit() -> Credit:
    return Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="trace",
        x_correlation_id="root",
        turn_index=0,
        num_turns=2,
        issued_at_ns=PROFILE_START_NS,
    )


def replay_gate(scheduler: LoopScheduler) -> ReplayIssueGate:
    metadata = DatasetMetadata(
        conversations=[], sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )
    return ReplayIssueGate(
        ReplayBarrierCoordinator(
            metadata, scheduler=scheduler, root_idle_gap_cap_seconds=300
        )
    )


@pytest.fixture
async def runner(runtime: VirtualTimingRuntime) -> PhaseRunner:
    publisher = MagicMock()
    for name in (
        "publish_phase_start",
        "publish_phase_baseline_request",
        "publish_phase_sending_complete",
        "publish_phase_complete",
        "publish_progress",
    ):
        setattr(publisher, name, AsyncMock())
    router = MagicMock()
    router.wait_for_workers = AsyncMock()
    result = make_runner(
        cfg(reqs=None, dur=3600, grace=30),
        MagicMock(),
        publisher,
        router,
        mock_conc_mgr(),
        mock_cancel_policy(),
        mock_callback(),
    )
    result._runtime = runtime
    result._scheduler = LoopScheduler(runtime=runtime)
    result._lifecycle = PhaseLifecycle(result._config, runtime=runtime)
    result._credit_issuer.replay_gate = replay_gate(result._scheduler)
    return result


async def test_sending_stop_then_last_return_leaves_root_watchdog(
    runtime: VirtualTimingRuntime, credit: Credit
) -> None:
    scheduler = LoopScheduler(runtime=runtime)
    gate = replay_gate(scheduler)
    gate.activate()
    await gate.observe_issued(credit)
    runtime.advance_to(SENDING_END_NS)
    await gate.cancel(notify_refused=False)
    runtime.advance_to(REQUESTS_END_NS)
    gate.complete(credit)
    scheduler.cancel_all()
    assert runtime.next_deadline_ns == WATCHDOG_NS
    with pytest.raises(
        RuntimeError, match=f"Unfinished timing deadline: {WATCHDOG_NS}"
    ):
        runtime.assert_drained()
    await gate.cancel(notify_refused=False)
    runtime.assert_drained()


async def test_final_phase_cleans_last_return_watchdog_after_handoff(
    runner: PhaseRunner,
    runtime: VirtualTimingRuntime,
    credit: Credit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = runner._credit_issuer.replay_gate

    async def sending(strategy: MockStrategy) -> None:
        gate.activate()
        await gate.observe_issued(credit)
        runtime.advance_to(SENDING_END_NS)
        await gate.cancel(notify_refused=False)

    async def returning(strategy: MockStrategy, *, phase_id: str) -> None:
        runtime.advance_to(REQUESTS_END_NS)
        gate.complete(credit)
        assert runtime.next_deadline_ns == WATCHDOG_NS

    async def finalize() -> None:
        # A source snapshot must still see state retained by the last return.
        assert runtime.next_deadline_ns == WATCHDOG_NS

    strategy = MockStrategy()
    strategy.finalize_phase = AsyncMock(side_effect=finalize)
    monkeypatch.setattr(runner, "_wait_for_sending_complete", sending)
    monkeypatch.setattr(runner, "_wait_for_returning_complete", returning)
    runtime.advance_to(PROFILE_START_NS)
    await runner._run_strategy(strategy, is_final_phase=True)
    await runtime.drain_current()
    runtime.assert_drained()
    assert runtime.now_ns() == REQUESTS_END_NS
    strategy.finalize_phase.assert_awaited_once()


async def test_return_failure_preserves_original_exception(
    runner: PhaseRunner,
    runtime: VirtualTimingRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ValueError("original return failure")
    monkeypatch.setattr(runner, "_wait_for_sending_complete", AsyncMock())
    monkeypatch.setattr(
        runner, "_wait_for_returning_complete", AsyncMock(side_effect=original)
    )
    strategy = MockStrategy()
    strategy.finalize_phase = AsyncMock()
    with pytest.raises(ValueError) as caught:
        await runner._run_strategy(strategy, is_final_phase=True)
    assert caught.value is original
    strategy.finalize_phase.assert_not_awaited()
    runner._progress_task.cancel()
    await asyncio.gather(runner._progress_task, return_exceptions=True)
    runner._scheduler.cancel_all()
    await runtime.drain_current()


async def test_seamless_non_final_phase_does_not_close_background_returns(
    runner: PhaseRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner._config = runner._config.model_copy(update={"seamless": True})
    gate = runner._credit_issuer.replay_gate
    gate.cancel = AsyncMock(wraps=gate.cancel)
    release = asyncio.Event()
    monkeypatch.setattr(runner, "_wait_for_sending_complete", AsyncMock())

    async def returning(strategy: MockStrategy, *, phase_id: str) -> None:
        await release.wait()

    monkeypatch.setattr(runner, "_wait_for_returning_complete", returning)
    await runner._run_strategy(MockStrategy(), is_final_phase=False)
    gate.cancel.assert_not_awaited()
    assert not runner._return_wait_task.done()
    release.set()
    await runner._return_wait_task
    await asyncio.gather(runner._progress_task, return_exceptions=True)


async def test_active_trace_idle_cap_keeps_exact_300_second_boundary(
    runtime: VirtualTimingRuntime, credit: Credit
) -> None:
    scheduler = LoopScheduler(runtime=runtime)
    gate = replay_gate(scheduler)
    sent = []

    async def request() -> None:
        sent.append(runtime.now_ns())

    gate.activate()
    await gate.observe_issued(credit)
    scheduler.schedule_later(
        500, request(), group_id=credit.effective_root_correlation_id
    )
    gate.complete(credit)
    runtime.advance_to(299_999_999_999)
    await runtime.drain_current()
    assert not sent
    runtime.advance_to(300_000_000_000)
    await runtime.drain_current()
    assert sent == [300_000_000_000]
    await gate.cancel(notify_refused=False)
    scheduler.cancel_all()
    runtime.assert_drained()
