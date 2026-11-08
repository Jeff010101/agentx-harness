# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Count-driven phase waits must remain visible to the cooperative clock."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from pytest import param

from aiperf.common.timing_runtime import RealTimingRuntime, TimingRuntime
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.timing.phase.runner import PhaseRunner


def runner_for(runtime: TimingRuntime) -> PhaseRunner:
    return cast(PhaseRunner, SimpleNamespace(_runtime=runtime))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delay",
    [param(0.0, id="immediate-primers"), param(8.8, id="spread-primers")],
)  # fmt: skip
async def test_finished_dispatch_waits_for_original_virtual_warmup_deadline(
    delay: float,
) -> None:
    runtime = VirtualTimingRuntime()
    event = asyncio.Event()

    async def schedule_primers() -> None:
        if delay:
            runtime.call_later(delay, event.set)
        else:
            event.set()

    execution = runtime.create_task(schedule_primers())
    await execution

    async def phase_wait() -> None:
        with runtime.run_scope():
            await PhaseRunner._wait_for_event_or_task_exception(
                runner_for(runtime), event, execution, "warmup sending"
            )

    waiter = runtime.create_task(phase_wait())
    try:
        await asyncio.wait_for(runtime.drain_current(), timeout=1.0)
        if delay:
            assert not waiter.done()
            assert runtime.next_deadline_ns == 8_800_000_000
            runtime.advance_to(8_800_000_000)
            await asyncio.wait_for(runtime.drain_current(), timeout=1.0)
        await asyncio.wait_for(waiter, timeout=1.0)
        runtime.assert_drained()
        assert runtime.now_ns() == round(delay * 1_000_000_000)
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_event_completion_does_not_wait_for_unfinished_execution() -> None:
    runtime = VirtualTimingRuntime()
    event, hold = asyncio.Event(), asyncio.Event()

    async def execution_body() -> None:
        await runtime.wait_event(hold)

    execution = runtime.create_task(execution_body())
    runtime.call_later(2.0, event.set)

    async def phase_wait() -> None:
        with runtime.run_scope():
            await PhaseRunner._wait_for_event_or_task_exception(
                runner_for(runtime), event, execution, "warmup sending"
            )

    waiter = runtime.create_task(phase_wait())
    try:
        await asyncio.wait_for(runtime.drain_current(), timeout=1.0)
        runtime.advance_to(2_000_000_000)
        await asyncio.wait_for(runtime.drain_current(), timeout=1.0)
        await asyncio.wait_for(waiter, timeout=1.0)
        assert not execution.done()
    finally:
        waiter.cancel()
        execution.cancel()
        await asyncio.gather(waiter, execution, return_exceptions=True)
    await runtime.drain_current()
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_execution_failure_wins_over_same_tick_event() -> None:
    runtime = RealTimingRuntime()
    event = asyncio.Event()

    async def failed_execution() -> None:
        event.set()
        raise RuntimeError("credit dispatch failed")

    execution = runtime.create_task(failed_execution())
    await asyncio.gather(execution, return_exceptions=True)
    with pytest.raises(RuntimeError, match="credit dispatch failed"):
        await PhaseRunner._wait_for_event_or_task_exception(
            runner_for(runtime), event, execution, "warmup sending"
        )


@pytest.mark.asyncio
async def test_cancelled_phase_wait_drains_its_owned_event_activity() -> None:
    runtime = VirtualTimingRuntime()
    event = asyncio.Event()

    async def finished_execution() -> None:
        return

    execution = runtime.create_task(finished_execution())
    await execution

    async def phase_wait() -> None:
        with runtime.run_scope():
            await PhaseRunner._wait_for_event_or_task_exception(
                runner_for(runtime), event, execution, "warmup sending"
            )

    waiter = runtime.create_task(phase_wait())
    try:
        await asyncio.wait_for(runtime.drain_current(), timeout=1.0)
        assert not waiter.done()
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
    await runtime.drain_current()
    runtime.assert_drained()


@pytest.mark.asyncio
@pytest.mark.parametrize("grace", [None, float("inf"), 10.0])
async def test_accelerated_warmup_handoff_drains_at_virtual_return(
    grace: float | None,
) -> None:
    runtime = VirtualTimingRuntime()
    progress = SimpleNamespace(in_flight=1, all_credits_returned_event=asyncio.Event())
    runner = cast(
        PhaseRunner,
        SimpleNamespace(
            _runtime=runtime,
            _progress=progress,
            _config=SimpleNamespace(grace_period_sec=grace),
            info=Mock(),
        ),
    )
    runner._wait_for_accelerated_warmup_wire_drain = (
        lambda **kwargs: PhaseRunner._wait_for_accelerated_warmup_wire_drain(
            runner, **kwargs
        )
    )

    def returned() -> None:
        progress.in_flight = 0
        progress.all_credits_returned_event.set()

    runtime.call_later(8.8, returned)
    waiter = runtime.create_task(
        PhaseRunner._wait_for_accelerated_warmup_handoff(runner)
    )
    try:
        await asyncio.wait_for(runtime.drain_current(), timeout=1.0)
        assert not waiter.done()
        assert runtime.next_deadline_ns == 8_800_000_000
        runtime.advance_to(8_800_000_000)
        await asyncio.wait_for(runtime.drain_current(), timeout=1.0)
        await asyncio.wait_for(waiter, timeout=1.0)
        runtime.assert_drained()
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_accelerated_warmup_wire_timeout_uses_virtual_deadline() -> None:
    runtime = VirtualTimingRuntime()
    runner = cast(
        PhaseRunner,
        SimpleNamespace(
            _runtime=runtime,
            _progress=SimpleNamespace(
                in_flight=1, all_credits_returned_event=asyncio.Event()
            ),
        ),
    )

    async def wait_for_timeout() -> None:
        with pytest.raises(TimeoutError):
            await PhaseRunner._wait_for_accelerated_warmup_wire_drain(
                runner, timeout=5.0
            )

    waiter = runtime.create_task(wait_for_timeout())
    try:
        await asyncio.wait_for(runtime.drain_current(), timeout=1.0)
        assert not waiter.done()
        assert runtime.next_deadline_ns == 5_000_000_000
        runtime.advance_to(5_000_000_000)
        await asyncio.wait_for(runtime.drain_current(), timeout=1.0)
        await waiter
        runtime.assert_drained()
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("router_failure", [False, True])
async def test_virtual_warmup_cancel_failure_cannot_force_completion(
    router_failure: bool,
) -> None:
    runtime = VirtualTimingRuntime()
    runtime.wait_wall_event = AsyncMock(side_effect=TimeoutError)
    progress = SimpleNamespace(
        in_flight=1,
        all_credits_returned_event=asyncio.Event(),
        create_stats=Mock(),
    )
    router = SimpleNamespace(
        cancel_all_credits=AsyncMock(),
        wait_for_cancellation_drain=AsyncMock(
            side_effect=TimeoutError if router_failure else None
        ),
    )
    runner = cast(
        PhaseRunner,
        SimpleNamespace(
            _runtime=runtime,
            _progress=progress,
            _credit_router=router,
            _lifecycle=Mock(),
            warning=Mock(),
            info=Mock(),
            error=Mock(),
            _release_stuck_slots=Mock(),
        ),
    )
    runner._wait_for_cancellation_drain = (
        lambda timeout: PhaseRunner._wait_for_cancellation_drain(runner, timeout)
    )
    with pytest.raises(RuntimeError, match="credits did not drain"):
        await PhaseRunner._cancel_accelerated_warmup_drain(runner, timeout=5.0)
    assert not progress.all_credits_returned_event.is_set()
    runner._release_stuck_slots.assert_not_called()
    router.wait_for_cancellation_drain.assert_awaited_once()
