# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot timers retain their phase origin while initial credits backpressure."""

from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from tests.unit.timing.strategies.test_agentic_replay import (
    _burst_two_child_snapshot_source,
    _issued_child,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefill_wait_seconds",
    [param(0.0, id="ready"), param(5.0, id="backpressured")],
)  # fmt: skip
async def test_snapshot_later_timer_keeps_phase_origin(
    prefill_wait_seconds: float,
) -> None:
    source = _burst_two_child_snapshot_source()
    elapsed = 0.0
    deadlines: list[float] = []
    pending: list[Coroutine[Any, Any, object]] = []

    async def capture(_turn: object) -> bool:
        nonlocal elapsed
        elapsed += prefill_wait_seconds
        return True

    def schedule(
        delay: float, coro: Coroutine[Any, Any, object], **_kwargs: object
    ) -> None:
        deadlines.append(elapsed + delay)
        pending.append(coro)

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    issuer.dispatch_child_turn.side_effect = _issued_child(capture)
    scheduler = MagicMock()
    scheduler.schedule_later.side_effect = schedule
    config = MagicMock()
    config.phase = CreditPhase.PROFILING
    config.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=config,
        conversation_source=source,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        branch_orchestrator=MagicMock(),
    )
    try:
        await strategy.setup_phase()
        await strategy.execute_phase()
        # Global spread starts the 20s child at zero and the 95s child at 75s.
        assert deadlines == [pytest.approx(75.0)]
    finally:
        for coro in pending:
            coro.close()
