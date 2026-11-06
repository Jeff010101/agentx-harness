# SPDX-License-Identifier: Apache-2.0
"""Finite populations preserve cache, HTTP and virtual-clock boundaries."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pytest import param

from aiperf.simulation.fixed import FixedDriver, FixedPhase, FixedPlan
from aiperf.simulation.protocol import Grant, Ready, SimulationError, Terminal


def plan() -> FixedPlan:
    """Use an independent warmup prompt and two prepared prefix lanes."""
    return FixedPlan(
        run_id="fixed-test",
        phases=[
            FixedPhase(
                name="warmup",
                concurrency=2,
                request_indices=[4] * 4,
                input_tokens=3,
                output_tokens=2,
            ),
            FixedPhase(
                name="prepare",
                concurrency=2,
                request_indices=[0, 1],
                input_tokens=1,
                output_tokens=1,
                expected_cached_tokens=0,
            ),
            FixedPhase(
                name="profiling",
                concurrency=2,
                request_indices=[0, 1, 2, 3],
                input_tokens=3,
                output_tokens=2,
                expected_cached_tokens=1,
            ),
        ],
    )


PROMPTS = [[11, 12, 13], [21, 22, 23], [11, 32, 33], [21, 42, 43], [51, 52, 53]]


class Control:
    def __init__(self, fault: str | None = None) -> None:
        self.now = 0
        self.rounds: list[Ready] = []
        self.posted: dict[str, tuple[Any, ...]] = {}
        self.pending: dict[str, int] = {}
        self.completed = 0
        self.max_outstanding = 0
        self.fault = fault

    async def post(
        self, rid: str, ids: Any, sampling: dict[str, Any]
    ) -> dict[str, Any]:
        future = asyncio.get_running_loop().create_future()
        self.posted[rid] = (list(ids), sampling, future)
        return await future

    async def request(self, ready: Ready) -> Grant:
        self.rounds.append(ready)
        assert ready.round == len(self.rounds) - 1
        if ready.finished:
            assert not self.pending
            return Grant(ready.run_id, ready.round, self.now, [], "FINISHED")
        for op in ready.operations:
            assert op.rid in self.posted
            self.pending[op.rid] = self.now
        self.max_outstanding = max(self.max_outstanding, len(self.pending))
        rid = list(self.pending)[-1]
        arrival = self.pending.pop(rid)
        ids, sampling, future = self.posted[rid]
        if self.fault == "http":
            future.set_exception(ValueError("HTTP failed"))
            await asyncio.Event().wait()
        self.now += 100
        count = sampling["max_new_tokens"]
        first = self.now - 80
        last = first if count == 1 else self.now - 10
        snapshot = dict(
            version=1,
            rid=rid,
            arrival_ns=arrival,
            first_token_ns=first,
            last_token_ns=last,
            commit_ns=self.now,
            terminal_ns=self.now,
            prompt_tokens=len(ids),
            completion_tokens=count,
            cached_tokens=1 if self.completed >= 6 else 0,
        )
        if self.fault == "cache":
            snapshot["cached_tokens"] = len(ids)
        elif self.fault == "tokens":
            snapshot["completion_tokens"] = count + 1
        elif self.fault == "snapshot":
            snapshot["arrival_ns"] = self.now + 1
        future.set_result(
            {
                "meta_info": {
                    "sglang_simulator": snapshot,
                    "prompt_tokens": len(ids),
                    "completion_tokens": count,
                }
            }
        )
        self.completed += 1
        terminal = Terminal(
            "unknown" if self.fault == "terminal" else rid, "COMPLETED", self.now
        )
        return Grant(
            ready.run_id, ready.round + int(self.fault == "round"), self.now, [terminal]
        )


@pytest.mark.asyncio
async def test_completion_refill_and_phase_barriers_exclude_warmup() -> None:
    """Out-of-order terminals must not mix metric populations or exceed C."""
    control = Control()
    events = []

    async def emit(row: dict[str, Any]) -> None:
        events.append(row)

    result = await FixedDriver(control, control.post, emit, plan()).run(PROMPTS)
    assert result["completed"]
    assert result["phase_counts"] == {"warmup": 4, "prepare": 2, "profiling": 4}
    assert result["profiling"]["input_tokens"] == 12
    assert result["profiling"]["output_tokens"] == 8
    assert result["profiling"]["duration_seconds"] == 400 / 1e9
    assert control.max_outstanding == 2
    requests = [row for row in events if row["event"] == "Request"]
    assert [row["phase"] for row in requests] == ["warmup"] * 4 + ["prepare"] * 2 + [
        "profiling"
    ] * 4
    assert {row["prompt_index"] for row in requests[:4]} == {4}
    assert all(row["cached_tokens"] == 1 for row in requests[-4:])
    assert requests[-4]["arrival_ns"] == 600
    assert control.rounds[-1].finished
    assert all(sampling["ignore_eos"] for _, sampling, _ in control.posted.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [param("cache", id="cache"), param("tokens", id="tokens"), param("snapshot", id="timestamp"), param("terminal", id="unknown-terminal"), param("round", id="wrong-round")])  # fmt: skip
async def test_invalid_evidence_never_finishes_or_retries(fault: str) -> None:
    """A failed observation retains a partial run instead of a success result."""
    control = Control(fault)

    async def emit(row: dict[str, Any]) -> None:
        pass

    driver = FixedDriver(control, control.post, emit, plan())
    with pytest.raises(SimulationError):
        await driver.run(PROMPTS)
    assert not driver.pending
    assert not any(ready.finished for ready in control.rounds)
    assert len(
        {rid for ready in control.rounds for rid in [op.rid for op in ready.operations]}
    ) == len(control.posted)


@pytest.mark.asyncio
async def test_http_failure_interrupts_control_wait_and_cancels_other_requests() -> (
    None
):
    """Fail immediately when HTTP breaks while the coordinator is waiting."""
    control = Control("http")

    async def emit(row: dict[str, Any]) -> None:
        pass

    driver = FixedDriver(control, control.post, emit, plan())
    with pytest.raises(ValueError, match="HTTP failed"):
        await asyncio.wait_for(driver.run(PROMPTS), 1)
    assert len(control.rounds) == 1
    assert not driver.pending
    assert all(future.done() for _, _, future in control.posted.values())


@pytest.mark.asyncio
async def test_corrupt_input_fails_before_first_ready() -> None:
    """Malformed IDs must not leave an ambiguous submitted request."""
    control = Control()

    async def emit(row: dict[str, Any]) -> None:
        pass

    with pytest.raises(ValueError, match="uint32"):
        await FixedDriver(control, control.post, emit, plan()).run(
            [[True, 12, 13], *PROMPTS[1:]]
        )
    assert not control.rounds and not control.posted


def test_cache_preparation_after_profiling_is_rejected() -> None:
    """Keep preparation outside the reported profiling envelope."""
    selected = plan()
    selected.phases.reverse()
    with pytest.raises(ValueError, match="precede profiling"):
        selected.validate_inputs(PROMPTS)
