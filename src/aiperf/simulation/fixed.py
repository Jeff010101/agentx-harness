# SPDX-License-Identifier: Apache-2.0
"""Finite, completion-driven native requests over the Ready/Grant protocol.

This driver is separate from agentic replay: callers supply exact post-template
token IDs and explicitly separate cache preparation from profiling requests.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol

import msgspec
from pydantic import ConfigDict, Field, StrictInt

from aiperf.common.models import AIPerfBaseModel
from aiperf.simulation.protocol import (
    Grant,
    Ready,
    SimulationError,
    Submit,
)
from aiperf.simulation.record_timing import TimingSnapshot, _validate_snapshot


class FixedPhase(AIPerfBaseModel):
    """An explicit request population, drained before the next phase starts."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["warmup", "prepare", "profiling"] = Field(
        description="Metric ownership; only profiling contributes reported metrics."
    )
    concurrency: StrictInt = Field(gt=0, description="Maximum outstanding requests.")
    request_indices: list[
        Annotated[StrictInt, Field(ge=0, description="Zero-based prompt bank index.")]
    ] = Field(
        min_length=1, description="Ordered indices into the supplied prompt bank."
    )
    input_tokens: StrictInt = Field(gt=0, description="Exact prompt prefix to send.")
    output_tokens: StrictInt = Field(gt=0, description="Exact requested output count.")
    expected_cached_tokens: StrictInt | None = Field(
        default=None, ge=0, description="Required observed cache hit, if asserted."
    )


class FixedPlan(AIPerfBaseModel):
    """A finite workload; no model, source qualification or acceptance is inferred."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, description="Unique coordinator run identity.")
    phases: list[FixedPhase] = Field(
        min_length=1, description="Sequential phase populations with explicit barriers."
    )
    sampling_params: dict[str, Any] = Field(
        default_factory=dict, description="Original native sampling parameters."
    )
    wall_timeout_seconds: float = Field(
        default=1800,
        gt=0,
        allow_inf_nan=False,
        description="Physical failure deadline; never a virtual-time input.",
    )

    def validate_inputs(self, prompts: Sequence[Sequence[int]]) -> None:
        """Reject unsupported geometry before opening the control connection."""
        if any(
            k in self.sampling_params
            for k in ("max_new_tokens", "min_new_tokens", "ignore_eos")
        ):
            raise ValueError("phase output count and ignore_eos are driver-owned")
        if not any(p.name == "profiling" for p in self.phases):
            raise ValueError("a fixed plan requires a profiling phase")
        profiling_started = False
        for phase in self.phases:
            if profiling_started and phase.name != "profiling":
                raise ValueError("warmup and preparation must precede profiling")
            profiling_started |= phase.name == "profiling"
            if (
                phase.expected_cached_tokens is not None
                and phase.expected_cached_tokens > phase.input_tokens
            ):
                raise ValueError("cache assertion exceeds input length")
            for index in phase.request_indices:
                if (
                    not 0 <= index < len(prompts)
                    or len(prompts[index]) < phase.input_tokens
                ):
                    raise ValueError(
                        "request index or input length exceeds prompt bank"
                    )
        used = {index for phase in self.phases for index in phase.request_indices}
        for index in used:
            if any(
                type(token) is not int or not 0 <= token < 2**32
                for token in prompts[index]
            ):
                raise ValueError("prompt bank requires uint32 token IDs")


class FixedControl(Protocol):
    """The shared v1 wire client, injectable for protocol tests."""

    async def request(self, ready: Ready) -> Grant: ...


@dataclass(slots=True)
class _Pending:
    """One native response awaiting its matching virtual terminal."""

    task: asyncio.Task[dict[str, Any]]
    """The original HTTP exchange; never retried."""
    arrival_ns: int
    """Authoritative submission time from the previous Grant."""
    prompt_index: int
    """Original prompt-bank index retained in request evidence."""


PostRequest = Callable[[str, Sequence[int], dict[str, Any]], Awaitable[dict[str, Any]]]
EmitEvent = Callable[[dict[str, Any]], Awaitable[None]]


class FixedDriver:
    """Refill at each completed terminal, preserving phase and HTTP barriers."""

    def __init__(
        self, control: FixedControl, post: PostRequest, emit: EmitEvent, plan: FixedPlan
    ) -> None:
        self.control, self.post, self.emit, self.plan = control, post, emit, plan
        self.now_ns = self.round = self.serial = 0
        self.pending: dict[str, _Pending] = {}

    async def _exchange(
        self, operations: list[Submit], *, finished: bool = False
    ) -> Grant:
        ready = Ready(self.plan.run_id, self.round, operations, finished=finished)
        await self.emit({"event": "Ready", **msgspec.to_builtins(ready)})
        exchange = asyncio.create_task(self.control.request(ready))
        try:
            async with asyncio.timeout(self.plan.wall_timeout_seconds):
                while not exchange.done():
                    active = [
                        p.task for p in self.pending.values() if not p.task.done()
                    ]
                    for pending in self.pending.values():
                        if pending.task.done():
                            pending.task.result()
                    done, _ = await asyncio.wait(
                        [exchange, *active], return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        if task is not exchange:
                            task.result()
                grant = exchange.result()
            if (
                grant.protocol_version != 1
                or grant.run_id != ready.run_id
                or grant.round != ready.round
                or type(grant.now_ns) is not int
                or grant.now_ns < self.now_ns
                or grant.status == "ERROR"
                or any(event.at_ns != grant.now_ns for event in grant.terminals)
            ):
                raise SimulationError("fixed workload received an invalid Grant")
            await self.emit({"event": "Grant", **msgspec.to_builtins(grant)})
            self.now_ns, self.round = grant.now_ns, self.round + 1
            return grant
        finally:
            if not exchange.done():
                exchange.cancel()
            await asyncio.gather(exchange, return_exceptions=True)

    async def _phase(
        self, phase: FixedPhase, prompts: Sequence[Sequence[int]]
    ) -> list[dict[str, Any]]:
        queued = 0
        records: list[dict[str, Any]] = []
        while queued < len(phase.request_indices) or self.pending:
            operations = []
            while (
                queued < len(phase.request_indices)
                and len(self.pending) < phase.concurrency
            ):
                index = phase.request_indices[queued]
                rid = f"{self.plan.run_id}.fixed.{self.serial:016x}"
                sampling = {
                    **self.plan.sampling_params,
                    "max_new_tokens": phase.output_tokens,
                    "ignore_eos": True,
                }
                task = asyncio.create_task(
                    self.post(rid, prompts[index][: phase.input_tokens], sampling)
                )
                self.pending[rid] = _Pending(task, self.now_ns, index)
                operations.append(Submit(rid))
                self.serial += 1
                queued += 1
            grant = await self._exchange(operations)
            if grant.status != "CONTINUE":
                raise SimulationError("fixed workload ended before phase drain")
            for terminal in grant.terminals:
                if terminal.rid not in self.pending or terminal.kind != "COMPLETED":
                    raise SimulationError("unexpected fixed request terminal")
                pending = self.pending[terminal.rid]
                response = await asyncio.wait_for(
                    asyncio.shield(pending.task), self.plan.wall_timeout_seconds
                )
                record = self._record(
                    response,
                    rid=terminal.rid,
                    at_ns=terminal.at_ns,
                    pending=pending,
                    phase=phase,
                )
                await self.emit(record)
                records.append(record)
                del self.pending[terminal.rid]
        return records

    @staticmethod
    def _record(
        response: dict[str, Any],
        *,
        rid: str,
        at_ns: int,
        pending: _Pending,
        phase: FixedPhase,
    ) -> dict[str, Any]:
        if not isinstance(response, dict) or not isinstance(
            response.get("meta_info"), dict
        ):
            raise SimulationError("fixed native response omitted metadata")
        meta = response["meta_info"]
        try:
            snapshot = _validate_snapshot(meta.get("sglang_simulator"), rid, None)
        except (TypeError, ValueError) as error:
            raise SimulationError("fixed native timing snapshot is invalid") from error
        expected = (pending.arrival_ns, at_ns, phase.input_tokens, phase.output_tokens)
        actual = (
            snapshot.arrival_ns,
            snapshot.terminal_ns,
            snapshot.prompt_tokens,
            snapshot.completion_tokens,
        )
        if actual != expected or any(
            type(meta.get(k)) is not int or meta[k] != v
            for k, v in (
                ("prompt_tokens", phase.input_tokens),
                ("completion_tokens", phase.output_tokens),
            )
        ):
            raise SimulationError(
                "fixed response timing or token usage differs from submitted work"
            )
        if (
            phase.expected_cached_tokens is not None
            and snapshot.cached_tokens != phase.expected_cached_tokens
        ):
            raise SimulationError(
                "fixed response cache hit differs from requested prefix"
            )
        return {
            "event": "Request",
            "phase": phase.name,
            "prompt_index": pending.prompt_index,
            **{
                field: getattr(snapshot, field)
                for field in TimingSnapshot.__struct_fields__
            },
        }

    async def run(self, prompts: Sequence[Sequence[int]]) -> dict[str, Any]:
        """Require complete phase populations and FINISHED before reporting success."""
        self.plan.validate_inputs(prompts)
        records = []
        try:
            for phase in self.plan.phases:
                records.extend(await self._phase(phase, prompts))
            grant = await self._exchange([], finished=True)
            if grant.status != "FINISHED" or grant.terminals:
                raise SimulationError("coordinator did not confirm drained FINISHED")
            return {
                "completed": True,
                "clock": "source_coordinator_virtual_ns",
                "phase_counts": {
                    name: sum(row["phase"] == name for row in records)
                    for name in ("warmup", "prepare", "profiling")
                },
                "profiling": summarize_fixed(
                    [row for row in records if row["phase"] == "profiling"]
                ),
            }
        finally:
            tasks = [p.task for p in self.pending.values()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.pending.clear()


def summarize_fixed(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-request means and a submission-to-terminal throughput envelope."""
    if not rows:
        raise SimulationError("no profiling requests")
    start = min(row["arrival_ns"] for row in rows)
    end = max(row["terminal_ns"] for row in rows)
    if end <= start:
        raise SimulationError("nonpositive profiling duration")
    ttft = [(row["first_token_ns"] - row["arrival_ns"]) / 1e9 for row in rows]
    e2el = [(row["last_token_ns"] - row["arrival_ns"]) / 1e9 for row in rows]
    tpots = [
        (row["last_token_ns"] - row["first_token_ns"])
        / 1e9
        / (row["completion_tokens"] - 1)
        for row in rows
        if row["completion_tokens"] > 1
    ]
    duration = (end - start) / 1e9
    inputs = sum(row["prompt_tokens"] for row in rows)
    outputs = sum(row["completion_tokens"] for row in rows)
    mean_tpot = math.fsum(tpots) / len(tpots) if len(tpots) == len(rows) else None
    return {
        "completed_requests": len(rows),
        "duration_seconds": duration,
        "input_tokens": inputs,
        "output_tokens": outputs,
        "mean_ttft_seconds": math.fsum(ttft) / len(rows),
        "mean_e2el_seconds": math.fsum(e2el) / len(rows),
        "mean_tpot_seconds": mean_tpot,
        "mean_interactivity_tokens_per_second_per_user": 1 / mean_tpot
        if mean_tpot and mean_tpot > 0
        else None,
        "total_throughput_tokens_per_second": (inputs + outputs) / duration,
        "output_throughput_tokens_per_second": outputs / duration,
    }
