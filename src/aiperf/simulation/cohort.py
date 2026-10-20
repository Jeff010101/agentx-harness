# SPDX-License-Identifier: Apache-2.0
"""Keep a fresh-workflow cohort separate from the fixed measurement window."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import orjson
from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.simulation.protocol import SimulationError

if TYPE_CHECKING:
    from aiperf.common.timing_runtime import TimingRuntime


class WorkflowRecord(AIPerfBaseModel):
    """One whole tree admitted during the measurement window."""

    root_id: str = Field(
        min_length=1, description="Unique session-tree correlation ID."
    )
    trace_id: str = Field(min_length=1, description="Source root trace ID.")
    admitted_ns: int = Field(
        ge=0, description="Virtual time of session-slot acquisition."
    )
    completed_ns: int | None = Field(
        default=None, ge=0, description="Virtual time when the whole tree drained."
    )


class WorkflowCohort:
    """Observe actual tree admission/release; finish sending after the last tree."""

    def __init__(
        self,
        runtime: TimingRuntime,
        *,
        is_drained: Callable[[], bool],
        finish_sending: Callable[[], None],
    ) -> None:
        self._runtime = runtime
        self._is_drained = is_drained
        self._finish_sending = finish_sending
        self._closed = False
        self._finished = False
        self._roots: dict[str, WorkflowRecord] = {}

    def admit(self, root_id: str, trace_id: str) -> None:
        """Record admission when the real registry acquires its session slot."""
        if self._closed or root_id in self._roots:
            raise SimulationError("workflow admitted after cutoff or admitted twice")
        self._roots[root_id] = WorkflowRecord(
            root_id=root_id, trace_id=trace_id, admitted_ns=self._runtime.now_ns()
        )

    def complete(self, root_id: str) -> None:
        """Record completion only when root and all descendants have drained."""
        record = self._roots.get(root_id)
        if record is None or record.completed_ns is not None:
            raise SimulationError("unknown or duplicate workflow completion")
        record.completed_ns = self._runtime.now_ns()
        self._maybe_finish()

    def close_admission(self) -> None:
        """Leave existing turns and tool waits live after the window ends."""
        self._closed = True
        self._maybe_finish()

    def _maybe_finish(self) -> None:
        if self._closed and not self._finished and self._is_drained():
            if any(r.completed_ns is None for r in self._roots.values()):
                raise SimulationError(
                    "tree registry drained with an unfinished workflow"
                )
            self._finished = True
            self._finish_sending()

    async def save(
        self,
        path: Path,
        *,
        start_ns: int,
        end_ns: int,
        benchmark_id: str,
        phase_index: int | None,
    ) -> None:
        """Export uncensored cohort records; window throughput is computed separately."""
        if not self._finished:
            raise SimulationError("cannot export an incomplete workflow cohort")
        if not self._roots or any(
            not start_ns <= r.admitted_ns < end_ns
            or r.completed_ns is None
            or r.completed_ns < r.admitted_ns
            for r in self._roots.values()
        ):
            raise SimulationError(
                "workflow cohort has invalid admission/completion boundaries"
            )
        payload = {
            "version": 1,
            "clock": "virtual_ns",
            "benchmark_id": benchmark_id,
            "phase_index": phase_index,
            "measurement_start_ns": start_ns,
            "measurement_end_ns": end_ns,
            "admission_policy": "fresh workflows only; stop admission at measurement end",
            "completion_policy": "finish every admitted tree, including future turns and tool waits",
            "complete": True,
            "workflows": [r.model_dump() for r in self._roots.values()],
        }
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        async with aiofiles.open(path, "xb") as stream:
            await stream.write(
                orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n"
            )
