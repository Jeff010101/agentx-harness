# SPDX-License-Identifier: Apache-2.0
"""A deadline closes root admission, not unfinished branches or tool waits."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import orjson
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.plugin.enums import TimingMode
from aiperf.simulation.cohort import WorkflowCohort
from aiperf.simulation.config import validate_simulation
from aiperf.simulation.protocol import SimulationError
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.stop_conditions import StopConditionChecker
from aiperf.timing.session_tree import SessionTreeRegistry
from tests.unit.timing.strategies.test_agentic_replay import _make_strategy
from tests.unit.timing.test_simulator_config import configuration


@pytest.mark.asyncio
async def test_background_descendants_finish_after_window(tmp_path: Path) -> None:
    clock = VirtualTimingRuntime()
    registry = SessionTreeRegistry(Mock())
    finished = Mock()
    cohort = WorkflowCohort(
        clock, is_drained=lambda: registry.open_count(0) == 0, finish_sending=finished
    )
    registry.set_drain_callback(
        lambda root, phase: cohort.complete(root),
        on_open=lambda root, phase: cohort.admit(root, "trace"),
    )
    registry.open_tree("root", 0, root_pending=True)
    registry.register_descendants("root", 2)
    clock.advance_to(500_000_000)
    assert not registry.on_root_terminal("root")
    assert not registry.on_descendant_done("root")
    clock.advance_to(1_000_000_000)
    cohort.close_admission()
    finished.assert_not_called()
    clock.advance_to(3_000_000_000)
    assert registry.on_descendant_done("root")
    finished.assert_called_once_with()
    cohort.close_admission()
    finished.assert_called_once_with()
    path = tmp_path / "cohort.json"
    await cohort.save(
        path, start_ns=0, end_ns=1_000_000_000, benchmark_id="bench", phase_index=0
    )
    result = orjson.loads(path.read_bytes())
    assert result["complete"] is True
    assert result["measurement_end_ns"] == 1_000_000_000
    assert result["workflows"] == [
        dict(
            root_id="root", trace_id="trace", admitted_ns=0, completed_ns=3_000_000_000
        )
    ]


@pytest.mark.asyncio
async def test_strategy_finalization_exports_into_the_actual_run(
    tmp_path: Path,
) -> None:
    run = MagicMock(artifact_dir=tmp_path, benchmark_id="bench")
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING, trajectories=[], run=run
    )
    strategy.config.phase_index = 0
    strategy.config.expected_duration_sec = 1
    strategy._system_idle_gap_cap_seconds = None
    clock = VirtualTimingRuntime()
    lifecycle = PhaseLifecycle(strategy.config, runtime=clock)
    lifecycle.start()
    strategy.lifecycle = lifecycle
    cohort = WorkflowCohort(clock, is_drained=lambda: True, finish_sending=Mock())
    strategy._workflow_cohort = cohort
    cohort.admit("root", "trace")
    clock.advance_to(3_000_000_000)
    cohort.complete("root")
    cohort.close_admission()
    await strategy.finalize_phase()
    result = orjson.loads((tmp_path / "workflow_cohort.0.json").read_bytes())
    assert result["benchmark_id"] == "bench"
    assert result["measurement_end_ns"] == 1_000_000_000
    assert result["workflows"][0]["completed_ns"] == 3_000_000_000


@pytest.mark.parametrize("drain", [False, True])
def test_cutoff_and_cancellation_gates(drain: bool) -> None:
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        expected_duration_sec=1,
        drain_admitted_workflows=drain,
    )
    clock = VirtualTimingRuntime()
    lifecycle = PhaseLifecycle(config, runtime=clock)
    lifecycle.start()
    checker = StopConditionChecker(config, lifecycle, CreditCounter(config))
    assert checker.can_start_new_session()
    clock.advance_to(1_000_000_000)
    assert not checker.can_start_new_session()
    assert checker.can_send_any_turn() is drain
    assert checker.can_send_dag_child_turn() is drain
    lifecycle.cancel()
    assert not checker.can_send_any_turn()
    assert not checker.can_send_dag_child_turn()


@pytest.mark.asyncio
async def test_incomplete_or_invalid_cohort_is_not_exported(tmp_path: Path) -> None:
    clock = VirtualTimingRuntime()
    cohort = WorkflowCohort(clock, is_drained=lambda: True, finish_sending=Mock())
    cohort.admit("root", "trace")
    with pytest.raises(SimulationError, match="incomplete"):
        await cohort.save(
            tmp_path / "unfinished.json",
            start_ns=0,
            end_ns=1,
            benchmark_id="bench",
            phase_index=0,
        )
    with pytest.raises(SimulationError, match="unfinished workflow"):
        cohort.close_admission()
    with pytest.raises(SimulationError, match="after cutoff"):
        cohort.admit("late", "trace")
    with pytest.raises(SimulationError, match="unknown"):
        cohort.complete("unknown")
    cohort.complete("root")
    with pytest.raises(SimulationError, match="duplicate"):
        cohort.complete("root")
    with pytest.raises(SimulationError, match="boundaries"):
        await cohort.save(
            tmp_path / "wrong-window.json",
            start_ns=1,
            end_ns=2,
            benchmark_id="bench",
            phase_index=0,
        )


def cohort_configuration():
    run, timing = configuration(
        expected_duration_sec=1,
        grace_period_sec=float("inf"),
        drain_admitted_workflows=True,
    )
    run.cfg.get_profiling_phases = lambda: [SimpleNamespace(burst_phase_starts=True)]
    timing = timing.model_copy(
        update=dict(trajectory_start_min_ratio=0, trajectory_start_max_ratio=0)
    )
    return run, timing


def test_full_workflow_policy_is_opt_in() -> None:
    run, timing = cohort_configuration()
    validate_simulation(run, timing, "tcp://127.0.0.1:30001")
    with pytest.raises(NotImplementedError, match="fresh|trajectory-start"):
        validate_simulation(
            run,
            timing.model_copy(update=dict(trajectory_start_max_ratio=0.5)),
            "tcp://127.0.0.1:30001",
        )
    run.cfg.get_profiling_phases = lambda: [SimpleNamespace(burst_phase_starts=False)]
    with pytest.raises(NotImplementedError, match="burst-phase"):
        validate_simulation(run, timing, "tcp://127.0.0.1:30001")


@pytest.mark.parametrize('fields',[
    param(dict(total_expected_requests=1),id='request-cap'),
    param(dict(expected_num_sessions=1),id='session-cap'),
    param(dict(grace_period_sec=1),id='finite-grace'),
    param(dict(expected_duration_sec=None),id='no-duration'),
    param(dict(expected_duration_sec=float('inf')),id='infinite-duration'),
])  # fmt: skip
def test_truncating_policies_are_rejected(fields: dict) -> None:
    run, timing = cohort_configuration()
    phase = timing.phase_configs[0].model_copy(update=fields)
    timing = timing.model_copy(update=dict(phase_configs=[phase]))
    with pytest.raises(NotImplementedError, match="workflow cohort requires"):
        validate_simulation(run, timing, "tcp://127.0.0.1:30001")
