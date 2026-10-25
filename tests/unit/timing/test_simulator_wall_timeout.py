# SPDX-License-Identifier: Apache-2.0
"""A physical control budget must not alter virtual request or phase timing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from aiperf.common.environment import Environment, _TimingSettings
from aiperf.common.messages import ProfileConfigureCommand
from aiperf.credit.messages import CreditReturn
from aiperf.simulation.identity import request_id
from aiperf.simulation.protocol import Grant, Ready, SimulationError, Terminal
from aiperf.simulation.router import SimulatorCreditRouter
from aiperf.timing.manager import TimingManager
from tests.unit.timing.test_simulator_router import Clock, credit


def test_wall_timeout_default_and_environment_override(monkeypatch) -> None:
    monkeypatch.delenv("AIPERF_TIMING_SIMULATOR_WALL_TIMEOUT", raising=False)
    assert _TimingSettings().SIMULATOR_WALL_TIMEOUT == 30.0
    monkeypatch.setenv("AIPERF_TIMING_SIMULATOR_WALL_TIMEOUT", "600")
    assert _TimingSettings().SIMULATOR_WALL_TIMEOUT == 600.0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "invalid"])
def test_wall_timeout_rejects_nonfinite_or_nonpositive(monkeypatch, value) -> None:
    monkeypatch.setenv("AIPERF_TIMING_SIMULATOR_WALL_TIMEOUT", value)
    with pytest.raises(ValidationError):
        _TimingSettings()


@pytest.mark.asyncio
async def test_timing_manager_passes_actual_environment_budget(monkeypatch) -> None:
    monkeypatch.setenv("AIPERF_TIMING_SIMULATOR_WALL_TIMEOUT", "600")
    monkeypatch.setenv("AIPERF_TIMING_SIMULATOR_COORDINATOR", "tcp://127.0.0.1:12345")
    monkeypatch.setattr(Environment, "TIMING", _TimingSettings())
    constructor = Mock()
    monkeypatch.setattr("aiperf.simulation.router.SimulatorCreditRouter", constructor)
    monkeypatch.setattr("aiperf.simulation.config.validate_simulation", Mock())
    monkeypatch.setattr("aiperf.simulation.completion.SimulationPhasePublisher", Mock())
    orchestrator = Mock(initialize=AsyncMock())
    monkeypatch.setattr(
        "aiperf.timing.manager.PhaseOrchestrator", Mock(return_value=orchestrator)
    )
    manager = SimpleNamespace(
        info=Mock(),
        debug=Mock(),
        _wait_for_dataset_or_failure=AsyncMock(),
        _dataset_failed_event=asyncio.Event(),
        _dataset_metadata=object(),
        sticky_router=Mock(),
        service_id="timing",
        pub_client=Mock(),
        config=SimpleNamespace(phase_configs=[object()]),
        run=SimpleNamespace(artifact_dir=None, benchmark_id="run"),
    )
    await TimingManager._profile_configure_command(
        manager,
        ProfileConfigureCommand.model_construct(service_id="controller", config={}),
    )
    assert constructor.call_args.kwargs["timeout_s"] == 600.0
    assert constructor.call_args.kwargs["endpoint"] == "tcp://127.0.0.1:12345"
    orchestrator.initialize.assert_awaited_once()


@pytest.mark.parametrize("budget, succeeds", [(0.001, False), (0.2, True)])
@pytest.mark.asyncio
async def test_delayed_control_return_preserves_virtual_terminal(
    monkeypatch, budget, succeeds
) -> None:
    client = Mock(request=AsyncMock())
    monkeypatch.setattr("aiperf.simulation.router.CoordinatorClient", lambda _: client)
    native = Mock(send_credit=AsyncMock())
    runtime = Clock()
    router = SimulatorCreditRouter(
        native,
        benchmark_id="run",
        endpoint="tcp://127.0.0.1:12345",
        runtime=runtime,
        timeout_s=budget,
    )
    item = credit(1)
    rid = request_id("run", 0, item.id)
    callbacks = []

    async def returned(worker, message) -> None:
        callbacks.append((message.credit.id, runtime.now_ns()))

    async def exchange(ready: Ready) -> Grant:
        if ready.round == 0:
            delay = asyncio.Event()
            handle = asyncio.get_running_loop().call_later(0.02, delay.set)
            try:
                await (
                    delay.wait()
                )  # Physical work, independent of the test sleep patch.
            finally:
                handle.cancel()
            assert runtime.now_ns() == 0
            await native.set_return_callback.call_args.args[0](
                "worker", CreditReturn(credit=item)
            )
            return Grant("run", 0, 123, [Terminal(rid, "COMPLETED", 123)])
        assert ready.finished
        return Grant("run", 1, 123, [], status="FINISHED")

    router.set_return_callback(returned)
    client.request.side_effect = exchange
    await router.send_credit(item)
    if succeeds:
        await router.drive()
        assert callbacks == [(1, 123)] and runtime.now_ns() == 123
        assert client.request.call_args_list[0].args[0].round == 0
    else:
        with pytest.raises(SimulationError, match="did not settle"):
            await router.drive()
        assert callbacks == [] and runtime.now_ns() == 0
    client.close.assert_called_once()
