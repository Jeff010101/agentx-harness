# SPDX-License-Identifier: Apache-2.0
"""Real HTTP/ZMQ exchanges validate the finite client's two-channel barrier."""

from __future__ import annotations

import asyncio
from array import array
from pathlib import Path
from typing import Any

import msgspec
import orjson
import pytest
import zmq
import zmq.asyncio
from aiohttp import web

from aiperf.simulation.fixed import FixedPhase, FixedPlan
from aiperf.simulation.fixed_cli import execute
from aiperf.simulation.protocol import Grant, Ready, Terminal


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_native_http_waits_for_ready_and_preserves_profiling_population(
    tmp_path: Path,
) -> None:
    """Exercise real sockets without treating the test peer as silicon evidence."""
    received: dict[str, tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]] = {}
    arrived = asyncio.Event()
    submitted: list[list[int]] = []

    async def generate(request: web.Request) -> web.Response:
        body = await request.json()
        assert body["stream"] is False and body["sampling_params"]["ignore_eos"] is True
        future = asyncio.get_running_loop().create_future()
        received[body["rid"]] = (body, future)
        submitted.append(body["input_ids"])
        arrived.set()
        return web.json_response(await future)

    app = web.Application()
    app.router.add_post("/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    control_port = socket.bind_to_random_port("tcp://127.0.0.1")

    async def server() -> None:
        pending: dict[str, int] = {}
        now = 0
        while True:
            ready = msgspec.convert(
                orjson.loads(await socket.recv()), type=Ready, strict=True
            )
            if ready.finished:
                assert not pending
                await socket.send(
                    orjson.dumps(
                        msgspec.to_builtins(
                            Grant(ready.run_id, ready.round, now, [], "FINISHED")
                        )
                    )
                )
                return
            for op in ready.operations:
                pending[op.rid] = now
            while any(rid not in received for rid in pending):
                arrived.clear()
                await arrived.wait()
            rid = next(iter(pending))
            arrival = pending.pop(rid)
            body, future = received[rid]
            now += 1_000_000
            count = body["sampling_params"]["max_new_tokens"]
            snapshot = dict(
                version=1,
                rid=rid,
                arrival_ns=arrival,
                first_token_ns=now - 800_000,
                last_token_ns=now - 100_000,
                commit_ns=now,
                terminal_ns=now,
                prompt_tokens=len(body["input_ids"]),
                completion_tokens=count,
                cached_tokens=0,
            )
            future.set_result(
                {
                    "meta_info": {
                        "sglang_simulator": snapshot,
                        "prompt_tokens": len(body["input_ids"]),
                        "completion_tokens": count,
                    }
                }
            )
            await socket.send(
                orjson.dumps(
                    msgspec.to_builtins(
                        Grant(
                            ready.run_id,
                            ready.round,
                            now,
                            [Terminal(rid, "COMPLETED", now)],
                        )
                    )
                )
            )

    peer = asyncio.create_task(server())
    plan = FixedPlan(
        run_id="wire-control",
        phases=[
            FixedPhase(
                name="warmup",
                concurrency=2,
                request_indices=[2, 2],
                input_tokens=3,
                output_tokens=2,
            ),
            FixedPhase(
                name="profiling",
                concurrency=2,
                request_indices=[0, 1],
                input_tokens=3,
                output_tokens=2,
                expected_cached_tokens=0,
            ),
        ],
    )
    output = tmp_path / "new-run"
    try:
        result = await asyncio.wait_for(
            execute(
                plan,
                [array("I", [1, 2, 3]), array("I", [4, 5, 6]), array("I", [7, 8, 9])],
                url=f"http://127.0.0.1:{port}",
                coordinator=f"tcp://127.0.0.1:{control_port}",
                output=output,
            ),
            10,
        )
        await asyncio.wait_for(peer, 2)
    finally:
        peer.cancel()
        await asyncio.gather(peer, return_exceptions=True)
        socket.close()
        context.term()
        await runner.cleanup()
    assert result["phase_counts"] == {"warmup": 2, "prepare": 0, "profiling": 2}
    assert result["profiling"]["input_tokens"] == 6
    assert submitted[:2] == [[7, 8, 9]] * 2
    assert submitted[2:] == [[1, 2, 3], [4, 5, 6]]
    assert (output / "events.jsonl").is_file() and (output / "result.json").is_file()
    assert not (output / "failure.json").exists()
