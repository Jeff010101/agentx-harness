# SPDX-License-Identifier: Apache-2.0
"""Run an explicitly prepared finite native workload without agentic replay."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
from array import array
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiofiles
import aiohttp
import orjson

from aiperf.common.path_safety import safe_read_template_path
from aiperf.simulation.fixed import FixedDriver, FixedPlan
from aiperf.simulation.protocol import CoordinatorClient, validate_endpoint


def validate_url(value: str) -> str:
    """Keep native token-ID requests in the coordinator's local namespace."""
    parsed = urlsplit(value)
    try:
        loopback = (
            parsed.hostname == "localhost"
            or ipaddress.ip_address(parsed.hostname or "").is_loopback
        )
        valid = (
            parsed.scheme in ("http", "https") and loopback and parsed.port is not None
        )
    except ValueError:
        valid = False
    if (
        not valid
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "fixed simulator URL must be a loopback HTTP(S) origin with an explicit port"
        )
    return value.rstrip("/")


async def execute(
    plan: FixedPlan, prompts: list[array], *, url: str, coordinator: str, output: Path
) -> dict[str, Any]:
    """Persist raw rounds and native snapshots, retaining failures without retry."""
    plan.validate_inputs(prompts)
    url = validate_url(url)
    validate_endpoint(coordinator)
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    async with aiofiles.open(output / "plan.json", "xb") as handle:
        await handle.write(orjson.dumps(plan.model_dump(), option=orjson.OPT_INDENT_2))
    client = CoordinatorClient(coordinator)
    try:
        async with (
            aiofiles.open(output / "events.jsonl", "xb") as events,
            aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, limit_per_host=0),
                timeout=aiohttp.ClientTimeout(total=plan.wall_timeout_seconds),
            ) as http,
        ):

            async def emit(row: dict[str, Any]) -> None:
                await events.write(orjson.dumps(row) + b"\n")
                await events.flush()

            async def post(
                rid: str, ids: Any, sampling: dict[str, Any]
            ) -> dict[str, Any]:
                body = orjson.dumps(
                    {
                        "rid": rid,
                        "input_ids": list(ids),
                        "stream": False,
                        "sampling_params": sampling,
                    }
                )
                async with http.post(
                    url + "/generate",
                    data=body,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    return orjson.loads(await response.read())

            result = await FixedDriver(client, post, emit, plan).run(prompts)
        async with aiofiles.open(output / "result.json", "xb") as handle:
            await handle.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))
        return result
    except BaseException as error:
        async with aiofiles.open(output / "failure.json", "xb") as handle:
            await handle.write(
                orjson.dumps(
                    {
                        "completed": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "automatic_retry": False,
                    }
                )
            )
        raise
    finally:
        client.close()


async def _main(args: argparse.Namespace) -> None:
    plan_raw = await asyncio.to_thread(safe_read_template_path, str(args.plan))
    bank_raw = await asyncio.to_thread(safe_read_template_path, str(args.requests))
    if plan_raw is None or bank_raw is None:
        raise ValueError(
            "plan and requests must be readable regular files without symlink components"
        )
    plan = FixedPlan.model_validate(orjson.loads(plan_raw))
    prompts = []
    for line in bank_raw.splitlines():
        values = orjson.loads(line)
        if (
            not isinstance(values, dict)
            or set(values) != {"input_ids"}
            or not isinstance(values["input_ids"], list)
        ):
            raise ValueError(
                "each prompt-bank row must contain only an input_ids array"
            )
        ids = values["input_ids"]
        if not ids or any(
            type(token) is not int or not 0 <= token < 2**32 for token in ids
        ):
            raise ValueError("prompt-bank token IDs must be uint32 integers")
        prompts.append(array("I", ids))
    result = await execute(
        plan, prompts, url=args.url, coordinator=args.coordinator, output=args.output
    )
    print(orjson.dumps(result).decode())


def main() -> None:
    """Explicit module CLI; no profile flag or existing scenario is reinterpreted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
