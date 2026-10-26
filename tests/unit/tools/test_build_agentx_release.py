# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The core checkout and isolated AgentX release retain distinct identities."""

import subprocess
from importlib.metadata import version
from pathlib import Path

import pytest
import tomlkit

import aiperf
from tools.build_agentx_release import prepare_source, release_version


def _tracked_source(tmp_path: Path, lookup: str) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    root = Path(__file__).resolve().parents[3]
    (source / "pyproject.toml").write_bytes((root / "pyproject.toml").read_bytes())
    init = source / "src/aiperf/__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text(
        "from importlib.metadata import version\n" + lookup + "\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release control",
            "-c",
            "user.email=release-control@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Tracked release fixture",
        ],
        cwd=source,
        check=True,
        capture_output=True,
    )
    return source


def test_core_version_uses_installed_core_distribution() -> None:
    assert aiperf.__version__ == version("aiperf")


@pytest.mark.parametrize("distribution", ["aiperf", "agentx-harness"])
def test_release_adapter_preserves_core_and_rebinds_only_copy(
    tmp_path: Path, distribution: str
) -> None:
    source = _tracked_source(tmp_path, f'__version__ = version("{distribution}")')
    init = source / "src/aiperf/__init__.py"
    original_init = init.read_bytes()
    original_project = (source / "pyproject.toml").read_bytes()
    destination = tmp_path / "release"

    prepare_source(source, destination, release_version("agentx-v1.0.12"))

    assert init.read_bytes() == original_init
    assert (source / "pyproject.toml").read_bytes() == original_project
    project = tomlkit.parse((destination / "pyproject.toml").read_text())["project"]
    assert project["name"] == "agentx-harness"
    assert project["version"] == "1.0.12"
    assert project["scripts"]["agentx"] == "agentx_cli:main"
    release_init = (destination / "src/aiperf/__init__.py").read_text()
    assert release_init.count('version("agentx-harness")') == 1
    assert 'version("aiperf")' not in release_init
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    assert (destination / "src/aiperf/_build_info.py").read_text() == (
        f'COMMIT_SHA = "{commit}"\n'
    )


@pytest.mark.parametrize(
    "lookup",
    [
        '__version__ = "unknown"',
        '__version__ = version("aiperf") + version("agentx-harness")',
        '__version__ = version("aiperf") + version("aiperf")',
    ],
)
def test_release_adapter_rejects_missing_or_ambiguous_lookup(
    tmp_path: Path, lookup: str
) -> None:
    source = _tracked_source(tmp_path, lookup)
    with pytest.raises(ValueError, match="version lookup changed"):
        prepare_source(source, tmp_path / "release", "1.0.12")
