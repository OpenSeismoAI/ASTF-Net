import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch_job(tmp_path: Path) -> Callable:
    project = tmp_path / "project with spaces"
    project.mkdir()
    shutil.copyfile(REPO / "pyproject.toml", project / "pyproject.toml")
    (project / "astfnet").mkdir()
    (project / "config").mkdir()
    (project / "config/config-transformer.yaml").write_text("model_name: transformer\n")
    (project / "custom config.yaml").write_text("model_name: simplecnn\n")
    (project / "data with spaces").mkdir()
    bin_dir = project / ".venv/bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    print(os.environ.get('JOB_TEST_PYTHON', '3.12.10'))\n"
        "else:\n"
        "    pathlib.Path('launch.json').write_text(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
        "    sys.exit(int(os.environ.get('JOB_TEST_EXIT', '0')))\n"
    )
    python.chmod(0o755)
    cli = bin_dir / "astfnet-train"
    cli.write_text("#!/bin/sh\nexit 99\n")
    cli.chmod(0o755)
    # Slurm executes a copied script; its directory is not the repository root.
    script = tmp_path / "slurm spool script"
    shutil.copyfile(REPO / "job.sh", script)

    def run(*args: str, cwd: Path | None = None, **environment: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(script), *args],
            cwd=cwd or project,
            env={**os.environ, **environment},
            capture_output=True,
            text=True,
            timeout=10,
        )

    return run


@pytest.mark.parametrize("custom,exit_code", [(False, 0), (True, 37)])
def test_job_launch(launch_job: Callable, tmp_path: Path, custom: bool, exit_code: int) -> None:
    project = tmp_path / "project with spaces"
    config = str(project / "custom config.yaml") if custom else "config/config-transformer.yaml"
    extra = [config, "--skip-test"] if custom else []
    result = launch_job("data with spaces", *extra, JOB_TEST_EXIT=str(exit_code))
    assert result.returncode == exit_code, result.stderr
    launch = json.loads((project / "launch.json").read_text())
    assert launch == {
        "cwd": str(project),
        "args": ["-m", "astfnet.cli.train", "--config", config, "--data", "data with spaces"]
        + (["--skip-test"] if custom else []),
    }


@pytest.mark.parametrize(
    "problem,message",
    [
        ("arguments", "Usage:"),
        ("root", "repository root"),
        ("data", "Data directory"),
        ("config", "Config file"),
        ("python", "uv sync --frozen"),
        ("entrypoint", "uv sync --frozen"),
        ("version", "Python 3.12.10"),
    ],
)
def test_job_rejects_invalid_setup(launch_job: Callable, tmp_path: Path, problem: str, message: str) -> None:
    project = tmp_path / "project with spaces"
    args = ["data with spaces"]
    cwd = project
    environment = {}
    if problem == "arguments":
        args = []
    elif problem == "root":
        cwd = tmp_path
    elif problem == "data":
        args = ["missing data"]
    elif problem == "config":
        args.append("missing config.yaml")
    elif problem in {"python", "entrypoint"}:
        filename = "python" if problem == "python" else "astfnet-train"
        (project / ".venv/bin" / filename).unlink()
    elif problem == "version":
        environment["JOB_TEST_PYTHON"] = "3.10.16"
    result = launch_job(*args, cwd=cwd, **environment)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (project / "launch.json").exists()
