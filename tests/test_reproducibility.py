import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from astfnet.constants import resolve_data_paths
from astfnet.reproducibility import configure_reproducibility, file_record
from examples.prepare_repro_smoke import prepare_data

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed,deterministic", [(0, True), (42, False)])
def test_seed_controls(seed: int, deterministic: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PL_GLOBAL_SEED", "99")
    monkeypatch.setenv("PL_SEED_WORKERS", "0")
    config = {"seed": 123, "deterministic": not deterministic}
    configure_reproducibility(config, seed, deterministic)
    first = (random.random(), np.random.random(), torch.rand(3))
    configure_reproducibility(config, seed, deterministic)
    assert random.random() == first[0]
    assert np.random.random() == first[1]
    assert torch.equal(torch.rand(3), first[2])
    assert torch.initial_seed() == seed
    assert config["seed"] == seed and config["deterministic"] is deterministic
    assert os.environ["PL_SEED_WORKERS"] == "1"


@pytest.mark.parametrize(
    "config",
    [{"seed": -1}, {"seed": 2**32}, {"seed": True}, {"deterministic": "false"}, {"benchmark": True}],
)
def test_invalid_controls(config: dict) -> None:
    with pytest.raises(ValueError):
        configure_reproducibility(config, None, None)


def test_file_checksum(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"first")
    before = file_record(path)
    assert before["sha256"] == hashlib.sha256(b"first").hexdigest()
    path.write_bytes(b"other")
    assert file_record(path)["sha256"] != before["sha256"]


def test_seeded_cpu_cli_runs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    prepare_data(data_dir)
    with pytest.raises(FileExistsError):
        prepare_data(data_dir)
    records = []
    weights = []
    for run in range(2):
        config = OmegaConf.load(REPO / "config/config-repro-smoke.yaml")
        config.tb_output_dir = str(tmp_path / f"run-{run}")
        # Exercise CLI overrides as well as config defaults in real subprocesses.
        config.seed = 999
        config.deterministic = False
        if run:
            for setting in ("seed", "deterministic", "benchmark"):
                del config[setting]
        config_path = tmp_path / f"config-{run}.yaml"
        OmegaConf.save(config, config_path)
        command = [
            sys.executable,
            "-m",
            "astfnet.cli.train",
            "--config",
            str(config_path),
            "--data",
            str(data_dir),
            "--dataset-revision",
            "synthetic-cli-test",
        ]
        if not run:
            command.extend(["--seed", "42", "--deterministic"])
        result = subprocess.run(
            command,
            cwd=REPO,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        run_dir = tmp_path / f"run-{run}/cpu/version_0"
        record = json.loads((run_dir / "manifest.json").read_text())
        assert record["status"] == "completed"
        assert record["config"]["seed"] == 42
        assert record["config"]["deterministic"] is True
        assert record["deterministic_algorithms"] is True
        assert record["cudnn_benchmark"] is False
        assert record["dataset"]["declared_revision"] == "synthetic-cli-test"
        assert record["global_step"] == 4
        assert record["dependency_lock"] == file_record(REPO / "uv.lock")
        assert (run_dir / "uv.lock").read_bytes() == (REPO / "uv.lock").read_bytes()
        assert OmegaConf.load(run_dir / "config.yaml").seed == 42
        assert record["git"]["commit"]
        paths = resolve_data_paths(str(data_dir))
        assert record["dataset"]["train"] == file_record(Path(paths["train_hdf5_file"]))
        assert record["dataset"]["validation"] == file_record(Path(paths["val_hdf5_file"]))
        assert record["dataset"]["test"] == [file_record(Path(path)) for path in paths["test_hdf5_files"]]
        assert record["hardware"]["device"] == "cpu"
        assert record["test_metrics"] and len(record["test_metrics"]) == 3
        assert all(np.isfinite(value) for value in [*record["fit_metrics"].values(), *record["test_metrics"].values()])
        checkpoint = record["checkpoints"]["last"]
        assert checkpoint == file_record(Path(checkpoint["path"]))
        weights.append(torch.load(checkpoint["path"], weights_only=False, map_location="cpu")["state_dict"])
        records.append(record)
    assert records[0]["fit_metrics"] == records[1]["fit_metrics"]
    assert records[0]["test_metrics"] == records[1]["test_metrics"]
    assert all(torch.equal(weights[0][key], weights[1][key]) for key in weights[0])


@pytest.mark.parametrize("fail_training", [False, True])
def test_manifest_skip_and_failure(tmp_path: Path, fail_training: bool) -> None:
    data_dir = tmp_path / "data"
    prepare_data(data_dir)
    for path in resolve_data_paths(str(data_dir))["test_hdf5_files"]:
        Path(path).unlink()
    config = OmegaConf.load(REPO / "config/config-repro-smoke.yaml")
    config.tb_output_dir = str(tmp_path / "run")
    if fail_training:
        config.callbacks.early_stopping.monitor = "missing_metric"
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "astfnet.cli.train",
            "--config",
            str(config_path),
            "--data",
            str(data_dir),
            "--skip-test",
            "--no-deterministic",
        ],
        cwd=REPO,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    record = json.loads((tmp_path / "run/cpu/version_0/manifest.json").read_text())
    assert record["skip_test"] is True
    assert record["dataset"]["test"] == [] and record["test_metrics"] == {}
    assert record["deterministic_algorithms"] is False
    if fail_training:
        assert result.returncode != 0
        assert "missing_metric" in result.stderr
        assert record["status"] == "failed" and record["exception"] == "RuntimeError"
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert record["status"] == "completed"
