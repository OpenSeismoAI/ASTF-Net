"""Seed training and record the inputs and outputs needed to repeat a run."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.version import cuda as cuda_build


def configure_reproducibility(config: dict[str, Any], seed: int | None, deterministic: bool | None) -> None:
    """Resolve and validate settings, then seed before data and model creation."""
    seed = config.get("seed", 42) if seed is None else seed
    deterministic = config.get("deterministic", True) if deterministic is None else deterministic
    benchmark = config.get("benchmark", False)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    if type(deterministic) is not bool or type(benchmark) is not bool:
        raise ValueError("deterministic and benchmark must be YAML booleans")
    if deterministic and benchmark:
        raise ValueError("deterministic training requires benchmark: false")
    config.update(seed=pl.seed_everything(seed, workers=True), deterministic=deterministic, benchmark=benchmark)


def file_record(path: Path) -> dict[str, Any]:
    """Describe an existing file using its absolute path, size and full SHA-256."""
    path = path.resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _metrics(trainer: pl.Trainer) -> dict[str, float]:
    """Convert scalar Lightning metrics to portable JSON numbers."""
    return {key: float(value.detach().cpu()) for key, value in trainer.callback_metrics.items()}


class RunManifest(pl.Callback):
    """Write provenance once before fitting, then append actual metrics and artifacts."""

    def __init__(self, config: dict[str, Any], data_paths: dict[str, Any], skip_test: bool) -> None:
        """Keep the effective configuration and only the files used in this run."""
        self.config = config
        self.data_paths = data_paths
        self.skip_test = skip_test
        self.record: dict[str, Any] = {}
        self.test_runs = 0

    def _save(self) -> None:
        """Atomically replace the run manifest."""
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Capture the source, environment, hardware and actual input checksums."""
        if not trainer.is_global_zero:
            return
        run_dir = Path(str(trainer.log_dir)).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "manifest.json"
        source_root = Path(__file__).resolve().parents[1]
        git: dict[str, Any] = {"commit": None, "dirty": None}
        if (source_root / ".git").exists():
            git["commit"] = subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
            ).strip()
            git["dirty"] = bool(subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"]))
        lock = Path(self.config.get("dependency_lock", source_root / "uv.lock"))
        lock_record = file_record(lock) if lock.is_file() else None
        if lock_record is not None:
            shutil.copyfile(lock, run_dir / "uv.lock")
        OmegaConf.save(OmegaConf.create(self.config), run_dir / "config.yaml", resolve=True)
        test_paths = [] if self.skip_test else self.data_paths["test_hdf5_files"]
        device = trainer.strategy.root_device
        self.record = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "fit_started",
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "config": self.config,
            "git": git,
            "source_files": {
                str(path.relative_to(source_root)): file_record(path)["sha256"]
                for path in sorted((source_root / "astfnet").rglob("*.py"))
            },
            "dependency_lock": lock_record,
            "packages": {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]},
            "hardware": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "machine": platform.machine(),
                "torch_threads": torch.get_num_threads(),
                "device": str(device),
                "cuda_build": cuda_build,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "dataset": {
                "declared_revision": self.config.get("dataset_revision"),
                "train": file_record(Path(self.data_paths["train_hdf5_file"])),
                "validation": file_record(Path(self.data_paths["val_hdf5_file"])),
                "test": [file_record(Path(path)) for path in test_paths],
            },
            "skip_test": self.skip_test,
            "test_metrics": {},
            "test_weights": "final in-memory model, not the best checkpoint",
            "run_dir": str(run_dir),
        }
        self._save()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record the final fit metrics and exact saved checkpoint files."""
        if not trainer.is_global_zero:
            return
        checkpoint = trainer.checkpoint_callback
        self.record.update(
            status="completed" if self.skip_test else "fit_completed",
            fit_metrics=_metrics(trainer),
            global_step=trainer.global_step,
            checkpoints={
                name: file_record(Path(path)) if path else None
                for name, path in {
                    "best": checkpoint.best_model_path if checkpoint else "",
                    "last": checkpoint.last_model_path if checkpoint else "",
                }.items()
            },
        )
        self._save()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Accumulate the metrics for each of the CLI's named test sets."""
        if not trainer.is_global_zero:
            return
        self.record["test_metrics"].update(_metrics(trainer))
        self.test_runs += 1
        if self.test_runs == len(self.data_paths["test_hdf5_files"]):
            self.record["status"] = "completed"
        self._save()

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        """Mark a started run as failed when Lightning observes an exception."""
        if trainer.is_global_zero and self.record:
            self.record.update(status="failed", exception=type(exception).__name__)
            self._save()
