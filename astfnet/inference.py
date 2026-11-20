"""Load registered backbones and stream paired HDF5 waveforms through them."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from astfnet.data_io.dataset import signed_log_inverse, signed_log_normalize
from astfnet.models.backbone import build_backbone


def sha256_file(path: Path) -> str:
    """Hash a file without retaining its contents in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_inference_model(
    checkpoint: Path, config_path: Path | None = None, device: str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a Lightning or backbone state_dict with strict architecture matching."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state_dict mapping")
    parameters = payload.get("hyper_parameters", {})
    if not isinstance(parameters, dict):
        parameters = OmegaConf.to_container(OmegaConf.create(parameters), resolve=True)
    if isinstance(parameters, dict) and "config" in parameters:
        parameters = parameters["config"]
    config_object = OmegaConf.create(parameters)
    if config_path is not None:
        config_object = OmegaConf.merge(config_object, OmegaConf.load(config_path))
    resolved = OmegaConf.to_container(config_object, resolve=True)
    if not isinstance(resolved, dict) or "model_name" not in resolved:
        raise ValueError("Checkpoint has no model_name; supply the matching training YAML with --config")
    config = cast(dict[str, Any], resolved)
    for key in ("log_normalize_input", "log_normalize_astf"):
        config.setdefault(key, True)
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a YAML boolean")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, cuda:INDEX, or auto")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    model = build_backbone(config)
    state = payload.get("state_dict", payload)
    # All three reference repositories use self.model for their Lightning wrapper.
    # A bare backbone state_dict is also accepted; other prefixes are not guessed.
    if state and all(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(target_device).eval(), config


def inspect_input(path: Path, dt: float | None = None) -> dict[str, Any]:
    """Validate paired inputs and optional raw ASTF labels before writing output."""
    with h5py.File(path, "r") as stream:
        for key in ("target_waveforms", "egfs"):
            if key not in stream or not isinstance(stream[key], h5py.Dataset):
                raise ValueError(f"{path}: missing dataset {key}")
        shape = stream["target_waveforms"].shape
        if len(shape) != 2 or not all(shape) or stream["egfs"].shape != shape:
            raise ValueError(f"{path}: target_waveforms and egfs must have the same nonempty (N, T) shape")
        labels = "astfs" in stream
        if labels and (len(stream["astfs"].shape) != 2 or stream["astfs"].shape[0] != shape[0]):
            raise ValueError(f"{path}: astfs must have shape (N, output_length)")
        for key in ("target_waveforms", "egfs", *(["astfs"] if labels else [])):
            if stream[key].dtype.kind not in "fiu":
                raise ValueError(f"{path}: {key} must contain real numeric values")
        interval = dt if dt is not None else stream.attrs.get("dt")
        if interval is not None and (not np.isfinite(interval) or interval <= 0):
            raise ValueError("dt must be a finite positive sample interval in seconds")
        return {"count": shape[0], "input_length": shape[1], "has_labels": labels, "dt": interval}


def predict_hdf5(
    model: torch.nn.Module,
    config: dict[str, Any],
    input_path: Path,
    output_path: Path,
    checkpoint: Path,
    batch_size: int = 128,
    dt: float | None = None,
) -> Path:
    """Stream every source row once and atomically publish physical ASTF predictions.

    Inputs use the same signed-log transform as training, without augmentation.
    ``predicted`` is inverse-transformed when ASTFs were trained in log space;
    optional ``actual`` contains the original raw labels. Existing files are never
    replaced. No waveform-sized array for the entire input is created.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    info = inspect_input(input_path, dt)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    device = next(model.parameters()).device
    model.eval()
    try:
        with h5py.File(input_path, "r") as source, h5py.File(temporary, "w") as destination:
            destination.attrs.update(
                source=str(input_path.resolve()),
                checkpoint=str(checkpoint.resolve()),
                checkpoint_sha256=sha256_file(checkpoint),
                config_json=json.dumps(config, sort_keys=True),
                device=str(device),
                prediction_space="physical_astf",
                sample_index_definition="zero-based row in source HDF5",
                status="running",
            )
            if info["dt"] is not None:
                destination.attrs["dt"] = info["dt"]
                destination.attrs["dt_source"] = "CLI" if dt is not None else "source HDF5 attribute"
            predicted_ds = actual_ds = None
            indices = destination.create_dataset("sample_index", (info["count"],), dtype="i8")
            with torch.inference_mode():
                for start in range(0, info["count"], batch_size):
                    stop = min(start + batch_size, info["count"])
                    inputs = []
                    for name in ("target_waveforms", "egfs"):
                        tensor = torch.as_tensor(source[name][start:stop], dtype=torch.float32, device=device)
                        if not torch.isfinite(tensor).all():
                            raise ValueError(f"{input_path}: nonfinite {name} in rows [{start}, {stop})")
                        if config.get("log_normalize_input", True):
                            tensor = signed_log_normalize(tensor)
                        inputs.append(tensor)
                    predicted = model(*inputs)
                    if config.get("log_normalize_astf", True):
                        predicted = signed_log_inverse(predicted)
                    if predicted.ndim != 2 or predicted.shape[0] != stop - start or not torch.isfinite(predicted).all():
                        raise ValueError(f"Invalid model output in rows [{start}, {stop})")
                    predicted = predicted.cpu().numpy().astype(np.float32, copy=False)
                    if predicted_ds is None:
                        shape = (info["count"], predicted.shape[1])
                        chunks = (min(batch_size, info["count"]), predicted.shape[1])
                        predicted_ds = destination.create_dataset(
                            "predicted", shape, dtype="f4", chunks=chunks, compression="gzip"
                        )
                        if info["has_labels"]:
                            if source["astfs"].shape != shape:
                                raise ValueError("ASTF label length differs from model output length")
                            actual_ds = destination.create_dataset(
                                "actual", shape, dtype="f4", chunks=chunks, compression="gzip"
                            )
                    predicted_ds[start:stop] = predicted
                    indices[start:stop] = np.arange(start, stop)
                    if actual_ds is not None:
                        actual = np.asarray(source["astfs"][start:stop], dtype=np.float32)
                        if not np.isfinite(actual).all():
                            raise ValueError(f"{input_path}: nonfinite astfs in rows [{start}, {stop})")
                        actual_ds[start:stop] = actual
            destination.attrs["status"] = "completed"
        # Both files are on the same filesystem. link() refuses an existing final
        # path, including one created by another process after the initial check.
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path
