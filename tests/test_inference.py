import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from astfnet.data_io.dataset import SeismicDatasetHDF5
from astfnet.inference import load_inference_model, predict_hdf5, sha256_file
from astfnet.models.backbone import build_backbone


def make_case(tmp_path: Path, model_name: str = "transformer", form: str = "flat", logarithmic: bool = True) -> tuple:
    config = {
        "model_name": model_name,
        "output_length": 256 if model_name == "unet1d" else 16,
        "log_normalize_input": logarithmic,
        "log_normalize_astf": logarithmic,
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 16,
        "cnn_kernel_size": 8,
        "cnn_stride": 8,
    }
    torch.manual_seed(17)
    model = build_backbone(config)
    state = {f"model.{key}": value for key, value in model.state_dict().items()}
    parameters = {"config": config} if form == "nested" else config
    payload = model.state_dict() if form == "bare" else {"state_dict": state, "hyper_parameters": parameters}
    checkpoint = tmp_path / "model.ckpt"
    torch.save(payload, checkpoint)
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    rng = np.random.default_rng(11)
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as stream:
        stream.create_dataset("target_waveforms", data=rng.normal(size=(5, 256)).astype("f4"))
        stream.create_dataset("egfs", data=rng.normal(size=(5, 256)).astype("f4"))
        stream.create_dataset("astfs", data=rng.normal(size=(5, config["output_length"])).astype("f4"))
        stream.attrs["dt"] = 0.02
    return checkpoint, config_path, path, model.eval(), config


@pytest.mark.parametrize(
    "model_name,form,logarithmic",
    [("simplecnn", "flat", True), ("transformer", "nested", False), ("unet1d", "bare", True)],
)
def test_streaming_matches_training_preprocessing(
    tmp_path: Path, model_name: str, form: str, logarithmic: bool
) -> None:
    checkpoint, config_path, source, original, config = make_case(tmp_path, model_name, form, logarithmic)
    model, resolved = load_inference_model(checkpoint, config_path if form == "bare" else None)
    output = predict_hdf5(model, resolved, source, tmp_path / "predictions.h5", checkpoint, batch_size=2)
    dataset = SeismicDatasetHDF5(str(source), log_normalize_input=logarithmic, log_normalize_astf=logarithmic)
    with torch.inference_mode():
        target = torch.stack([dataset[i]["target"] for i in range(len(dataset))])
        egf = torch.stack([dataset[i]["egf"] for i in range(len(dataset))])
        expected = original(target, egf)
        if logarithmic:
            expected = torch.sign(expected) * (torch.exp(expected.abs()) - 1)
    with h5py.File(output, "r") as stream, h5py.File(source, "r") as raw:
        np.testing.assert_allclose(stream["predicted"][:], expected.numpy(), rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(stream["actual"][:], raw["astfs"][:])
        np.testing.assert_array_equal(stream["sample_index"][:], np.arange(5))
        assert stream.attrs["status"] == "completed"
        assert stream.attrs["prediction_space"] == "physical_astf"
        assert stream.attrs["checkpoint_sha256"] == sha256_file(checkpoint)
        assert json.loads(stream.attrs["config_json"])["model_name"] == config["model_name"]
        assert stream.attrs["dt"] == 0.02
    before = sha256_file(output)
    with pytest.raises(FileExistsError):
        predict_hdf5(model, resolved, source, output, checkpoint)
    assert sha256_file(output) == before


def test_unlabeled_and_cli(tmp_path: Path) -> None:
    checkpoint, _, source, _, _ = make_case(tmp_path)
    with h5py.File(source, "a") as stream:
        del stream["astfs"]
        del stream.attrs["dt"]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "astfnet.cli.predict",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "export"),
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--dt",
            "0.01",
            "--plot-summary",
            "--plot-dpi",
            "35",
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(tmp_path / "export/input_predictions.h5", "r") as stream:
        assert "actual" not in stream
        assert stream["predicted"].shape == (5, 16)
        assert stream.attrs["dt"] == 0.01 and stream.attrs["dt_source"] == "CLI"
    assert (tmp_path / "export/plots/examples_column_1.pdf").exists()
    assert not (tmp_path / "export/plots/duration_cc_combined_2x3.pdf").exists()


@pytest.mark.parametrize("problem", ["nonfinite", "unpaired", "label_length", "batch_size"])
def test_bad_input_does_not_publish_output(tmp_path: Path, problem: str) -> None:
    checkpoint, _, source, _, _ = make_case(tmp_path)
    with h5py.File(source, "a") as stream:
        if problem == "nonfinite":
            stream["target_waveforms"][4, 0] = np.nan
        elif problem == "unpaired":
            del stream["egfs"]
        elif problem == "label_length":
            del stream["astfs"]
            stream.create_dataset("astfs", data=np.ones((5, 12)))
    model, config = load_inference_model(checkpoint)
    output = tmp_path / "predictions.h5"
    with pytest.raises(ValueError):
        predict_hdf5(model, config, source, output, checkpoint, batch_size=0 if problem == "batch_size" else 2)
    assert not output.exists()
    assert not list(tmp_path.glob(".predictions.h5.*.tmp"))


def test_incompatible_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint, config_path, _, _, _ = make_case(tmp_path)
    config = OmegaConf.load(config_path)
    config.output_length = 17
    OmegaConf.save(config, config_path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_inference_model(checkpoint, config_path)


def test_three_level_prediction_with_plot_summary(tmp_path: Path) -> None:
    checkpoint, _, source, _, _ = make_case(tmp_path)
    paths = []
    for level in (1, 2, 3):
        path = tmp_path / f"test_level_{level}.h5"
        path.write_bytes(source.read_bytes())
        paths.append(path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "astfnet.cli.predict",
            "--checkpoint",
            str(checkpoint),
            "--input",
            *map(str, paths),
            "--output-dir",
            str(tmp_path / "export"),
            "--plot-summary",
            "--plot-dpi",
            "35",
            "--batch-size",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summaries = json.loads((tmp_path / "export/plots/metrics_summary.json").read_text())
    assert [item["sample_count"] for item in summaries] == [5, 5, 5]
    assert (tmp_path / "export/plots/duration_cc_combined_2x3.pdf").exists()
