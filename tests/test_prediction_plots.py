import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from astfnet.prediction_plots import (  # noqa: E402
    _center_waveforms,
    duration_at_cumulative_area,
    plot_prediction_report,
    prepare_duration_values,
    read_prediction_metrics,
    waveform_cc,
)


def test_waveform_display_centers_each_pair_and_zero_pads_missing_samples() -> None:
    actual = np.zeros((2, 201))
    predicted = np.zeros_like(actual)
    actual[0, 0] = 4
    predicted[0, 4] = 1
    actual[1, 80] = -0.1
    predicted[1, 84] = 0.2
    predicted[1, 180] = 0.0005  # Below 1% of this curve's own peak.
    time, actual_display, predicted_display = _center_waveforms(actual, predicted, 0.01)
    assert time[[0, -1]] == pytest.approx([-0.10, 0.10])
    np.testing.assert_array_equal(actual_display[:, :8], 0)
    np.testing.assert_array_equal(actual_display[:, 8], [4, -0.1])
    np.testing.assert_array_equal(predicted_display[:, 12], [1, 0.2])
    assert time[[8, 12]] == pytest.approx([-0.02, 0.02])


def write_predictions(path: Path, singleton: bool = False, interval: bool = True) -> Path:
    actual = np.array([[0, 1, 2, 1, 0], [0, 0, 1, 2, 1], [0, 0, 0, 0, 0]], dtype=float)
    predicted = actual * np.array([[1.1], [0.8], [1]])
    if singleton:
        actual, predicted = actual[:1, None, :], predicted[:1, None, :]
    with h5py.File(path, "w") as stream:
        stream["actual"], stream["predicted"] = actual, predicted
        if interval:
            stream.attrs["dt"] = 0.02
    return path


def test_duration_definition_and_interpolation() -> None:
    values = np.array([[0, 2, 2, 0], [0, -2, -2, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, np.nan, 2, 0]])
    duration = duration_at_cumulative_area(values, 0.1)
    np.testing.assert_allclose(duration[:3], [0.16, 0.16, 0])
    assert np.isnan(duration[3:]).all()
    assert duration_at_cumulative_area(np.ones((1, 10)), 0.1)[0] == pytest.approx(0.7)


def test_cc_undefined_is_not_silently_zero() -> None:
    values = np.array([[0, 1, 3], [0, 1, 3], [1, 1, 1], [0, np.nan, 1]])
    other = np.array([[1, 3, 7], [0, -1, -3], [0, 1, 2], [0, 1, 2]])
    cc = waveform_cc(values, other)
    np.testing.assert_allclose(cc[:2], [1, -1])
    assert np.isnan(cc[2:]).all()


def test_display_masks_do_not_change_population_denominators() -> None:
    metrics = {
        "path": "synthetic",
        "dt": 0.01,
        "dt_source": "fixture",
        "space_source": "fixture",
        "actual_duration": np.array([0.02, 0.1, 2.0, 0.5, np.nan]),
        "predicted_duration": np.array([0.02, 0.1, 2.0, 0.6, 0.4]),
        "cc": np.array([0.2, 0.85, 0.95, 1.0, np.nan]),
        "mse": np.array([0, 0, 0, 0, np.nan]),
    }
    summary = prepare_duration_values(metrics)["summary"]
    assert summary["sample_count"] == 5
    assert summary["duration_display_pair_count"] == 3
    assert summary["duration_metric_count"] == 3  # includes the 2-second pair
    assert summary["cc_finite_count"] == 4 and summary["cc_histogram_count"] == 3
    assert summary["cc_gt_0.9_fraction"] == 0.5
    assert summary["duration_relative_error_fractions"]["0.1"] == pytest.approx(2 / 3)
    json.dumps(summary, allow_nan=False)


def test_fit_r2_is_distinct_from_identity_r2() -> None:
    metrics = {
        "path": "synthetic",
        "dt": 0.1,
        "dt_source": "fixture",
        "space_source": "fixture",
        "actual_duration": np.array([0.1, 0.2, 0.3]),
        "predicted_duration": np.array([0.2, 0.4, 0.6]),
        "cc": np.ones(3),
        "mse": np.ones(3),
    }
    summary = prepare_duration_values(metrics)["summary"]
    assert summary["duration_r2_fit"] == pytest.approx(1)
    assert summary["duration_relative_error_fractions"]["0.5"] == 0


def test_single_record_legacy_shape_and_explicit_dt(tmp_path: Path) -> None:
    path = write_predictions(tmp_path / "legacy.h5", singleton=True, interval=False)
    with pytest.raises(ValueError, match="supply --dt"):
        read_prediction_metrics(path)
    metrics = read_prediction_metrics(path, dt=0.1, batch_size=1)
    assert metrics["cc"].shape == (1,)
    assert metrics["cc"][0] == pytest.approx(1)
    assert metrics["actual_duration"][0] == pytest.approx(0.22)
    assert metrics["dt_source"] == "CLI"


@pytest.mark.parametrize("problem", ["no_labels", "wrong_space", "shape", "dt"])
def test_invalid_plot_inputs(tmp_path: Path, problem: str) -> None:
    path = write_predictions(tmp_path / "bad.h5")
    with h5py.File(path, "a") as stream:
        if problem == "no_labels":
            del stream["actual"]
        elif problem == "wrong_space":
            stream.attrs["prediction_space"] = "signed_log"
        elif problem == "shape":
            del stream["actual"]
            stream["actual"] = np.ones((3, 2, 5))
        else:
            stream.attrs["dt"] = 0
    with pytest.raises(ValueError):
        read_prediction_metrics(path)


def test_report_artifacts_and_empty_metrics(tmp_path: Path) -> None:
    paths = [write_predictions(tmp_path / f"test_level_{level}.h5") for level in (1, 2, 3)]
    with h5py.File(paths[2], "a") as stream:
        stream["actual"][:] = 0
        stream["predicted"][:] = 0
    summaries = plot_prediction_report(paths, tmp_path / "report", dpi=35, examples=2)
    assert summaries[2]["cc_finite_count"] == 0
    assert summaries[2]["duration_metric_count"] == 0
    assert summaries[2]["duration_r2_fit"] is None
    for name in (
        "duration_cc_combined_2x3.pdf",
        "duration_cc_combined_2x3.png",
        "duration_cc_combined_2x3.preview.png",
        "examples_column_1.pdf",
        "examples_column_3.png",
    ):
        assert (tmp_path / "report" / name).stat().st_size > 1000
    assert json.loads((tmp_path / "report/metrics_summary.json").read_text()) == summaries


def test_unlabeled_five_column_report_splits_phases_with_source_row_mapping(tmp_path: Path) -> None:
    total = 40
    # Reorder prediction rows so their phase cannot be inferred from alternating indices.
    mapping = np.concatenate((np.arange(1, total, 2), np.arange(0, total, 2)))
    phase = ["P", "S"] * (total // 2)
    target = np.tile([0, 1, -1, 0], (total, 1))
    source, predictions = tmp_path / "inputs.h5", tmp_path / "predictions.h5"
    with h5py.File(source, "w") as stream:
        stream["phase"] = np.asarray(phase, dtype="S1")
        stream["target_waveforms"] = stream["egfs"] = target
    with h5py.File(predictions, "w") as stream:
        stream["sample_index"] = mapping
        stream["predicted"] = np.tile([1, 0, 0, 0], (total, 1))
        stream.attrs["dt"] = 0.01
    out = tmp_path / "plots"
    assert not plot_prediction_report([predictions], out, dpi=35, source_paths=[source])
    assert (out / "examples_column_1.pdf").is_file()
    assert not (out / "duration_cc_combined_2x3.pdf").exists()
    with (out / "examples_column_1_rows.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["phase"] for row in rows] == ["P"] * 15 + ["S"] * 15
    assert len({row["source_row"] for row in rows}) == 30
    for row in rows:
        assert row["phase"] == phase[int(row["source_row"])]
        assert not row["actual_d80_s"]
        assert float(row["reconstruction_cc"]) == pytest.approx(1.0)


def test_offline_plot_cli(tmp_path: Path) -> None:
    paths = [write_predictions(tmp_path / f"test_level_{level}.h5", interval=False) for level in (1, 2, 3)]
    before = [path.read_bytes() for path in paths]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "astfnet.cli.plot_predictions",
            "--input",
            *map(str, paths),
            "--output-dir",
            str(tmp_path / "report"),
            "--dt",
            "0.01",
            "--dpi",
            "35",
            "--examples",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=40,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "report/duration_cc_combined_2x3.pdf").exists()
    assert [path.read_bytes() for path in paths] == before
