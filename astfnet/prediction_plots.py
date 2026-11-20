"""ASTF diagnostics adapted from the project's three-level 2-by-3 evaluation plots."""

import csv
import json
import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator


def _matrix(values: np.ndarray) -> np.ndarray:
    """Keep the record dimension, including a single (N, 1, T) waveform."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("Waveforms must have shape (N, T) or (N, 1, T)")
    return values


def duration_at_cumulative_area(waveforms: np.ndarray, dt: float, fraction: float = 0.8) -> np.ndarray:
    """Compute time from zero to cumulative absolute area, using the reference interpolation.

    This is neither a central 80% interval nor an energy/squared-amplitude
    duration. Zero-area and nonfinite waveforms have undefined (NaN) duration.
    """
    if not np.isfinite(dt) or dt <= 0 or not 0 < fraction <= 1:
        raise ValueError("dt must be finite and positive, and fraction must be in (0, 1]")
    values = _matrix(waveforms)
    cumulative = np.cumsum(np.abs(values), axis=1)
    total = cumulative[:, -1]
    target = fraction * total
    index = np.clip(np.sum(cumulative < target[:, None], axis=1), 0, values.shape[1] - 1)
    rows = np.arange(len(values))
    previous = np.where(index > 0, cumulative[rows, np.maximum(index - 1, 0)], 0.0)
    increment = cumulative[rows, index] - previous
    within = np.divide(target - previous, increment, out=np.zeros_like(total), where=increment > 0)
    duration = np.full(len(values), np.nan)
    valid = np.isfinite(total) & (total > 0)
    duration[valid] = (np.maximum(index[valid] - 1, 0) + within[valid]) * dt
    duration[valid & (index == 0)] = 0.0
    return duration


def waveform_cc(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return a Pearson CC per waveform pair, leaving constant/nonfinite pairs undefined."""
    actual, predicted = _matrix(actual), _matrix(predicted)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted shapes must match")
    left = actual - actual.mean(axis=1, keepdims=True)
    right = predicted - predicted.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cc = np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.full(len(actual), np.nan),
        where=np.isfinite(denominator) & (denominator > 0),
    )
    return np.clip(cc, -1, 1)


def _plot_interval(stream: h5py.File, path: Path, dt: float | None) -> float:
    if stream.attrs.get("prediction_space", "physical_astf") != "physical_astf":
        raise ValueError(f"{path}: plotting requires predictions in physical ASTF space")
    interval = dt if dt is not None else stream.attrs.get("dt")
    if interval is None or not np.isfinite(interval) or interval <= 0:
        raise ValueError(f"{path}: supply --dt in seconds or a positive HDF5 dt attribute")
    return float(interval)


def read_prediction_metrics(path: Path, dt: float | None = None, batch_size: int = 4096) -> dict[str, Any]:
    """Read waveform batches, retaining only per-record scalar metrics."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    with h5py.File(path, "r") as stream:
        if "actual" not in stream or "predicted" not in stream:
            raise ValueError(f"{path}: labeled plotting requires actual and predicted datasets")
        for key in ("actual", "predicted"):
            dataset = stream[key]
            if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in "fiu":
                raise ValueError(f"{path}: {key} must be a real numeric dataset")
            shape = dataset.shape
            if not (len(shape) == 2 or (len(shape) == 3 and shape[1] == 1)) or not shape[-1]:
                raise ValueError(f"{path}: waveforms must have shape (N, T) or (N, 1, T)")
        if stream["actual"].shape != stream["predicted"].shape:
            raise ValueError(f"{path}: actual/predicted shapes differ")
        interval = _plot_interval(stream, path, dt)
        metrics: dict[str, list] = {key: [] for key in ("cc", "actual_duration", "predicted_duration", "mse")}
        count = len(stream["actual"])
        for start in range(0, count, batch_size):
            actual = _matrix(stream["actual"][start : start + batch_size])
            predicted = _matrix(stream["predicted"][start : start + batch_size])
            metrics["cc"].append(waveform_cc(actual, predicted))
            metrics["actual_duration"].append(duration_at_cumulative_area(actual, interval))
            metrics["predicted_duration"].append(duration_at_cumulative_area(predicted, interval))
            metrics["mse"].append(np.mean((actual - predicted) ** 2, axis=1))
        return {
            "path": str(path.resolve()),
            "dt": interval,
            "dt_source": "CLI" if dt is not None else stream.attrs.get("dt_source", "HDF5 attribute"),
            "space_source": "HDF5 attribute"
            if "prediction_space" in stream.attrs
            else "legacy physical-space contract",
            **{key: np.concatenate(values) if values else np.empty(0) for key, values in metrics.items()},
        }


def prepare_duration_values(metrics: dict[str, Any], min_duration: float = 0.03, limit: float = 1.0) -> dict[str, Any]:
    """Separate display bounds from the full-population duration metric mask."""
    if not np.isfinite(min_duration) or min_duration < 0 or not np.isfinite(limit) or limit <= 0:
        raise ValueError("Duration threshold must be nonnegative and plot limit positive")
    actual, predicted = metrics["actual_duration"], metrics["predicted_duration"]
    finite = np.isfinite(actual) & np.isfinite(predicted)
    visible = finite & (actual >= 0) & (predicted >= 0) & (actual <= limit) & (predicted <= limit)
    histogram, xedges, yedges = np.histogram2d(actual[visible], predicted[visible], bins=220, range=[[0, limit]] * 2)
    eligible = finite & (actual > 0) & (actual >= min_duration)
    x, y = actual[eligible], predicted[eligible]
    relative = np.abs(y - x) / x
    fractions = {str(value): float(np.mean(relative <= value)) if len(x) else None for value in (0.1, 0.2, 0.5)}
    # Same linear-fit R² as the reference, explicitly named to distinguish it
    # from an identity/predictive R² that penalizes bias against y=x.
    r2_fit = None
    if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
        r2_fit = float(waveform_cc(x[None, :], y[None, :])[0] ** 2)
    cc = metrics["cc"]
    valid_cc = cc[np.isfinite(cc)]
    mse = metrics["mse"]
    summary = {
        "source_prediction": metrics["path"],
        "dt_seconds": metrics["dt"],
        "dt_source": metrics["dt_source"],
        "space_source": metrics["space_source"],
        "area_fraction": 0.8,
        "sample_count": len(cc),
        "cc_finite_count": len(valid_cc),
        "cc_undefined_count": int(np.sum(~np.isfinite(cc))),
        "cc_histogram_count": int(np.sum((valid_cc >= 0.6) & (valid_cc <= 1))),
        "cc_below_display_count": int(np.sum(valid_cc < 0.6)),
        "cc_gt_0.9_fraction": float(np.mean(valid_cc > 0.9)) if len(valid_cc) else None,
        "cc_gt_0.8_fraction": float(np.mean(valid_cc > 0.8)) if len(valid_cc) else None,
        "physical_mse": float(np.mean(mse[np.isfinite(mse)])) if np.isfinite(mse).any() else None,
        "mse_finite_count": int(np.isfinite(mse).sum()),
        "duration_finite_pair_count": int(finite.sum()),
        "duration_display_pair_count": int(visible.sum()),
        "duration_metric_count": int(eligible.sum()),
        "duration_min_seconds": min_duration,
        "duration_display_limit_seconds": limit,
        "duration_relative_error_fractions": fractions,
        "duration_r2_fit": r2_fit,
    }
    return {"histogram": histogram, "xedges": xedges, "yedges": yedges, "summary": summary}


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def _draw_duration(ax: Axes, prepared: dict[str, Any], norm: LogNorm) -> None:
    summary = prepared["summary"]
    limit = summary["duration_display_limit_seconds"]
    counts = prepared["histogram"].T
    colors = plt.get_cmap("OrRd")(norm(np.maximum(counts, 1)))
    colors[..., 3] = np.where(counts > 0, 0.04 + 0.96 * norm(np.maximum(counts, 1)), 0)
    ax.imshow(colors, extent=(0, limit, 0, limit), origin="lower", aspect="equal", interpolation="nearest")
    x = np.array([0, limit])
    for error, style in zip((0.1, 0.2, 0.5), ("--", "-.", ":")):
        label = f"±{error:.0%}: {_percentage(summary['duration_relative_error_fractions'][str(error)])}"
        ax.plot(x, x * (1 + error), color="0.45", ls=style, lw=1.6, label=label)
        ax.plot(x, x * (1 - error), color="0.45", ls=style, lw=1.6)
    ax.plot(x, x, color="black", lw=1.7)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Actual duration (s)", ylabel="Predicted duration (s)")
    r2 = summary["duration_r2_fit"]
    ax.text(
        0.05,
        0.94,
        f"Fit $R^2$ = {r2:.3f}" if r2 is not None else "Fit $R^2$ = N/A",
        transform=ax.transAxes,
        va="top",
        fontsize=19,
    )
    ax.legend(loc="lower right", fontsize=16, frameon=False)
    if not summary["duration_display_pair_count"]:
        ax.text(0.5, 0.5, "No pairs within display range", ha="center", transform=ax.transAxes, fontsize=15)


def _draw_cc(ax: Axes, metrics: dict[str, Any], summary: dict[str, Any]) -> None:
    cc = metrics["cc"]
    ax.hist(cc[np.isfinite(cc)], bins=50, range=(0.6, 1), color="#377eb8", edgecolor="0.35", linewidth=0.6)
    ax.set(xlim=(0.6, 1), xlabel="ASTF correlation coefficient", ylabel="Count")
    ax.set_ylim(0, max(1, ax.get_ylim()[1]) * 1.25)
    ax.set_box_aspect(1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 4))
    ax.text(
        0.04,
        0.94,
        f"CC > 0.9: {_percentage(summary['cc_gt_0.9_fraction'])}\n"
        f"CC > 0.8: {_percentage(summary['cc_gt_0.8_fraction'])}",
        transform=ax.transAxes,
        va="top",
        fontsize=19,
    )
    if not summary["cc_histogram_count"]:
        ax.text(0.5, 0.45, "No CC within display range", ha="center", transform=ax.transAxes, fontsize=15)


def _save_figure(figure: Figure, destination: Path, dpi: int, preview: bool = False) -> None:
    try:
        for suffix in ("pdf", "png"):
            figure.savefig(destination.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
        if preview:
            figure.savefig(destination.with_suffix(".preview.png"), dpi=90, bbox_inches="tight")
    finally:
        plt.close(figure)


def _center_waveforms(
    actual: np.ndarray, predicted: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center each pair on its joint effective interval and pad unrecorded samples with zeros."""
    pairs = np.stack((actual, predicted), axis=1)
    magnitude = np.where(np.isfinite(pairs), np.abs(pairs), 0.0)
    peak = magnitude.max(axis=-1, keepdims=True)
    active = ((magnitude >= 0.01 * peak) & (magnitude > 0)).any(axis=1)
    last_sample = pairs.shape[-1] - 1
    has_signal = active.any(axis=1)
    first = active.argmax(axis=1)
    last = last_sample - active[:, ::-1].argmax(axis=1)
    centers = np.where(has_signal, (first + last) // 2, last_sample // 2)
    if has_signal.any():
        span = (last - centers)[has_signal].max()
        half_window = int(span) + max(2, int(np.ceil(0.08 / dt)))
    else:
        half_window = max(1, pairs.shape[-1] // 2)

    relative_samples = np.arange(-half_window, half_window + 1)
    source_samples = centers[:, None] + relative_samples
    recorded = (source_samples >= 0) & (source_samples <= last_sample)
    indices = source_samples.clip(0, last_sample)
    centered = np.take_along_axis(pairs, indices[:, None, :], axis=-1)
    centered = np.where(recorded[:, None, :], centered, 0.0)
    return relative_samples * dt, centered[:, 0, :], centered[:, 1, :]


def _plot_legacy_examples(path: Path, destination: Path, dt: float, count: int, dpi: int) -> None:
    """Keep the ASTF comparison available for legacy exports without paired waveforms."""
    with h5py.File(path, "r") as stream:
        total = len(stream["actual"])
        if not total or not count:
            return
        rows = np.unique(np.linspace(0, total - 1, min(count, total), dtype=int))
        actual, predicted = _matrix(stream["actual"][rows]), _matrix(stream["predicted"][rows])
        source_rows = stream["sample_index"][rows] if "sample_index" in stream else rows
    time, actual_display, predicted_display = _center_waveforms(actual, predicted, dt)
    waveforms = np.concatenate((actual_display, predicted_display), axis=0)
    finite = waveforms[np.isfinite(waveforms)]
    scale = float(np.max(np.abs(finite), initial=0)) or 1.0
    low = float(finite.min(initial=0)) / scale
    high = float(finite.max(initial=0)) / scale
    spacing = max(1.0, high - low) + 0.22
    offsets = np.arange(len(rows)) * spacing
    figure, ax = plt.subplots(figsize=(9.2, 1.55 + 0.82 * len(rows)), layout="constrained")
    ax.set_title("ASTF waveform comparison", loc="left", pad=26, fontsize=15)
    for index, (left, right, offset, cc) in enumerate(
        zip(actual_display, predicted_display, offsets, waveform_cc(actual, predicted))
    ):
        ax.axhline(offset, color="0.87", lw=0.7, zorder=0)
        ax.plot(time, left / scale + offset, color="#1c1c1c", lw=1.6, label="Actual" if index == 0 else None)
        ax.plot(
            time,
            right / scale + offset,
            color="#c12932",
            ls="--",
            lw=1.45,
            label="Predicted" if index == 0 else None,
        )
        cc_label = f"CC {cc:.3f}" if np.isfinite(cc) else "CC unavailable"
        ax.text(1.015, offset, cc_label, transform=ax.get_yaxis_transform(), va="center", fontsize=9, color="#505050")

    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(low - 0.20, offsets[-1] + high + 0.16)
    ax.set_yticks(offsets, labels=[f"Row {row:,}" for row in source_rows])
    ax.set_xlabel("Time from waveform pair center (s)", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="x", labelsize=10, length=3, color="0.55")
    ax.tick_params(axis="y", labelsize=10, length=0, pad=8)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("0.65")
    ax.grid(axis="x", color="0.91", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1, 1.015),
        ncols=2,
        frameon=False,
        fontsize=10,
        handlelength=2.6,
    )
    _save_figure(figure, destination, dpi)


def _even_rows(rows: np.ndarray, count: int) -> np.ndarray:
    return rows[np.linspace(0, len(rows) - 1, min(len(rows), count), dtype=int)]


def _reconstruct(
    target: np.ndarray, egf: np.ndarray, astf: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align each convolution to its target with at most 0.40 s of zero-padded shift."""
    radius = min(int(np.floor(0.40 / dt)), target.shape[1] + astf.shape[1] - 1)
    shifts = sorted(range(-radius, radius + 1), key=lambda shift: (abs(shift), shift))
    reconstructions, coefficients, lags = [], [], []
    for trace, kernel, prediction in zip(target, egf, astf):
        convolution = np.pad(np.convolve(kernel, prediction, mode="full"), radius)
        candidates = np.stack([convolution[radius - lag : radius - lag + len(trace)] for lag in shifts])
        cc = waveform_cc(np.broadcast_to(trace, candidates.shape), candidates)
        best = int(np.nanargmax(cc)) if np.isfinite(cc).any() else 0
        reconstructions.append(candidates[best])
        coefficients.append(cc[best])
        lags.append(shifts[best] * dt)
    return np.stack(reconstructions), np.array(coefficients), np.array(lags)


def _normalize_display(traces: np.ndarray) -> np.ndarray:
    peaks = np.abs(traces).max(axis=1, keepdims=True)
    return traces / np.where(peaks > 0, peaks, 1)


def _load_examples(
    prediction: h5py.File, source: h5py.File, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Read only the selected waveform rows, using phase metadata when available."""
    total = len(prediction["predicted"])
    target_ds, egf_ds = source["target_waveforms"], source["egfs"]
    if target_ds.shape != egf_ds.shape:
        raise ValueError(f"{source.filename}: target_waveforms and egfs shapes differ")
    if "sample_index" in prediction:
        source_indices = prediction["sample_index"][:]
        if source_indices.shape != (total,) or source_indices.dtype.kind not in "iu":
            raise ValueError("sample_index must contain one integer per prediction")
    elif len(target_ds) == total:
        source_indices = np.arange(total)
    else:
        raise ValueError("Prediction rows require a sample_index mapping to the source waveforms")
    # Embedded waveforms already follow prediction row order. Their sample_index
    # may still refer to the original, larger input file.
    input_indices = np.arange(total) if source is prediction else source_indices
    if np.any(source_indices < 0) or np.any(input_indices >= len(target_ds)):
        raise ValueError("sample_index refers to rows outside the source waveforms")

    phase_rows = []
    phases = np.full(total, "", dtype="U1")
    if "phase" in source:
        phase_ds = source["phase"]
        if phase_ds.shape != (len(target_ds),):
            raise ValueError("phase must contain one string per source row")
        labels = np.char.upper(np.char.strip(phase_ds.asstr()[:].astype(str)))[input_indices]
        phases[labels == "P"], phases[labels == "S"] = "P", "S"
        phase_rows = [np.flatnonzero(phases == phase) for phase in ("P", "S")]
    if count >= 2 and phase_rows and all(len(rows) for rows in phase_rows):
        selected = np.sort(np.concatenate([_even_rows(rows, count // 2) for rows in phase_rows]))
    else:
        selected = _even_rows(np.arange(total), count)
    rows = source_indices[selected]
    unique_rows, reverse = np.unique(input_indices[selected], return_inverse=True)
    target, egf = (_matrix(dataset[unique_rows])[reverse] for dataset in (target_ds, egf_ds))
    predicted = _matrix(prediction["predicted"][selected])
    actual = _matrix(prediction["actual"][selected]) if "actual" in prediction else None
    if actual is not None:
        if actual.shape != predicted.shape:
            raise ValueError("Actual and predicted shapes must match")
        if "astfs" in source and not np.allclose(
            actual, _matrix(source["astfs"][unique_rows])[reverse], equal_nan=True
        ):
            raise ValueError("Source ASTFs do not match the selected prediction rows")
    return rows, phases[selected], target, egf, predicted, actual


def _draw_five_columns(
    destination: Path,
    rows: np.ndarray,
    phases: np.ndarray,
    target: np.ndarray,
    egf: np.ndarray,
    predicted: np.ndarray,
    actual: np.ndarray | None,
    dt: float,
    dpi: int,
) -> None:
    predicted_d80 = duration_at_cumulative_area(predicted, dt)
    actual_d80 = duration_at_cumulative_area(actual, dt) if actual is not None else None
    reconstructed, cc, lag = _reconstruct(target, egf, predicted, dt)
    order = np.argsort(predicted_d80, kind="stable")
    split = set(phases) == {"P", "S"}
    groups = [order[phases[order] == phase] for phase in ("P", "S")] if split else [order]
    plot_order = np.concatenate(groups)
    y = np.arange(len(rows), 0, -1)
    with destination.with_name(f"{destination.name}_rows.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["index", "source_row", "phase", "reconstruction_cc", "shift_s", "actual_d80_s", "predicted_d80_s"]
        )
        for index, i in zip(y, plot_order):
            writer.writerow(
                [
                    index,
                    rows[i],
                    phases[i],
                    cc[i],
                    lag[i],
                    actual_d80[i] if actual_d80 is not None else "",
                    predicted_d80[i],
                ]
            )

    t_astf, truth_view, pred_view = _center_waveforms(actual if actual is not None else predicted, predicted, dt)
    t_astf -= t_astf[0]
    astf_peak = np.maximum(np.abs(truth_view).max(axis=1), np.abs(pred_view).max(axis=1))[:, None]
    astf_peak = np.where(astf_peak > 0, astf_peak, 1)
    target_view, recon_view, egf_view = map(_normalize_display, (target, reconstructed, egf))
    time = np.arange(target.shape[1]) * dt
    blue, red, black = "#2869b2", "#bd3940", "#252933"
    colors = {"P": blue, "S": red}
    with plt.rc_context({"font.size": 14, "pdf.fonttype": 42, "axes.edgecolor": "#8d96a3", "axes.linewidth": 0.65}):
        figure, axes = plt.subplots(
            1, 5, figsize=(16.5, 15.5), sharey=True, gridspec_kw={"width_ratios": [1, 0.24, 1, 0.57, 0.34]}
        )
        figure.subplots_adjust(left=0.055, right=0.985, bottom=0.06, top=0.955, wspace=0.13)
        for offset, i in zip(y, plot_order):
            color = colors[phases[i]] if split else blue
            pred_color = color if split else red
            axes[0].plot(time, offset + 0.39 * target_view[i], color=black, lw=0.88)
            axes[0].plot(time, offset + 0.39 * recon_view[i], color=color, lw=0.88, ls="--")
            axes[2].plot(time, offset + 0.39 * egf_view[i], color=color, lw=0.88)
            if actual is not None:
                axes[3].plot(t_astf, offset + 0.47 * truth_view[i] / astf_peak[i], color=black, lw=0.88)
            axes[3].plot(t_astf, offset + 0.47 * pred_view[i] / astf_peak[i], color=pred_color, lw=0.88, ls="--")
        start = 0
        for group in groups:
            indexes = y[start : start + len(group)]
            color = colors[phases[group[0]]] if split else blue
            axes[1].plot(cc[group], indexes, "o-", color=color, lw=0.85, ms=2.8)
            if actual_d80 is not None:
                axes[4].plot(actual_d80[group], indexes, "o-", color=black, lw=0.8, ms=2.7)
            axes[4].plot(predicted_d80[group], indexes, "o--", color=color if split else red, lw=0.8, ms=2.7)
            start += len(group)

        titles = ("(a) Target waveforms", "(b) CCs", "(c) EGF waveforms", "(d) ASTFs", "(e) Durations")
        for ax, title, xlabel in zip(axes, titles, ("Time (s)", "Coeff", "Time (s)", "Time (s)", "Time (s)")):
            ax.set_title(title, fontsize=19, pad=11)
            ax.set_xlabel(xlabel, fontsize=19, labelpad=5)
            ax.tick_params(length=3, pad=3)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes[0].set(ylim=(0.35, len(rows) + 2.65), yticks=np.arange(5, len(rows) + 1, 5) if len(rows) >= 5 else y)
        axes[0].set_ylabel("Index", fontsize=19)
        for ax, t in zip((axes[0], axes[2], axes[3]), (time, time, t_astf)):
            ax.set_xlim(0, max(dt, float(t[-1])))
            if t[-1] >= 1:
                ax.xaxis.set_major_locator(MultipleLocator(0.5))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        for ax in axes[1:]:
            ax.tick_params(axis="y", left=False)
        finite_cc = cc[np.isfinite(cc)]
        axes[1].set_xlim(min(0.95, max(-1.0, float(finite_cc.min(initial=1)) - 0.04)), 1.02)
        axes[1].xaxis.set_major_locator(MultipleLocator(0.3, offset=0.1))
        durations = predicted_d80 if actual_d80 is None else np.concatenate((predicted_d80, actual_d80))
        max_duration = durations[np.isfinite(durations)].max(initial=0)
        axes[4].set_xlim(0, max(dt, float(max_duration) * 1.08))
        axes[4].xaxis.set_major_locator(MaxNLocator(nbins=3))
        axes[4].xaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        for ax in (axes[1], axes[4]):
            ax.grid(axis="x", color="#e6ebf1", lw=0.65)
            ax.set_axisbelow(True)
        if split:
            separation = len(groups[1]) + 0.5
            for ax in axes:
                ax.axhline(separation, color="0.58", lw=0.85, ls="--")
            for phase, position in (("P", (len(rows) + separation + 0.5) / 2), ("S", separation / 2 + 0.25)):
                axes[0].text(
                    -0.105,
                    position,
                    phase,
                    transform=axes[0].get_yaxis_transform(),
                    ha="center",
                    va="center",
                    color=colors[phase],
                    fontsize=15,
                    fontweight="bold",
                )

        legend_entries = [("Target", black, "-")]
        astf_entries = [("Actual", black, "-")] if actual is not None else []
        if split:
            legend_entries.extend((f"{phase} reconstructed", color, "--") for phase, color in colors.items())
            astf_entries.extend((f"{phase} predicted", color, "--") for phase, color in colors.items())
        else:
            legend_entries.append(("Reconstructed", blue, "--"))
            astf_entries.append(("Predicted", red, "--"))
        for ax, entries in ((axes[0], legend_entries), (axes[3], astf_entries)):
            for label, color, style in entries:
                ax.plot([], [], label=label, color=color, ls=style, lw=1.2)
            ax.legend(loc="upper right", fontsize=11)
        _save_figure(figure, destination, dpi, preview=True)


def _plot_examples(
    path: Path, source_path: Path | None, destination: Path, dt: float | None, count: int, dpi: int
) -> None:
    with h5py.File(path, "r") as prediction:
        interval = _plot_interval(prediction, path, dt)
        if not len(prediction["predicted"]):
            return
        embedded = all(key in prediction for key in ("target_waveforms", "egfs"))
        if source_path is None and not embedded and "source" in prediction.attrs:
            candidate = Path(prediction.attrs["source"])
            source_path = candidate if candidate.is_file() else None
        if source_path is None and not embedded:
            if "actual" not in prediction:
                raise ValueError(f"{path}: supply --source-input to plot predictions without ASTF labels")
            _plot_legacy_examples(path, destination, interval, count, dpi)
            return
        if source_path is None:
            samples = _load_examples(prediction, prediction, count)
        else:
            with h5py.File(source_path, "r") as source:
                samples = _load_examples(prediction, source, count)
    rows, phases, target, egf, predicted, actual = samples
    _draw_five_columns(destination, rows, phases, target, egf, predicted, actual, interval, dpi)


def _plot_three_level_summary(
    paths: list[Path], output_dir: Path, dt: float | None, min_duration: float, limit: float, dpi: int
) -> list[dict[str, Any]]:
    metrics = [read_prediction_metrics(path, dt) for path in paths]
    prepared = [prepare_duration_values(item, min_duration, limit) for item in metrics]
    summaries = [item["summary"] for item in prepared]
    norm = LogNorm(vmin=1, vmax=max(2, max(item["histogram"].max() for item in prepared)))
    with plt.rc_context(
        {"font.size": 24, "axes.labelsize": 26, "axes.titlesize": 26, "pdf.fonttype": 42, "ps.fonttype": 42}
    ):
        figure = plt.figure(figsize=(23, 15))
        grid = figure.add_gridspec(2, 3, left=0.055, right=0.905, bottom=0.13, top=0.94, wspace=0.11, hspace=0.23)
        axes = np.array([[figure.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)])
        for column, (path, item, values) in enumerate(zip(paths, metrics, prepared)):
            match = re.search(r"(?:level|lvl)[_-]?([123])", path.stem, re.IGNORECASE)
            title = f"Test level {match.group(1)}" if match else path.stem
            axes[0, column].set_title(title, pad=22)
            _draw_duration(axes[0, column], values, norm)
            _draw_cc(axes[1, column], item, values["summary"])
            for row in range(2):
                axes[row, column].text(
                    -0.12,
                    1.06,
                    f"({chr(97 + 3 * row + column)})",
                    transform=axes[row, column].transAxes,
                    fontweight="bold",
                )
                axes[row, column].tick_params(width=1.35, length=5.5)
                if column:
                    axes[row, column].set_ylabel("")
            axes[0, column].tick_params(labelleft=column == 0)
            summary = values["summary"]
            box = axes[1, column].get_position()
            figure.text(
                (box.x0 + box.x1) / 2,
                0.018,
                f"N={summary['sample_count']:,}; finite CC={summary['cc_finite_count']:,}\n"
                f"Duration stats={summary['duration_metric_count']:,}; "
                f"displayed={summary['duration_display_pair_count']:,}",
                ha="center",
                fontsize=13,
            )
        figure.canvas.draw()
        box = axes[0, 2].get_position()
        colorbar_ax = figure.add_axes((box.x1 + 0.006, box.y0, 0.012, box.height))
        figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="OrRd"), cax=colorbar_ax, label="Counts per bin")
        _save_figure(figure, output_dir / "duration_cc_combined_2x3", dpi, preview=True)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    return summaries


def plot_prediction_report(
    paths: list[Path],
    output_dir: Path,
    dt: float | None = None,
    min_duration: float = 0.03,
    limit: float = 1.0,
    dpi: int = 600,
    examples: int = 30,
    source_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Export per-file examples and a summary when given three labeled levels."""
    if not paths or dpi <= 0 or not 0 <= examples <= 30:
        raise ValueError("Supply predictions, positive dpi, and an example count between 0 and 30")
    if source_paths is not None and len(source_paths) != len(paths):
        raise ValueError("Supply one --source-input per prediction file, in the same order")
    labeled = []
    for path in paths:
        with h5py.File(path, "r") as stream:
            _plot_interval(stream, path, dt)
            if "predicted" not in stream:
                raise ValueError(f"{path}: missing predicted dataset")
            labeled.append("actual" in stream)
    has_summary = len(paths) == 3 and all(labeled)
    if not has_summary and not examples:
        raise ValueError("With examples disabled, plotting requires three labeled prediction files")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = _plot_three_level_summary(paths, output_dir, dt, min_duration, limit, dpi) if has_summary else []
    if examples:
        for index, path in enumerate(paths):
            source_path = source_paths[index] if source_paths is not None else None
            _plot_examples(path, source_path, output_dir / f"examples_column_{index + 1}", dt, examples, dpi)
    return summaries
