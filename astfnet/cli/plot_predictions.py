"""Plot saved predictions with or without true ASTFs, without repeating inference."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from astfnet.prediction_plots import plot_prediction_report  # noqa: E402


def main() -> None:
    """Render saved waveform diagnostics and a summary for three labeled levels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--source-input", type=Path, nargs="+", help="Paired input HDF5 files, in prediction-file order"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, help="Sample interval in seconds; required for legacy files without dt")
    parser.add_argument("--duration-threshold", type=float, default=0.03)
    parser.add_argument("--duration-limit", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--examples", type=int, default=30, help="Maximum rows per file; split equally by P/S when available"
    )
    args = parser.parse_args()
    try:
        plot_prediction_report(
            args.input,
            args.output_dir,
            args.dt,
            args.duration_threshold,
            args.duration_limit,
            args.dpi,
            args.examples,
            args.source_input,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(f"Plots and metric denominators saved: {args.output_dir}")


if __name__ == "__main__":
    main()
