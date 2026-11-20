"""Batch ASTF prediction from an explicit checkpoint and paired HDF5 inputs."""

import argparse
from pathlib import Path

from astfnet.constants import resolve_data_paths
from astfnet.inference import inspect_input, load_inference_model, predict_hdf5


def main() -> None:
    """Predict single files or the three canonical HiASTF test levels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Matching YAML; overrides checkpoint hyperparameters")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", type=Path, nargs="+", help="Paired target_waveforms/egfs HDF5 files")
    sources.add_argument("--data", type=Path, help="Dataset root containing the canonical test_set directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, cuda:INDEX, or auto")
    parser.add_argument("--dt", type=float, help="Sample interval in seconds; otherwise read the input dt attribute")
    args = parser.parse_args()
    paths = (
        [Path(path) for path in resolve_data_paths(str(args.data))["test_hdf5_files"]]
        if args.data is not None
        else args.input
    )
    try:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        if len({path.stem for path in paths}) != len(paths):
            raise ValueError("Input filenames must have distinct stems to avoid output collisions")
        outputs = [args.output_dir / f"{path.stem}_predictions.h5" for path in paths]
        for path, output in zip(paths, outputs):
            inspect_input(path, args.dt)
            if output.exists():
                raise FileExistsError(f"Output already exists: {output}")
        model, config = load_inference_model(args.checkpoint, args.config, args.device)
        for path, output in zip(paths, outputs):
            predict_hdf5(model, config, path, output, args.checkpoint, args.batch_size, args.dt)
            print(f"Predictions saved: {output}")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
