"""Create tiny, deterministic synthetic HDF5 splits for the CPU reference run."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from astfnet.constants import resolve_data_paths


def prepare_data(output: Path) -> None:
    """Write independent 8-record splits with 256-sample input and 32-sample ASTF."""
    paths = resolve_data_paths(str(output))
    files = [paths["train_hdf5_file"], paths["val_hdf5_file"], *paths["test_hdf5_files"]]
    if any(Path(filename).exists() for filename in files):
        raise FileExistsError("Reference files already exist; reuse them or choose a new output directory.")
    rng = np.random.default_rng(20261117)
    for filename in files:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        egfs = rng.normal(size=(8, 256)).astype(np.float32)
        centers = rng.uniform(6, 20, size=(8, 1))
        astfs = np.exp(-0.5 * ((np.arange(32) - centers) / 3) ** 2).astype(np.float32)
        target = np.stack([np.convolve(egf, astf)[:256] for egf, astf in zip(egfs, astfs)])
        with h5py.File(path, "x") as stream:
            for key, values in {"egfs": egfs, "target_waveforms": target, "astfs": astfs}.items():
                stream.create_dataset(key, data=values, track_times=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare_data(parser.parse_args().output)
