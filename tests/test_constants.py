from pathlib import Path

import h5py
import numpy as np
import pytest

from astfnet.constants import TEST_FILENAMES, TEST_SUBDIR, TRAIN_FILENAME, VAL_FILENAME, resolve_data_paths
from astfnet.data_io.dataset import SeismicDatasetHDF5


@pytest.mark.parametrize("layout", ["flat", "huggingface"])
def test_resolve_data_paths_loads_documented_layout(tmp_path: Path, layout: str) -> None:
    train_dir = tmp_path if layout == "flat" else tmp_path / "EGFs_new_3_with_ASTFs_vr_ratio2_min0"
    expected = [train_dir / TRAIN_FILENAME, train_dir / VAL_FILENAME]
    expected.extend(tmp_path / TEST_SUBDIR / name for name in TEST_FILENAMES)
    rng = np.random.default_rng(42)
    for path in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as stream:
            for field in ("target_waveforms", "egfs", "astfs"):
                stream.create_dataset(field, data=rng.normal(size=(2, 8)).astype(np.float32))

    resolved = resolve_data_paths(str(tmp_path))
    paths = [resolved["train_hdf5_file"], resolved["val_hdf5_file"], *resolved["test_hdf5_files"]]
    assert [Path(path) for path in paths] == expected
    for path in paths:
        dataset = SeismicDatasetHDF5(path)
        assert len(dataset) == 2
        sample = dataset[0]
        assert set(sample) == {"target", "egf", "astf"}
        assert all(tensor.shape == (8,) for tensor in sample.values())


def test_resolve_data_paths_prefers_flat_layout(tmp_path: Path) -> None:
    (tmp_path / TRAIN_FILENAME).touch()
    nested = tmp_path / "EGFs_new_3_with_ASTFs_vr_ratio2_min0"
    nested.mkdir()
    (nested / TRAIN_FILENAME).touch()
    (nested / VAL_FILENAME).touch()
    resolved = resolve_data_paths(str(tmp_path))
    assert resolved["train_hdf5_file"] == str(tmp_path / TRAIN_FILENAME)
    assert resolved["val_hdf5_file"] == str(tmp_path / VAL_FILENAME)


def test_resolve_data_paths_preserves_missing_data_paths(tmp_path: Path) -> None:
    resolved = resolve_data_paths(str(tmp_path))
    assert resolved == {
        "train_hdf5_file": str(tmp_path / TRAIN_FILENAME),
        "val_hdf5_file": str(tmp_path / VAL_FILENAME),
        "test_hdf5_files": [str(tmp_path / TEST_SUBDIR / name) for name in TEST_FILENAMES],
    }
