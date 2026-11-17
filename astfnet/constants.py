"""Common data constants and path resolution for ASTF-net."""

import os
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Canonical file names
# ---------------------------------------------------------------------------

TRAIN_FILENAME = "Train_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5"
VAL_FILENAME = "Validation_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5"

HIASTF_TRAIN_SUBDIR = "EGFs_new_3_with_ASTFs_vr_ratio2_min0"
TEST_SUBDIR = "test_set"
TEST_FILENAMES: List[str] = [
    "Test_level1_new_3_pairs_Samezero256_normalize_vr_1.h5",
    "Test_level2_new_3_pairs_Samezero256_normalize_vr_1.h5",
    "Test_level3_new_3_pairs_Samezero256_normalize_vr_1.h5",
]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_data_paths(data_path: str) -> Dict[str, Any]:
    """Derive canonical HDF5 paths from a single data directory.

    Args:
        data_path: Root directory containing train/val files directly or in
            ``HIASTF_TRAIN_SUBDIR``, and a ``test_set/`` subdirectory for test
            files. Direct train/val paths take precedence over the nested
            Hugging Face layout.

    Returns:
        Dictionary with keys ``train_hdf5_file``, ``val_hdf5_file``, and
        ``test_hdf5_files`` (a list of paths to all test HDF5 files).
    """
    train_dir = data_path
    nested_train_dir = os.path.join(data_path, HIASTF_TRAIN_SUBDIR)
    if not os.path.isfile(os.path.join(data_path, TRAIN_FILENAME)) and os.path.isfile(
        os.path.join(nested_train_dir, TRAIN_FILENAME)
    ):
        train_dir = nested_train_dir
    test_dir = os.path.join(data_path, TEST_SUBDIR)
    return {
        "train_hdf5_file": os.path.join(train_dir, TRAIN_FILENAME),
        "val_hdf5_file": os.path.join(train_dir, VAL_FILENAME),
        "test_hdf5_files": [os.path.join(test_dir, fname) for fname in TEST_FILENAMES],
    }
