# Training quickstart

Use this guide to train SimpleCNN on a small HiASTF subset, evaluate it on the
three test levels, and locate the resulting checkpoint.

## Install ASTF-Net

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The project pins Python 3.12.10 and records its dependencies in `uv.lock`.

```bash
git clone --branch dev https://github.com/OpenSeismoAI/ASTF-Net.git
cd ASTF-Net
uv sync --frozen --python 3.12.10
uv run --frozen astfnet-train --help
```

Run the remaining commands from the checkout root.

## Locate the data

Get the training, validation and three test HDF5 files from
[HiASTF on Hugging Face](https://huggingface.co/datasets/pangruncheng/HiASTF).
Set `HIASTF_ROOT` to the directory that contains the two top-level files and
the `test_set` directory below. If they arrived under a nested folder, use
the folder that matches this layout or arrange local copies or links under
a new root.

```text
HIASTF_ROOT/
├── Train_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5
├── Validation_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5
└── test_set/
    ├── Test_level1_new_3_pairs_Samezero256_normalize_vr_1.h5
    ├── Test_level2_new_3_pairs_Samezero256_normalize_vr_1.h5
    └── Test_level3_new_3_pairs_Samezero256_normalize_vr_1.h5
```

Each file contains paired `target_waveforms`, `egfs` and `astfs` datasets.
The SimpleCNN configuration in this guide uses 256 samples per waveform
and per ASTF. Matching row indices identify one training or test example.

## Run one CPU epoch

Point the first line at your data directory. The example copies eight rows
from each file into a new temporary directory. It uses a batch size of four,
one data loader worker, and the shipped configuration's input normalization.

```bash
HIASTF_ROOT=/path/to/HiASTF
DEMO_DIR=$(mktemp -d)
uv run --frozen python - "$HIASTF_ROOT" "$DEMO_DIR" <<'PY'
import sys
from pathlib import Path

import h5py
from omegaconf import OmegaConf

from astfnet.constants import resolve_data_paths

root = Path(sys.argv[1]).resolve()
demo = Path(sys.argv[2])
paths = resolve_data_paths(str(root))
files = [paths["train_hdf5_file"], paths["val_hdf5_file"], *paths["test_hdf5_files"]]
for filename in files:
    source = Path(filename)
    subset = demo / "data" / source.relative_to(root)
    subset.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as original, h5py.File(subset, "x") as small:
        for key in ("target_waveforms", "egfs", "astfs"):
            small[key] = original[key][:8]

config = OmegaConf.load("config/config-simplecnn.yaml")
config.update({
    "device": "cpu",
    "gpus": 1,
    "max_epochs": 1,
    "batch_size": 4,
    "num_workers": 1,
    "augmentations": [],
    "max_augmentations": 0,
    "tb_output_dir": str(demo / "runs"),
    "tb_exp_name": "quickstart",
})
OmegaConf.save(config, demo / "cpu.yaml")
PY

uv run --frozen astfnet-train --config "$DEMO_DIR/cpu.yaml" --data "$DEMO_DIR/data"
find "$DEMO_DIR/runs" -name '*.ckpt' -print
```

Expect one training epoch, three test loss reports, and checkpoints under
`$DEMO_DIR/runs/quickstart/version_0/checkpoints/`. The run saves `last.ckpt`
and a checkpoint selected by validation loss. The eight-row splits are for
checking that the workflow runs.

## Train with your own configuration

The shipped [SimpleCNN](../config/config-simplecnn.yaml) and
[Transformer](../config/config-transformer.yaml) configurations are starting
points for GPU training. Copy the configuration you want to edit. Check
`device`, `gpus`, `batch_size`, `num_workers`, `max_epochs`, and `tb_output_dir`
before starting a run. The data loader reads the HDF5 arrays into host memory,
so plan memory use for the full training and validation files.

```bash
uv run --frozen astfnet-train --config config/config-simplecnn.yaml --data "$HIASTF_ROOT"
```

Pass `--skip-test` when you want training and validation only. The trainer
writes TensorBoard logs and checkpoints under
`<tb_output_dir>/<tb_exp_name>/version_<n>/`.
