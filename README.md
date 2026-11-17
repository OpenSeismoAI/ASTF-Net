# Overview

ASTF-net is an open-source deep learning framework designed for apparent seismic source time function inversion.

# Data
![ASTF-net Logo](https://github.com/user-attachments/assets/c7c4a6e9-006a-447b-b697-31ed34e42424)

The dataset used for ASTF-net is **HiASTF**, available on Hugging Face at
[pangruncheng/HiASTF](https://huggingface.co/datasets/pangruncheng/HiASTF).
It was public and ungated when these instructions were verified on 2026-11-17;
no login was required.

After completing [Installation](#installation), the locked environment already
includes the Hugging Face CLI. Download the five files required by the current
configuration at the fixed [dataset revision](https://huggingface.co/datasets/pangruncheng/HiASTF/tree/90da8636f2a1de949b7c42eb6f8654be3ee0c538):

```bash
uv run --frozen huggingface-cli download pangruncheng/HiASTF \
  "EGFs_new_3_with_ASTFs_vr_ratio2_min0/Train_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5" \
  "EGFs_new_3_with_ASTFs_vr_ratio2_min0/Validation_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5" \
  "test_set/Test_level1_new_3_pairs_Samezero256_normalize_vr_1.h5" \
  "test_set/Test_level2_new_3_pairs_Samezero256_normalize_vr_1.h5" \
  "test_set/Test_level3_new_3_pairs_Samezero256_normalize_vr_1.h5" \
  --repo-type dataset \
  --revision 90da8636f2a1de949b7c42eb6f8654be3ee0c538 \
  --local-dir /path/to/data/
```

The selected files total **10,379,225,769 bytes (10.38 GB / 9.67 GiB)**.
Environment files, temporary files, checkpoints and logs require additional
space. The sizes and SHA-256 values below come from this revision's Hugging Face
LFS metadata:

| Split | Bytes | SHA-256 |
|---|---:|---|
| Train | 8,778,650,449 | `052c082be09de598748142245acaaca0cf044f40e2081d41c6be88693b94afb6` |
| Validation | 817,857,596 | `05dbf53acb4166fe7570dec41d75df197ea101f3b2e342f2b949bffa46c58303` |
| Test level 1 | 204,796,148 | `b3a83bffb095659ae6f46a145c543f52fe435c38fa79cda2e5c3e5780fa6ec2c` |
| Test level 2 | 126,150,668 | `9efbd9d1b0d105271dc62a3a1e98d007af9dac2d2f03a1764391577ab34077b7` |
| Test level 3 | 451,770,908 | `4d88f6881057450470612c03cf59474955bc56799126f86a849f1e96f2692b65` |

# Dataset Setup

The download preserves the repository's directory structure:

```text
/path/to/data/
├── EGFs_new_3_with_ASTFs_vr_ratio2_min0/
│   ├── Train_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5
│   └── Validation_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5
└── test_set/
    ├── Test_level1_new_3_pairs_Samezero256_normalize_vr_1.h5
    ├── Test_level2_new_3_pairs_Samezero256_normalize_vr_1.h5
    └── Test_level3_new_3_pairs_Samezero256_normalize_vr_1.h5
```

Pass the download root to the training CLI:

```bash
uv run --frozen astfnet-train --config config/config-simplecnn.yaml --data /path/to/data/
```

`resolve_data_paths()` also supports the existing layout with train/validation
files directly under `/path/to/data/`. When both layouts are present, the direct
layout takes precedence. Test files remain under `/path/to/data/test_set/` in
both layouts.

The canonical filenames and Hugging Face training subdirectory are defined in
[`astfnet/constants.py`](astfnet/constants.py).

### Requirements
Ensure that [Python](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads) are configured on your machine with:
```bash
python3 --version
git --version
```

It is recommended to use [uv](https://docs.astral.sh/uv/) for dependency management. uv can be installed with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Other [uv installation methods](https://docs.astral.sh/uv/getting-started/installation/) can be found on the uv website.

### Installation
Download the repository and navigate to its root directory with:
```bash
git clone https://github.com/pangruncheng/ASTF-net.git
cd ASTF-net
```

Create a virtual environment and activate it with:
```bash
uv venv
source .venv/bin/activate
```

Install the `astfnet` package. Choose the mode that fits your use case:

```bash
# Standard install — core dependencies only
uv sync

# Development install — adds pre-commit, pytest, pytest-cov, ruff, and ty
uv sync --extra dev

# Full install — core and development dependencies
uv sync --all-extras
```

By default, `uv sync` installs the package in editable mode, which means that the package does not need to be reinstalled for modifications to the source code to take effect. Running `uv sync` also exposes the CLI entry points listed under `[project.scripts]` in `pyproject.toml`.

Install pre-commit hooks with:
```bash
pre-commit install
```
Test
```bash
pytest tests/
```
### Usage
Installing the application with uv sync exposes the following CLI entry points:
```
astfnet-train               # train a model
```
The entry points can be called directly. For example:
```bash
astfnet-train --config config/config-simplecnn.yaml --data /path/to/data/
```

More information on the usage and the options available for these entry points can be obtained by calling the --help option. For example:
```bash
astfnet-train --help
```
