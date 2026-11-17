# Reproducible training

The training CLI calls `pl.seed_everything(seed, workers=True)` before constructing
datasets and model weights. This seeds Python, NumPy and PyTorch and enables
Lightning's DataLoader worker seeding. Existing configs use these defaults:

```yaml
seed: 42
deterministic: true
benchmark: false
```

CLI options `--seed`, `--deterministic` and `--no-deterministic` take precedence
over YAML. Seeds must be integers in [0, 4294967295]. The two flags must be YAML
booleans; deterministic mode requires `benchmark: false`.

Strict deterministic algorithms may be slower and can raise an error when no
deterministic implementation exists. To prioritize speed, use
`--no-deterministic` and optionally set `benchmark: true` in the config. The seed
still applies, but deterministic equivalence is then not promised. Neither mode
guarantees matching results across PyTorch versions, CPU/GPU, different hardware
or worker counts. See [PyTorch 2.7 reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html).

## What each run records

At fit start, global rank zero writes `manifest.json` and the resolved
`config.yaml` into the TensorBoard run directory. The manifest contains:

- The exact command, working directory, effective seed and trainer flags.
- The source Git commit and dirty flag, plus SHA-256 for the package's Python
  source files. A wheel without a Git checkout reports a null commit.
- `uv.lock` path, size and SHA-256, a copy of that lock, and actual installed
  package versions. To use a lock outside the source checkout, set
  `dependency_lock: /absolute/path/to/uv.lock` in YAML. A missing lock is recorded
  as null; preserve one with the run to reproduce the environment.
- Python/platform/processor information, selected device, thread count, CUDA
  build and cuDNN version, GPU name when used, and effective deterministic/cuDNN
  and `CUBLAS_WORKSPACE_CONFIG` settings.
- Absolute paths, byte sizes and **full-file SHA-256** for training, validation
  and the test files that will be used. Hashing adds one complete read of the
  input files before the first training batch. `--skip-test` omits test files
  from this step, so they need not be present.

Declare the dataset's source revision using `--dataset-revision REVISION` or
`dataset_revision` in YAML. It is stored as **declared_revision**, independently
of the checksums computed from the actual local files. An unspecified revision
is null; local files are never silently labeled as a particular Hub snapshot.

For example, when using the documented pinned HiASTF download:

```bash
uv run --frozen astfnet-train --config config/config-simplecnn.yaml \
  --data /path/to/data --seed 42 --deterministic \
  --dataset-revision 90da8636f2a1de949b7c42eb6f8654be3ee0c538
```

The nested Hub layout requires the dataset-path correction tracked in issue #20.
This reproducibility change does not change dataset selection or normalization.

The manifest is updated atomically with final fit metrics, best/last checkpoint
paths and hashes, and each named test-set metric. Testing retains the existing
CLI behavior: it evaluates the **final in-memory weights**, without reloading
the best checkpoint. `fit_metrics` reports the last epoch; the best checkpoint
may belong to an earlier epoch.

Status progresses through `fit_started`, `fit_completed`, and `completed`.
With `--skip-test`, fit completion is run completion. Exceptions observed by
Lightning after the initial manifest are marked `failed` with their exception
type. An externally killed process can retain an unfinished status; failures
before the initial provenance write may have no manifest.

## Output locations

TensorBoard uses `<tb_output_dir>/<tb_exp_name>/version_N/`, with a fresh version
for each invocation. For the SimpleCNN example, this is
`.sdev/output/cnn_dev/version_N/`; checkpoints are in its `checkpoints/` directory.
The actual paths and checkpoint hashes are in the manifest.

```text
version_N/
├── manifest.json
├── config.yaml
├── uv.lock
├── hparams.yaml
├── events.out.tfevents...
└── checkpoints/
    ├── best-....ckpt
    └── last.ckpt
```

## Measured CPU smoke reference

This is an **engineering reference on generated data**, not a HiASTF or paper
accuracy result. It exercises the actual SimpleCNN, dropout, random noise
augmentation, shuffled loading with one worker, Adam, checkpointing, validation
and all three test-set paths. No large download or GPU is needed.

From the repository root:

```bash
uv sync --frozen
uv run --frozen python examples/prepare_repro_smoke.py --output outputs/repro-data
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen astfnet-train --config config/config-repro-smoke.yaml \
  --data outputs/repro-data
```

Run the training command twice to create `version_0` and `version_1` under
`outputs/repro-smoke/cpu/` (or the next available versions). Compare their
`fit_metrics`, `test_metrics` and final checkpoint **state_dict** tensors.
Generate data once and reuse it; the generator refuses to overwrite existing
reference files.
Checkpoint archive hashes can differ due to serialized paths/metadata even
when all model tensors are identical.

The generator uses NumPy seed 20261117, producing 8 records per split with
256-sample EGF/target and a 32-sample Gaussian ASTF; targets are truncated EGF/ASTF
convolutions. All five splits are generated independently. Training uses seed 42,
batch size 4, two epochs and four optimizer steps. Loss is unweighted MSE on
unnormalized synthetic ASTFs; the equal-sized batches make the epoch mean also
the sample mean.

Measured on 2026-11-17 with Python 3.12.10, PyTorch 2.7.0, Lightning 2.5.2,
NumPy 2.3.1, h5py 3.14.0 and SpeechBrain 1.0.3; Linux x86_64,
Intel Xeon Platinum 8476C, CPU execution with one PyTorch thread.
The dependency lock SHA-256 is
`870d3520af7ba72924d2516c9bb0a0497309b70172566f5694d0cadd27af6308`.

| Metric | Expected value in this reference environment |
| --- | ---: |
| Final `val/loss_epoch` | 0.2879800200 |
| Synthetic test level 1 MSE | 0.2900207341 |
| Synthetic test level 2 MSE | 0.2883585691 |
| Synthetic test level 3 MSE | 0.2883750200 |

Independent seeded runs in the same environment produced identical metrics and
all final model tensors. For comparison with the rounded table, use absolute
tolerance 1e-6; investigate differences using the manifests. These values are
not a cross-platform guarantee or a claim about real seismic generalization.
Full HiASTF scientific reference metrics and trained checkpoints remain to be
established using a documented real-data experiment.

The integration test repeats real CPU CLI runs in separate processes, comparing
all model tensors and fit/test metrics. It also checks skip-test operation with
absent test files and failure reporting. Run it with:

```bash
uv run --frozen --extra dev pytest tests/test_reproducibility.py -q
```
