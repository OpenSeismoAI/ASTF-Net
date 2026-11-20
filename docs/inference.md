# Checkpoint inference

From a source checkout, install the locked environment and run:

```bash
uv sync --frozen
uv run --frozen astfnet-predict --checkpoint /path/to/best.ckpt \
  --data /path/to/HiASTF --output-dir outputs/predictions --device cpu --dt 0.01
```

`--data` selects all three canonical files under `test_set/`. For a single
paired HDF5, or a different collection of files, use `--input` instead:

```bash
uv run --frozen astfnet-predict --checkpoint /path/to/best.ckpt \
  --input /path/to/input.h5 --output-dir outputs/single --batch-size 128
```

Inputs must have real numeric `target_waveforms` and `egfs` datasets with the
same nonempty `(N, T)` shape. `astfs` is optional; if provided, it must have
shape `(N, output_length)` and contain raw ASTF labels. Unlabeled paired inputs
can be predicted without creating dummy labels. Nonfinite values are rejected
with the affected batch row range. No augmentation, shuffle or dropped final
batch is used.

## Model and preprocessing

Select an explicit checkpoint; the command never guesses a best model by
directory order or a filename's rounded metric. The backbone comes from the
same registry used by training. Supported architectures are those actually
registered in this checkout: SimpleCNN, CNN-Transformer and U-Net1D.

Native Lightning checkpoints may store flat hyperparameters or a nested
`config`. Bare backbone state dictionaries are also supported with a matching
`--config /path/to/training.yaml`. An explicit YAML overrides checkpoint
settings and is recorded in the output. The model name, dimensions and
normalization must match training; arbitrary legacy architectures or state key
prefixes are not inferred. Weights are loaded strictly. The known Lightning
`model.` prefix is removed when loading the backbone. Native checkpoint
hyperparameter metadata is deserialized along with its weights.

The input signed-log transform is shared with `SeismicDatasetHDF5`. When
`log_normalize_astf` is true, predictions are converted back using
`sign(y) * (exp(abs(y)) - 1)`. Missing normalization flags use the training
defaults, both true. The exported `actual` values are the raw input labels.
Models run in evaluation/inference mode, with dropout disabled.

Default execution is CPU; `--device cuda`, `cuda:INDEX` or `auto` is optional.
Inference reads and writes one batch at a time rather than retaining all
waveforms. Batch size controls memory use; it does not select a subset.

## Saved results

Each source produces `<source-stem>_predictions.h5`:

- `predicted`: `(N, output_length)`, float32, in physical ASTF space.
- `actual`: optional raw labels with the same shape.
- `sample_index`: zero-based source row indices, preserving every input row.
- Attributes: source and checkpoint paths, checkpoint SHA-256, effective config
  JSON, device, prediction-space definition and completed status.

Files are built at a temporary path and published only after all batches
succeed. Existing outputs are never overwritten, including concurrently created
files. Use a new output directory for another run. Input names must have distinct
stems when processing multiple files.

`--dt` declares the sample interval in seconds. If omitted, it is read from the
input HDF5's `dt` attribute when available. Otherwise the prediction remains
valid without a time interval. The command does not silently assume a sampling
rate. Timing diagnostics require a known interval. Waveform preparation such as
resampling, filtering and event/EGF pairing must already match the trained model;
these are not guessed from raw SAC files.
