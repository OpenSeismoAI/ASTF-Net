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

## Three-level plots

Add `--plot-summary` to save plots under `<output-dir>/plots`. This works with
single or multiple inputs, including `--data`. Each file produces a waveform
figure. Three files with ASTF labels also produce the 2×3 summary. A known `dt`
is required. Replot exported results without loading a model:

```bash
uv run --frozen astfnet-plot-predictions \
  --input outputs/predictions/test_level_1_predictions.h5 \
          outputs/predictions/test_level_2_predictions.h5 \
          outputs/predictions/test_level_3_predictions.h5 \
  --output-dir outputs/figures
```

The files are columns in the supplied order. The 2×3 layout follows the
project's three-level evaluation figure: top row cumulative-area duration
density, bottom row ASTF-label Pearson CC histograms. A shared logarithmic
colorbar represents duration-bin counts. PDF and PNG exports default to
600 dpi (`--dpi` for replotting, `--plot-dpi` during inference); a smaller PNG
preview is also saved. `metrics_summary.json` records counts and denominators.

Duration is the interpolated time from zero to 80% cumulative **absolute ASTF
area**, matching the reference evaluator. It is not an energy duration or a
central 80% interval. Zero-area or nonfinite curves have undefined duration and
are excluded, with counts retained. Relative-error fractions use every finite
pair whose actual duration is positive and at least 0.03 s; `--duration-threshold`
changes that cutoff. The default 0–1 s plot window (`--duration-limit`) only
controls display. It does not restrict the statistics. `Fit R²` is the linear-fit
correlation squared used by the reference, not predictive R² against the identity.

CC fractions use all finite waveform CC values, including those below the
histogram's 0.6 lower bound. Constant/nonfinite pairs are counted as undefined,
not assigned zero. This CC compares predicted and actual ASTFs; it is not the
CC of reconstructed mainshock waveforms. Physical-space MSE is reported
separately and is not the model-space training loss.

Each five-column example shows the target and reconstructed trace, their
Pearson CC, the EGF, the ASTF, and D80. Reconstruction convolves the EGF with
the predicted ASTF and finds the best CC within ±0.40 s of time shift on the
original target window. The first and third columns retain the full input
window. For display, those traces are peak scaled, and each ASTF pair shares
an amplitude scale. Only ASTFs are centered and zero padded for display. An
ASTF pair shifts together using the midpoint of its joint effective interval,
defined by each curve's samples at or above 1% of its absolute peak. Reported
D80 is calculated from the original ASTF. If true ASTFs are missing, the last
two columns show predictions alone.

An optional string `phase` dataset in the paired source file supplies one P/S
label per source row. With both phases present, the default selects 15 evenly
spaced records per phase, or all available when fewer exist. P is above S, and
each group is ordered by increasing predicted D80. Without both phase groups,
the default takes up to 30 evenly spaced records in a single group. The CSV by
each five-column figure records the source rows, phase, reconstruction CC,
time shift and D80 values. `--examples` sets the total maximum number of rows,
and `--examples 0` requests only the three-level summary.

Offline plotting finds paired waveforms embedded in the prediction HDF5 or in
its recorded source path. Supply `--source-input` in prediction-file order to
use a different path, such as after moving input files. `sample_index` maps rows
back to the paired source. Legacy files with only `(N, T)` or `(N, 1, T)`
`actual`/`predicted` data can still produce the ASTF-only comparison. Predictions
must already be in physical ASTF space. Supply `--dt` if the predictions lack
that attribute. Replotting replaces figures with the same name in the chosen
output directory.
