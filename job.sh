#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --job-name=astfnet-train
#SBATCH --output=%x__%j.log

set -euo pipefail

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

if (( $# == 0 )); then
    printf 'Usage: job.sh DATA_DIR [CONFIG_YAML [TRAIN_OPTIONS...]]\n' >&2
    exit 2
fi

data_dir=$1
config=${2:-config/config-transformer.yaml}
if (( $# > 1 )); then
    shift 2
else
    shift
fi

[[ -f pyproject.toml && -d astfnet ]] || fail 'Run from the repository root; use sbatch --chdir=/path/to/ASTF-Net.'
[[ -d "$data_dir" ]] || fail "Data directory not found: $data_dir"
[[ -f "$config" && -r "$config" ]] || fail "Config file not readable: $config"

python="$PWD/.venv/bin/python"
[[ -x "$python" && -x .venv/bin/astfnet-train ]] || fail "Project environment missing or incomplete; run 'uv sync --frozen'."
version=$("$python" -c 'import platform; print(platform.python_version())')
[[ "$version" == 3.12.10 ]] || fail "Python 3.12.10 is required (found $version); run 'uv sync --frozen'."

exec "$python" -m astfnet.cli.train --config "$config" --data "$data_dir" "$@"
