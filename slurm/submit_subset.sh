#!/bin/bash
# CPU-only 13-level subset of the 1.5deg 37-level zarr. Reads the whole 37-level
# store (~9 TB for 40yr hourly) -> writes the 84-feature 13-level store (~3.4 TB).
# Resumable: re-run (or chain) and it continues from the existing output.
#
#   export VENV=$WS/venv
#   IN=$WS/data/era5_37level_1p5.zarr OUT=$WS/data/era5_13level_1p5.zarr \
#   NORM_IN=data/normalization_1p5_37 NORM_OUT=data/normalization_1p5_13 \
#       sbatch slurm/submit_subset.sh
#
#SBATCH --job-name=subset_13
#SBATCH --partition=cpuonly          # 1.5deg data is on the workspace, no LSDF/GPU needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module purge 2>/dev/null || true
: "${VENV:?Set VENV, e.g. export VENV=\$WS/venv}"
source "$VENV/bin/activate"
mkdir -p logs

IN="${IN:?Set IN=.../era5_37level_1p5.zarr}"
OUT="${OUT:?Set OUT=.../era5_13level_1p5.zarr}"
NORM_IN="${NORM_IN:-}"
NORM_OUT="${NORM_OUT:-}"

ls -d "$IN" >/dev/null || { echo "ERROR: cannot see input $IN"; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

args=(--in "$IN" --out "$OUT")
[ -n "$NORM_IN" ] && [ -n "$NORM_OUT" ] && args+=(--norm-in "$NORM_IN" --norm-out "$NORM_OUT")

echo "subset $IN -> $OUT  (resumable)"
python -u scripts/make_level_subset.py "${args[@]}"
echo "done."
