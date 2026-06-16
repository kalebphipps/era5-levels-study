#!/bin/bash
# Create subsets of 37 pressure level zarr.
#
#SBATCH --job-name=subset_13
#SBATCH --partition=intel-spr
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
