#!/bin/bash
# Submit jobs to coarsen the 0.25deg 37-level zarr to 1.5deg
#
#SBATCH --job-name=coarsen_1p5
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
: "${VENV:?Set VENV to your venv dir, e.g. export VENV=\$WS/venv}"
source "$VENV/bin/activate"
mkdir -p logs

IN="${IN:?Set IN=/path/to/era5_37level_0p25.zarr}"
OUT="${OUT:?Set OUT=/path/to/era5_37level_1p5.zarr}"
START="${START:-1979-01-01}"
END="${END:-2022-12-31}"
STRIDE="${STRIDE:-1}"                 # 1 = keep every timestep (HOURLY). Only set >1 to deliberately subsample.
BLOCK="${BLOCK:-48}"                  # checkpoint granularity (timesteps per append)

# Confirm the input is actually visible before run
ls -d "$IN" >/dev/null || { echo "ERROR: cannot see input $IN from this node"; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
echo "coarsen $IN -> $OUT  range=[$START,$END] stride=$STRIDE block=$BLOCK cpus=$OMP_NUM_THREADS"
python -u scripts/coarsen_to_1p5.py \
    --in "$IN" --out "$OUT" \
    --factor 6 --time-range "$START" "$END" --time-stride "$STRIDE" --block "$BLOCK"
echo "done."
