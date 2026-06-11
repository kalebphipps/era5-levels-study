#!/bin/bash
# CPU-only coarsen of the 0.25deg 37-level zarr -> 1.5deg. No GPU needed.
#
# Defaults to FTP's Intel Sapphire Rapids partition (free compute). IMPORTANT:
# it is NOT confirmed that the HoreKa workspace (/hkfs/work/workspace) is mounted
# on FTP nodes. If the job can't see your input zarr or your venv, run this on
# HoreKa instead by setting  --partition=cpuonly  below (data + venv guaranteed
# visible there). Check support docs / a quick `ls $IN` in a test job first.
#
# Submit (set the paths + venv):
#   export VENV=$WS/venv
#   IN=/path/era5_37level_0p25.zarr OUT=$WS/data/era5_37level_1p5.zarr \
#   START=1990-01-01 END=2022-12-31 STRIDE=6 \
#       sbatch slurm/coarsen_cpu.sh
#
#SBATCH --job-name=coarsen_1p5
#SBATCH --partition=intel-spr        # FTP CPU (Sapphire Rapids). HoreKa fallback: cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
## #SBATCH --account=PROJECT          # uncomment + set if your account needs it

set -euo pipefail
module purge 2>/dev/null || true
: "${VENV:?Set VENV to your venv dir, e.g. export VENV=\$WS/venv}"
source "$VENV/bin/activate"
mkdir -p logs

IN="${IN:?Set IN=/path/to/era5_37level_0p25.zarr}"
OUT="${OUT:?Set OUT=/path/to/era5_37level_1p5.zarr}"
START="${START:-1990-01-01}"
END="${END:-2022-12-31}"
STRIDE="${STRIDE:-1}"                 # 1 = keep every timestep (HOURLY). Only set >1 to deliberately subsample.
BLOCK="${BLOCK:-48}"                  # checkpoint granularity (timesteps per append)

# sanity: confirm the input is actually visible from this node before the long run
ls -d "$IN" >/dev/null || { echo "ERROR: cannot see input $IN from this node"; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
echo "coarsen $IN -> $OUT  range=[$START,$END] stride=$STRIDE block=$BLOCK cpus=$OMP_NUM_THREADS"
# Resumable: re-running continues from however many timesteps are already in OUT.
python -u scripts/coarsen_to_1p5.py \
    --in "$IN" --out "$OUT" \
    --factor 6 --time-range "$START" "$END" --time-stride "$STRIDE" --block "$BLOCK"
echo "done."
