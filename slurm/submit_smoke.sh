#!/bin/bash
# Mini smoke test.
#
#SBATCH --job-name=levels_smoke
#SBATCH --partition=TODO_DEV_PARTITION   # Add in with sbatch command
#SBATCH --account=TODO_ACCOUNT # Add in with sbatch command
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module purge 2>/dev/null || true
: "${WS:?Set WS}"; : "${WORKDIR:=$(pwd)}"
source "$WS/venv/bin/activate"
mkdir -p logs
export OUTPUT_DIR="$WS/results" PROFILE_DIR="$WS/results/profiles" WORKDIR
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export TMPDIR=/tmp/${SLURM_JOB_ID} PYTHONPYCACHEPREFIX=/tmp/${SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX"

# config sanity first (no GPU/beast needed), then the real 1-GPU run:
python -m era5_levels.main --config configs/smoke.yaml --dry-run
srun python -u -m era5_levels.main --config configs/smoke.yaml
