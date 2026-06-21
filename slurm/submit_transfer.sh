#!/bin/bash
# Frozen-core transfer.
#
#SBATCH --job-name=transfer
#SBATCH --partition=accelerated-h200-8
#SBATCH --account=hk-project-test-p0028019
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module purge 2>/dev/null || true
: "${WS:?Set WS, e.g. export WS=\$(ws_find levels)}"; : "${WORKDIR:=$(pwd)}"
source "$WS/venv/bin/activate"
mkdir -p logs

BASE_CONFIG="${1:?pass a base config, e.g. configs/base_1p5.yaml}"
LEVELS_OVERLAY="${2:?pass the TARGET levels overlay, e.g. configs/levels13_1p5.yaml}"
TRANSFER_OVERLAY="${3:?pass a transfer overlay, e.g. configs/transfer_37core_13io.yaml}"

export OUTPUT_DIR="$WS/results" WORKDIR
export RUN_DIR="${RUN_DIR:-$WS/results/$(basename "$TRANSFER_OVERLAY" .yaml)}"
mkdir -p "$RUN_DIR"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export TMPDIR=/tmp/${SLURM_JOB_ID} PYTHONPYCACHEPREFIX=${TMPDIR}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX"

echo "transfer train: $LEVELS_OVERLAY + $TRANSFER_OVERLAY -> $RUN_DIR"
srun python -u -m era5_levels.main \
    --config "$BASE_CONFIG" --overlay "$LEVELS_OVERLAY" --overlay "$TRANSFER_OVERLAY" \
    --run-dir "$RUN_DIR"

# Evaluate the trained result on the 13 standard levels.
echo "transfer eval on $RUN_DIR"
srun python -u scripts/run_subset_eval.py \
    --config "$BASE_CONFIG" --overlay "$LEVELS_OVERLAY" \
    --results-dir "$RUN_DIR" --dump-maps
