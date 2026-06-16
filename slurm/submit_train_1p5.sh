#!/bin/bash
# 1.5deg training.
#
#SBATCH --job-name=levels_1p5
#SBATCH --partition=accelerated-h200-8
#SBATCH --account=TODO_ADD_ACCOUNT   # Add in with sbatch command
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# No --constraint=LSDF: 1.5deg data lives on the workspace.

set -euo pipefail
module purge 2>/dev/null || true
: "${WS:?Set WS, e.g. export WS=\$(ws_find levels)}"; : "${WORKDIR:=$(pwd)}"
source "$WS/venv/bin/activate"
mkdir -p logs

BASE_CONFIG="${1:?pass a base config, e.g. configs/base_1p5.yaml}"
OVERLAY="${2:?pass a levels overlay, e.g. configs/levels37_1p5.yaml}"

export OUTPUT_DIR="$WS/results" WORKDIR
export RUN_DIR="${RUN_DIR:-$WS/results/$(basename "$OVERLAY" .yaml)}"   # stable -> auto-resume
mkdir -p "$RUN_DIR"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export TMPDIR=/tmp/${SLURM_JOB_ID} PYTHONPYCACHEPREFIX=${TMPDIR}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX"

echo "1.5deg: $BASE_CONFIG + $OVERLAY on $SLURM_NTASKS GPUs  RUN_DIR=$RUN_DIR"
srun python -u -m era5_levels.main \
    --config "$BASE_CONFIG" --overlay "$OVERLAY" --run-dir "$RUN_DIR"
