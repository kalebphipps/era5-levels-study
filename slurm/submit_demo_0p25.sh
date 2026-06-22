#!/bin/bash
# 0.25-degree feasibility study.
#
#SBATCH --job-name=demo_0p25
#SBATCH --partition=accelerated-h200-8
#SBATCH --account=hk-project-test-p0028019
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --constraint=LSDF
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module purge 2>/dev/null || true
: "${WS:?Set WS, e.g. export WS=\$(ws_find levels)}"; : "${WORKDIR:=$(pwd)}"
source "$WS/venv/bin/activate"
mkdir -p logs

BASE_CONFIG="${1:?pass a base config, e.g. configs/base_0p25.yaml}"
OVERLAY="${2:?pass an overlay, e.g. configs/demo_0p25_37.yaml}"

export OUTPUT_DIR="$WS/results" WORKDIR
export RUN_DIR="${RUN_DIR:-$WS/results/$(basename "$OVERLAY" .yaml)}"   # stable -> auto-resume
mkdir -p "$RUN_DIR"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export TMPDIR=/tmp/${SLURM_JOB_ID} PYTHONPYCACHEPREFIX=${TMPDIR}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX"

echo "0.25deg demo: $BASE_CONFIG + $OVERLAY on $SLURM_NTASKS GPUs  RUN_DIR=$RUN_DIR"
srun python -u -m era5_levels.main \
    --config "$BASE_CONFIG" --overlay "$OVERLAY" --run-dir "$RUN_DIR"
