#!/bin/bash
# 1.5deg training on a single 8-GPU node (accelerated-h200-8). The 8-GPU mesh
# [1,2,1,2,2] is in base_1p5.yaml, so this just needs 8 tasks on one node. Data
# is on the workspace (no LSDF). 2-day limit = the full 48h budget in one job;
# auto-resumes from RUN_DIR if it dies, so just resubmit.
#
#   export WS=$(ws_find levels); export WORKDIR=$(pwd)
#   sbatch slurm/submit_train_1p5.sh configs/base_1p5.yaml configs/levels37_1p5.yaml
#   sbatch slurm/submit_train_1p5.sh configs/base_1p5.yaml configs/levels13_1p5.yaml
#
#SBATCH --job-name=levels_1p5
#SBATCH --partition=accelerated-h200-8   # 8x H200 / node (check the exact name with sinfo)
#SBATCH --account=hk-project-test-p0028019   # HoreKa requires this. Override on the fly: sbatch --account=<other> slurm/submit_train_1p5.sh ...
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8              # = product(mesh_dims) = 8
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
