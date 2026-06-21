#!/bin/bash
# Batched subset evaluation.
#
#SBATCH --job-name=subset_eval
#SBATCH --partition=accelerated-h200-8
#SBATCH --account=hk-project-test-p0028019
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module purge 2>/dev/null || true
: "${WS:?Set WS, e.g. export WS=\$(ws_find levels)}"; : "${WORKDIR:=$(pwd)}"
source "$WS/venv/bin/activate"
mkdir -p logs

BASE_CONFIG="${1:?pass a base config, e.g. configs/base_1p5.yaml}"
OVERLAY="${2:?pass the levels overlay used to train the run}"
RESULTS_DIR="${3:?pass the run dir to evaluate (holds checkpoints/)}"

export OUTPUT_DIR="$WS/results" WORKDIR
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export TMPDIR=/tmp/${SLURM_JOB_ID} PYTHONPYCACHEPREFIX=${TMPDIR}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX"

echo "subset eval: $OVERLAY on $RESULTS_DIR"
srun python -u scripts/run_subset_eval.py \
    --config "$BASE_CONFIG" --overlay "$OVERLAY" \
    --results-dir "$RESULTS_DIR" --dump-maps
