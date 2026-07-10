#!/bin/bash
#
#SBATCH --job-name=eval_a100
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-test-p0028019   # override: sbatch --account=<other> ...
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module purge 2>/dev/null || true
: "${WS:=$(ws_find levels 2>/dev/null || echo /hkfs/work/workspace/scratch/xk5289-level_comparison)}"; : "${WORKDIR:=$(pwd)}"
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

echo "A100 eval: $OVERLAY on $RESULTS_DIR with 4-GPU mesh (jchannel=2 preserved)"
srun python -u scripts/run_subset_eval.py \
    --config "$BASE_CONFIG" --overlay "$OVERLAY" --overlay configs/mesh_4gpu.yaml \
    --results-dir "$RESULTS_DIR" --dump-maps
