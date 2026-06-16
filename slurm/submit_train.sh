#!/bin/bash
# Distributed training.
#
#SBATCH --job-name=levels
#SBATCH --partition=TODO_PARTITION      # Add in with sbatch command
#SBATCH --account=TODO_ADD_ACCOUNT  # Add in with sbatch command
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=LSDF              # needed for 0.25deg (data on lsdf)

set -euo pipefail
module purge 2>/dev/null || true
: "${WS:?Set WS}"; : "${DATA_DIR:?Set DATA_DIR (zarr root)}"; : "${WORKDIR:=$(pwd)}"
source "$WS/venv/bin/activate"
mkdir -p logs

BASE_CONFIG="${1:?pass a base config, e.g. configs/base_0p25.yaml}"
OVERLAY="${2:?pass a levels overlay, e.g. configs/levels37.yaml}"
export OUTPUT_DIR="$WS/results"
export PROFILE_DIR="$WS/results/profiles"
export DATA_DIR WORKDIR
export RUN_DIR="${RUN_DIR:-$WS/results/$(basename "$OVERLAY" .yaml)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR (auto-resumes if checkpoints already exist there)"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

export TMPDIR=/tmp/${SLURM_JOB_ID}
export PYTHONPYCACHEPREFIX=${TMPDIR}/pycache
srun -N "$SLURM_NNODES" --ntasks-per-node 1 bash -c 'mkdir -p ${PYTHONPYCACHEPREFIX}'

echo "Training: base=$BASE_CONFIG overlay=$OVERLAY  on $SLURM_NTASKS GPUs"
srun python -u -m era5_levels.main \
    --config "$BASE_CONFIG" --overlay "$OVERLAY" --run-dir "$RUN_DIR"
