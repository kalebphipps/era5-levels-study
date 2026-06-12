#!/bin/bash
# Distributed training on HoreKa (TEAL/Ruby H100/H200). One task per GPU.
#
#   export WS=$(ws_find levels); export DATA_DIR=/path/to/zarrs; export WORKDIR=$(pwd)
#   sbatch slurm/submit_train.sh configs/base_0p25.yaml configs/levels13.yaml
#   sbatch slurm/submit_train.sh configs/base_0p25.yaml configs/levels37.yaml
#
# IMPORTANT: NODES * GPUS_PER_NODE must equal the product of mesh_dims in the
# config. base_0p25.yaml is now SINGLE-NODE: mesh [1,1,1,2,2] = 4 GPUs = 1 node.
# (To scale to 8 later: set mesh [1,1,1,4,2] + parallelism 8 and --gres=gpu:8 on
# an accelerated-h200-8 node, or --nodes=2 on a 4-GPU partition.)
#
#SBATCH --job-name=levels
#SBATCH --partition=TODO_PARTITION      # <- e.g. accelerated-h200 or accelerated-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4             # = GPUs per node = product(mesh_dims)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=LSDF              # needed for 0.25deg (data on /lsdf); harmless but
                                       # unnecessary for 1.5deg (data on the workspace)

set -euo pipefail
module purge 2>/dev/null || true
: "${WS:?Set WS}"; : "${DATA_DIR:?Set DATA_DIR (zarr root)}"; : "${WORKDIR:=$(pwd)}"
source "$WS/venv/bin/activate"
mkdir -p logs

BASE_CONFIG="${1:?pass a base config, e.g. configs/base_0p25.yaml}"
OVERLAY="${2:?pass a levels overlay, e.g. configs/levels37.yaml}"

# Outputs. RUN_DIR is a STABLE per-experiment dir (checkpoints + metrics.csv).
# Reusing it auto-resumes from the latest checkpoint — that's how chained jobs
# survive time-limit kills (see submit_chain.sh). Default is derived from the
# overlay name so levels13 / levels37 don't collide; export RUN_DIR yourself to
# override or to start a clean experiment.
export OUTPUT_DIR="$WS/results"
export PROFILE_DIR="$WS/results/profiles"
export DATA_DIR WORKDIR
export RUN_DIR="${RUN_DIR:-$WS/results/$(basename "$OVERLAY" .yaml)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR (auto-resumes if checkpoints already exist there)"

# torch.distributed rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
# Multi-node NCCL: pick the InfiniBand interface (verify the name with `ip a`
# on a compute node — has been ib0 / ibs1 on different systems).
# export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

# Keep temp + python bytecode cache on node-local disk, off the shared FS.
export TMPDIR=/tmp/${SLURM_JOB_ID}
export PYTHONPYCACHEPREFIX=${TMPDIR}/pycache
srun -N "$SLURM_NNODES" --ntasks-per-node 1 bash -c 'mkdir -p ${PYTHONPYCACHEPREFIX}'

echo "Training: base=$BASE_CONFIG overlay=$OVERLAY  on $SLURM_NTASKS GPUs"
srun python -u -m era5_levels.main \
    --config "$BASE_CONFIG" --overlay "$OVERLAY" --run-dir "$RUN_DIR"
