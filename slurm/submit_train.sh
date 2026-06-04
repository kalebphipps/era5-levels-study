#!/bin/bash
# Distributed training on HoreKa (TEAL/Ruby H100/H200). One task per GPU.
#
#   export WS=$(ws_find levels); export DATA_DIR=/path/to/zarrs; export WORKDIR=$(pwd)
#   sbatch slurm/submit_train.sh configs/base_0p25.yaml configs/levels13.yaml
#   sbatch slurm/submit_train.sh configs/base_0p25.yaml configs/levels37.yaml
#
# IMPORTANT: NODES * GPUS_PER_NODE must equal the product of mesh_dims in the
# config (base_0p25.yaml defaults to 8 GPUs -> e.g. 2 nodes x 4 GPUs).
#
#SBATCH --job-name=levels
#SBATCH --partition=TODO_PARTITION      # <- set to the TEAL/Ruby H100/H200 partition (check `sinfo`)
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4             # = GPUs per node
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
## #SBATCH --constraint=LSDF            # uncomment if data is on LSDF

set -euo pipefail
module purge 2>/dev/null || true
: "${WS:?Set WS}"; : "${DATA_DIR:?Set DATA_DIR (zarr root)}"; : "${WORKDIR:=$(pwd)}"
source "$WS/venv/bin/activate"
mkdir -p logs

BASE_CONFIG="${1:?pass a base config, e.g. configs/base_0p25.yaml}"
OVERLAY="${2:?pass a levels overlay, e.g. configs/levels37.yaml}"

# Outputs (training_loop writes to OUTPUT_DIR/<partition>/<jobid>/...)
export OUTPUT_DIR="$WS/results"
export PROFILE_DIR="$WS/results/profiles"
export DATA_DIR WORKDIR

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
srun python -u -m era5_levels.main --config "$BASE_CONFIG" --overlay "$OVERLAY"
