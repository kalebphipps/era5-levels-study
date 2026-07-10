#!/bin/bash
# Same as submit_maps_dev.sh but on the A100 'accelerated' partition (4 GPUs).
#
#SBATCH --job-name=maps_a100
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-test-p0028019
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

BASE=configs/base_1p5.yaml
LEVELS=configs/levels37_1p5.yaml
MESH=configs/mesh_4gpu.yaml
MAP_VARS="geopotential_950 geopotential_3 temperature_950 temperature_3"

export OUTPUT_DIR="$WS/results" WORKDIR
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export TMPDIR=/tmp/${SLURM_JOB_ID} PYTHONPYCACHEPREFIX=${TMPDIR}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX"

for RUN in levels37_1p5 transfer_13core_37io; do
    echo "== dumping OOD maps for $RUN =="
    srun python -u scripts/run_subset_eval.py \
        --config "$BASE" --overlay "$LEVELS" --overlay "$MESH" \
        --results-dir "$WS/results/$RUN" \
        --dump-maps --map-vars $MAP_VARS \
        --out-csv "$WS/results/$RUN/subset_metrics_maps.csv"
done
echo "done -> maps in $WS/results/{levels37_1p5,transfer_13core_37io}/maps/"
