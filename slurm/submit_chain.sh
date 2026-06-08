#!/usr/bin/env bash
# Submit a CHAIN of training jobs that survive SLURM time-limit kills.
#
# Each job runs submit_train.sh against the SAME stable RUN_DIR and depends on
# the previous one with `afterany` (so it starts whether the previous job
# finished, was killed at the time limit, or failed). On start each job
# auto-resumes from the latest checkpoint in RUN_DIR, so a long run is just
# "N short jobs that hand off the checkpoint".
#
#   export WS=$(ws_find levels); export DATA_DIR=/zarr/root; export WORKDIR=$(pwd)
#   bash slurm/submit_chain.sh configs/base_0p25.yaml configs/levels37.yaml 4
#                              ^base                  ^overlay              ^#links (default 3)
#
# Run it once for levels13 and once for levels37 — they form two independent
# chains and train in parallel.
set -euo pipefail

BASE="${1:?pass a base config, e.g. configs/base_0p25.yaml}"
OVERLAY="${2:?pass a levels overlay, e.g. configs/levels37.yaml}"
N="${3:-3}"

: "${WS:?Set WS, e.g. export WS=\$(ws_find levels)}"
RUN_DIR="${RUN_DIR:-$WS/results/$(basename "$OVERLAY" .yaml)}"
mkdir -p "$RUN_DIR" logs

echo "Chaining $N jobs sharing RUN_DIR=$RUN_DIR"
prev=""
for i in $(seq 1 "$N"); do
    dep=""
    [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    jid=$(sbatch --parsable $dep \
          --export=ALL,RUN_DIR="$RUN_DIR" \
          slurm/submit_train.sh "$BASE" "$OVERLAY")
    echo "  link $i/$N: job $jid ${prev:+(after $prev)}"
    prev="$jid"
done
echo "Done. The chain stops early on its own once n_epochs is reached"
echo "(later links resume to an already-complete state and exit fast)."
