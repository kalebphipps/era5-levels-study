#!/usr/bin/env bash
# Submit a CHAIN of training jobs.
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
    jid=$(sbatch --parsable $dep ${ACCOUNT:+--account="$ACCOUNT"} \
          --export=ALL,RUN_DIR="$RUN_DIR" \
          slurm/submit_train.sh "$BASE" "$OVERLAY")
    echo "  link $i/$N: job $jid ${prev:+(after $prev)}"
    prev="$jid"
done
echo "Done. The chain stops early on its own once n_epochs is reached"
echo "(later links resume to an already-complete state and exit fast)."
