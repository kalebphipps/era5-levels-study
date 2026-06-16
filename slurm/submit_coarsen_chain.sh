#!/usr/bin/env bash
# Chain N coarsen jobs that keep resuming until the 1.5deg zarr is complete.
set -euo pipefail

N="${1:-4}"
: "${IN:?Set IN=/path/to/era5_37level_0p25.zarr}"
: "${OUT:?Set OUT=/path/to/era5_37level_1p5.zarr}"
mkdir -p logs

echo "Chaining $N coarsen links: $IN -> $OUT"
prev=""
for i in $(seq 1 "$N"); do
    dep=""
    [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    jid=$(sbatch --parsable $dep --export=ALL slurm/coarsen_cpu.sh)
    echo "  link $i/$N: job $jid ${prev:+(after $prev)}"
    prev="$jid"
done
echo "Each link resumes from OUT and exits fast once complete."
