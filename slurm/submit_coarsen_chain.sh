#!/usr/bin/env bash
# Chain N coarsen jobs that keep resuming until the 1.5deg zarr is complete.
#
# coarsen_to_1p5.py is resumable (it continues from however many timesteps are
# already in OUT), so each link picks up where the previous one stopped/died.
# Links run back-to-back via afterany (start whether the previous finished,
# hit the time limit, or failed).
#
#   export VENV=$WS/venv
#   IN=/path/era5_37level_0p25.zarr OUT=$WS/data/era5_37level_1p5.zarr \
#   START=1990-01-01 END=2022-12-31 STRIDE=6 \
#       bash slurm/submit_coarsen_chain.sh 6        # 6 links of up to 24h each
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
