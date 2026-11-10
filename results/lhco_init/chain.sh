#!/usr/bin/env bash
# initialise-then-optimise stage: tune -> select -> report -> score (from the repository root; skips finished jobs)
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=".venv/bin/python -m qmaml_hep.experiments.lhco_init"
par=${1:-4}
runstage() {
  $PY jobs --stage "$1" | while read -r cmd; do
    tag=$(sed -E 's/.*--stage ([a-z]+) --start ([a-z]+) --lr ([0-9.e-]+) --seed ([0-9]+).*/\1_\2_lr\3_s\4/' <<<"$cmd")
    [ -f "results/lhco_init/${tag}.json" ] || echo "$cmd --threads 1 > results/lhco_init/logs/${tag}.log 2>&1"
  done | xargs -P "$par" -I{} bash -c '{}'
}
echo "tune start $(date)"; runstage tune
$PY select || exit 1
echo "report start $(date)"; runstage report
$PY score > results/lhco_init/logs/score.log 2>&1
echo "chain done $(date)"
