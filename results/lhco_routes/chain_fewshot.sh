#!/usr/bin/env bash
# few-shot stage: tune the adaptation learning rate on validation points, select it, run five seeds, score
set -eu
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="env PYTHONPATH=src .venv/bin/python -m qmaml_hep.experiments.lhco_routes"
echo "fstune start $(date)"
results/lhco_routes/run_jobs.sh fstune 6
n=$(ls results/lhco_routes/fstune_*.json | wc -l); [ "$n" -ge 18 ] || { echo "fstune incomplete ($n)"; exit 1; }
$PY select-fs
results/lhco_routes/run_jobs.sh fewshot 6
for sp in 100 150; do $PY score-fewshot --spacing $sp; done
echo "chain done $(date)"
