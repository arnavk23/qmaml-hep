#!/usr/bin/env bash
# after tuning: select lrs, run the report stage, fit the oracle BDTs, score each spacing
set -eu
cd "$(dirname "$0")/../.."
while pgrep -f "run_jobs.sh tune" >/dev/null; do sleep 30; done
n=$(ls results/lhco_routes/tune_*.json | wc -l); echo "tune files $n $(date)"
[ "$n" -ge 54 ] || { echo "tuning incomplete"; exit 1; }
PY="env PYTHONPATH=src .venv/bin/python -m qmaml_hep.experiments.lhco_routes"
$PY select
results/lhco_routes/run_jobs.sh report 6
$PY oracle > results/lhco_routes/logs/oracle.log 2>&1
for sp in 100 150 200; do $PY score --spacing $sp; done
echo "chain done $(date)"
