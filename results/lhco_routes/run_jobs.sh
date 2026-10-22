#!/usr/bin/env bash
# usage: run_jobs.sh <stage> <parallel>   (from the repository root; skips jobs whose output exists)
set -u
stage=$1; par=$2
PYTHONPATH=src .venv/bin/python -m qmaml_hep.experiments.lhco_routes jobs --stage "$stage" \
| sed 's/--threads [0-9]*//; s/$/ --threads 1/' \
| while read -r cmd; do
    arm=$(sed -E 's/.*--arm ([a-z_]+).*/\1/' <<<"$cmd"); lr=$(sed -E 's/.*--lr ([0-9.e-]+).*/\1/' <<<"$cmd")
    sp=$(sed -E 's/.*--spacing ([0-9]+).*/\1/' <<<"$cmd"); s=$(sed -E 's/.*--seed ([0-9]+).*/\1/' <<<"$cmd")
    lrn=$(python3 -c "print(f'{float(\"$lr\"):g}')")
    out="results/lhco_routes/${stage}_sp${sp}_${arm}_lr${lrn}_s${s}.json"
    [ -f "$out" ] || echo "PYTHONPATH=src $cmd > results/lhco_routes/logs/${stage}_sp${sp}_${arm}_lr${lrn}_s${s}.log 2>&1"
  done | xargs -P "$par" -I{} bash -c '{}'
echo "stage $stage done $(date)"
