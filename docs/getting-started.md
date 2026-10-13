# Getting started

## Install

```bash
git clone https://github.com/arnavk23/qmaml-hep && cd qmaml-hep
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plots]"
```

Only CPU is needed. On Linux, `pip install torch --index-url https://download.pytorch.org/whl/cpu` beforehand
avoids downloading the CUDA build.

## Run the tests

```bash
make test        # fast suite: physics, simulators vs PennyLane, gradient flow, statistics, CLI
make test-all    # also the end-to-end reproduction and the rescoring regression test
```

## Run one track

```bash
qmaml-benchmark --track J_lmg --seeds 0          # or: python -m qmaml_hep.benchmark ...
```

This pretrains the Learner, collects warm-start optima, adapts every initializer on 6 held-out tasks, and writes
`results/unified/J_lmg_interp_h2_s0.json`. It takes about 3 minutes on a laptop CPU. Useful flags:

| Flag | Effect |
|---|---|
| `--split extrap` | Train on the lowest 60% of the first descriptor, test on the top 25% |
| `--qubits N --depth L` | Larger system or deeper ansatz, where the track supports it |
| `--epochs B --only qmaml` | Compute-matched Q-MAML (pretraining evaluations equal the warm-start collection) |
| `--only a,b,c` | Run a subset of initializers |
| `--adapt_iters`, `--pretrain_lr` | Override the track's budget or learning rate |

Results go to `$QMAML_RESULTS` (default `results/`), datasets to `$QMAML_DATA` (default `data/`).

## Score and plot

```bash
qmaml-score                       # results/unified/*.json -> results/unified_summary.json
python scripts/make_figures.py    # figures/*.pdf
python scripts/make_tables.py     # figures/tables/*.tex
```
