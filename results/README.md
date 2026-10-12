# Results

Every number in the figures and tables is computed from the files in this directory by `qmaml_hep.analysis`,
`scripts/make_figures.py` and `scripts/make_tables.py`.

| Path | Contents |
|---|---|
| `unified/<track>[_q<n>][_d<L>]_<split>_h<layers>[_e<epochs>]_s<seed>.json` | One benchmark seed: train tasks and their exact E₀, pretraining loss and gradient history, warm-start collection cost, and for every held-out task and initializer the full energy trajectory, the gradient norm at the start, and wall time |
| `unified_summary.json` | Scored output of `python -m qmaml_hep.analysis`: per track and split, the opponent, mean d, 95% CI, raw and Holm-adjusted p, verdict, and compute accounting |
| `classification_v2/<dataset>_q<n>_{tune,report}.json` | Hyperparameter selection on validation tasks, then test accuracy and log-loss for every arm over 10 seeds |
| `barren_plateau_v2/{higgs,heisenberg}.json` | Gradient variance over live parameters versus qubit count, per initializer |
| `paper_replication/*.json` | Paper-scale runs (12-qubit Heisenberg, 10-qubit LiH): E₀, pretraining loss, 2000-iteration curves |

File-name fields: `h1` and `h2` are the Learner's hidden-layer count (`h2` is the paper's architecture, `h1` the
ablation). `_e<B>` marks compute-matched runs, which contain only the `qmaml` initializer; the analysis merges them
into the base group as `qmaml_cm`.
