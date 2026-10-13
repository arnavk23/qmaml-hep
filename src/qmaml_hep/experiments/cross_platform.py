"""Cross-platform reproducibility of Q-MAML verdicts.

Every initializer except Q-MAML reproduces exactly across numerical environments (see docs/reproducibility.md), so
a rerun only needs to repeat Q-MAML: pretraining and adaptation, ``qmaml-benchmark ... --only qmaml``. This module
substitutes the rerun Q-MAML trajectories into the stored result files, rescored with the unchanged rule, and
compares per-task iterations to threshold, per-seed statistics and verdicts between the two environments.

Usage:
    QMAML_RESULTS=results/cross_platform/linux qmaml-benchmark --track J_lmg --seeds 0,1,2,3,4 --only qmaml
    python -m qmaml_hep.experiments.cross_platform
"""

import copy
import glob
import json
import os

import numpy as np

from qmaml_hep import analysis as A
from qmaml_hep.paths import RESULTS

STORED = os.path.join(RESULTS, "unified")
RERUN = os.path.join(RESULTS, "cross_platform", "linux", "unified")
OUT = os.path.join(RESULTS, "cross_platform", "summary.json")


def substitute(stored_run, rerun):
    """A copy of `stored_run` with its Q-MAML trajectories replaced by those of `rerun` (same track, seed, tasks)."""
    assert (stored_run["track"], stored_run["seed"], stored_run["split"]) == (
        rerun["track"],
        rerun["seed"],
        rerun["split"],
    )
    assert stored_run["train_tasks"] == rerun["train_tasks"], "different training tasks: not the same seed"
    out = copy.deepcopy(stored_run)
    for row, row_new in zip(out["test"], rerun["test"]):
        assert row["task"] == row_new["task"]
        row["runs"]["qmaml"] = row_new["runs"]["qmaml"]
    out["pretrain"] = rerun["pretrain"]
    return out


def compare(stored_dir=STORED, rerun_dir=RERUN):
    groups = {}
    for path in sorted(glob.glob(os.path.join(rerun_dir, "*.json"))):
        key = os.path.basename(path).rsplit("_s", 1)[0]
        groups.setdefault(key, []).append(path)
    out = {}
    for key, paths in groups.items():
        pairs = []
        for p in paths:
            new = json.load(open(p))
            old = json.load(open(os.path.join(stored_dir, os.path.basename(p))))
            pairs.append((old, substitute(old, new)))
        stored_runs = [a for a, _ in pairs]
        rerun_runs = [b for _, b in pairs]
        per_seed = []
        for old, new in pairs:
            T_old = A.task_metrics(old)["qmaml"]["T"]
            T_new = A.task_metrics(new)["qmaml"]["T"]
            per_seed.append({"seed": old["seed"], "T_stored": T_old, "T_rerun": T_new,
                             "pretrain_loss_stored": old["pretrain"]["loss"][-1],
                             "pretrain_loss_rerun": new["pretrain"]["loss"][-1]})  # fmt: skip
        s_old = A.score_group(stored_runs, np.random.default_rng(0))
        s_new = A.score_group(rerun_runs, np.random.default_rng(0))
        comps = {}
        for fam in ("classical", "warm"):
            a, b = s_old["comparisons"][fam], s_new["comparisons"][fam]
            comps[fam] = {
                "opponent": [a["opponent"], b["opponent"]],
                "mean_d": [a["mean_d"], b["mean_d"]],
                "ci": [a["ci"], b["ci"]],
                "verdict": [a["verdict"], b["verdict"]],
                "per_seed_mean_d": [a["per_seed_mean_d"], b["per_seed_mean_d"]],
            }
        out[key] = {"n_seeds": len(pairs), "per_seed": per_seed, "comparisons": comps}
    return out


def main():
    res = compare()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    for key, r in res.items():
        line = f"{key:24s} n={r['n_seeds']}"
        for fam, c in r["comparisons"].items():
            line += f" | {fam}: {c['verdict'][0]}->{c['verdict'][1]} d={c['mean_d'][0]:+.2f}->{c['mean_d'][1]:+.2f}"
        print(line)
    return res


if __name__ == "__main__":
    main()
