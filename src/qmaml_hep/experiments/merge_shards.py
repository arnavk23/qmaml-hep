"""Merge results/paper_replication/<name>_shard{k}of{n}.json into <name>.json (tasks in their original order).
Usage: python -m qmaml_hep.experiments.merge_shards heisenberg_q12_L20"""

import glob
import json
import os
import sys

from qmaml_hep.paths import RESULTS

OUT = os.path.join(RESULTS, "paper_replication")
name = sys.argv[1]
shards = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, f"{name}_shard*of*.json")))]
base = shards[0]
for s in shards[1:]:
    # identical Learner (bit-identical pretraining); exact E0 agrees to the sparse eigensolver's round-off
    assert s["pretrain_loss"] == base["pretrain_loss"], "shards trained different Learners"
    assert max(abs(a - b) for a, b in zip(s["E0"], base["E0"])) < 1e-9, "shards disagree on E0"
order = sorted((i, s, j) for s in shards for j, i in enumerate(s["task_index"]))
merged = {k: base[k] for k in ("E0", "tasks", "pretrain_loss", "train_E0_mean")}
merged["task_index"] = [i for i, _, _ in order]
merged["E0"] = [base["E0"][i] for i in merged["task_index"]]
merged["tasks"] = [base["tasks"][i] for i in merged["task_index"]]
merged["curves"] = {k: [s["curves"][k][j] for _, s, j in order] for k in base["curves"]}
json.dump(merged, open(os.path.join(OUT, f"{name}.json"), "w"))
print(f"merged {len(shards)} shards, {len(order)} tasks")
