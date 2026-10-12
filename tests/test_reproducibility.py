"""End-to-end: rerun one stored benchmark seed and compare with the published result file.

Every task-independent and warm-start initializer is deterministic given the seed and reproduces the stored
trajectories across machines. Q-MAML's Learner pretraining is not bit-reproducible across BLAS builds (see
docs/reproducibility.md), so it is compared only on the same machine, run twice.
"""

import json
import os

import numpy as np
import pytest

STORED = os.path.join(os.path.dirname(__file__), "..", "results", "unified", "J_lmg_interp_h2_s0.json")


@pytest.mark.slow
def test_lmg_seed0_matches_stored_result(tmp_path, monkeypatch):
    if not os.path.exists(STORED):
        pytest.skip("raw results not present (see results/README.md)")
    import qmaml_hep.benchmark as bm

    new = bm.run_seed("J_lmg", 0, "interp")
    old = json.load(open(STORED))
    assert new["train_tasks"] == old["train_tasks"]
    assert np.allclose(new["train_E0"], old["train_E0"], atol=1e-12)
    for t_new, t_old in zip(new["test"], old["test"]):
        assert t_new["task"] == t_old["task"]
        for s in t_old["runs"]:
            if s == "qmaml":
                continue
            assert np.abs(np.subtract(t_new["runs"][s]["E"], t_old["runs"][s]["E"])).max() < 1e-8, s
