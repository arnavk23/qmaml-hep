"""The cross-platform comparison: substituting rerun Q-MAML trajectories must touch nothing else, and identical
reruns must reproduce the stored verdicts exactly."""

import copy
import json
import os
import shutil

import pytest

from qmaml_hep.experiments import cross_platform as X

STORED = os.path.join(os.path.dirname(__file__), "..", "results", "unified")
SAMPLE = "J_lmg_interp_h2_s0.json"


@pytest.fixture
def stored_run():
    path = os.path.join(STORED, SAMPLE)
    if not os.path.exists(path):
        pytest.skip("raw results not present")
    return json.load(open(path))


def test_substitute_replaces_only_qmaml(stored_run):
    rerun = copy.deepcopy(stored_run)
    for row in rerun["test"]:
        row["runs"]["qmaml"]["E"] = [e + 1.0 for e in row["runs"]["qmaml"]["E"]]
    out = X.substitute(stored_run, rerun)
    for row_out, row_old, row_new in zip(out["test"], stored_run["test"], rerun["test"]):
        assert row_out["runs"]["qmaml"] == row_new["runs"]["qmaml"]
        for s in row_old["runs"]:
            if s != "qmaml":
                assert row_out["runs"][s] == row_old["runs"][s]


def test_substitute_refuses_a_different_seed(stored_run):
    other = copy.deepcopy(stored_run)
    other["seed"] += 1
    with pytest.raises(AssertionError):
        X.substitute(stored_run, other)


def test_identical_rerun_reproduces_every_verdict(tmp_path, stored_run):
    for s in range(5):
        name = f"J_lmg_interp_h2_s{s}.json"
        shutil.copy(os.path.join(STORED, name), tmp_path / name)
    res = X.compare(stored_dir=STORED, rerun_dir=str(tmp_path))
    r = res["J_lmg_interp_h2"]
    assert r["n_seeds"] == 5
    for c in r["comparisons"].values():
        assert c["verdict"][0] == c["verdict"][1] and c["mean_d"][0] == c["mean_d"][1]
    assert all(s["T_stored"] == s["T_rerun"] for s in r["per_seed"])
