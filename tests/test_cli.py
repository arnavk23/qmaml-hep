"""The command-line entry points: a tiny end-to-end benchmark run writes a well-formed result, and the analysis
re-scores the committed results to exactly the committed summary."""

import json
import os
import sys

import pytest

import qmaml_hep.analysis as A
import qmaml_hep.benchmark as B

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_benchmark_cli_writes_a_complete_result(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "RES", str(tmp_path))
    argv = ["qmaml_hep.benchmark", "--track", "J_lmg", "--seeds", "0", "--epochs", "1", "--adapt_iters", "4",
            "--only", "qmaml,zero,uniform,ws_nearest,ws_kernel"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    B.main()
    (path,) = tmp_path.iterdir()
    r = json.loads(path.read_text())
    assert r["track"] == "J_lmg" and r["seed"] == 0 and r["cfg"]["adapt_iters"] == 4
    assert len(r["train_tasks"]) == r["cfg"]["n_train"] and len(r["test"]) == r["cfg"]["n_test"]
    assert r["pretrain"]["evals"] == r["cfg"]["n_train"]
    assert r["warm_collect"]["evals"] == r["cfg"]["n_train"] * 4
    for row in r["test"]:
        assert set(row["runs"]) == {"qmaml", "zero", "uniform", "ws_nearest", "ws_kernel"}
        for run in row["runs"].values():
            assert len(run["E"]) == 5 and run["grad0"] >= 0
            assert min(run["E"]) >= row["E0"] - 1e-9  # variational principle

    # a second invocation skips the existing file instead of overwriting it
    B.main()
    assert len(list(tmp_path.iterdir())) == 1


def test_benchmark_rejects_unknown_track(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qmaml_hep.benchmark", "--track", "Z_nope"])
    with pytest.raises(SystemExit):
        B.main()


@pytest.mark.slow
def test_rescoring_committed_results_reproduces_the_summary(tmp_path, monkeypatch):
    committed = json.load(open(os.path.join(ROOT, "results", "unified_summary.json")))
    monkeypatch.setattr(A, "RES", os.path.join(ROOT, "results", "unified"))
    monkeypatch.setattr(A, "RESULTS", str(tmp_path))
    fresh = A.main()
    assert set(fresh) == set(committed)
    for group, s in committed.items():
        for fam, c in s["comparisons"].items():
            f = fresh[group]["comparisons"][fam]
            assert (f["verdict"], f["opponent"]) == (c["verdict"], c["opponent"]), (group, fam)
            assert abs(f["mean_d"] - c["mean_d"]) < 1e-9 and abs(f["p_holm"] - c["p_holm"]) < 1e-9
