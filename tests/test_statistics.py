"""Calibration of the statistical machinery: the hierarchical bootstrap and the Holm step-down procedure behave as
their definitions require, on data where the right answer is known."""

import itertools

import numpy as np
import pytest

import qmaml_hep.analysis as A


def _holm_reference(p):
    """Textbook Holm step-down: reject H_(k) while p_(k) <= alpha / (m - k); adjusted p by cumulative max."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


@pytest.mark.parametrize("seed", range(5))
def test_holm_matches_reference(seed):
    p = np.random.default_rng(seed).uniform(0, 0.2, 12)
    assert np.allclose(A.holm(p), _holm_reference(p))


def test_holm_controls_family_wise_error_under_the_global_null():
    rng = np.random.default_rng(0)
    false_rejections = sum((A.holm(rng.uniform(size=10)) < 0.05).any() for _ in range(4000))
    assert false_rejections / 4000 <= 0.06


def test_bootstrap_interval_coverage_under_the_null(monkeypatch):
    """With mean-zero data, the 95% interval should contain 0 in roughly 95% of experiments (bootstrap percentile
    intervals are mildly anti-conservative at 5 seeds, so the band is wide)."""
    monkeypatch.setattr(A, "NBOOT", 400)
    rng = np.random.default_rng(1)
    hits, small_p = 0, 0
    trials = 300
    for _ in range(trials):
        D = [rng.normal(0, 1, 6) + rng.normal(0, 0.5) for _ in range(5)]
        _, lo, hi, p = A.hboot(D, rng)
        hits += lo <= 0 <= hi
        small_p += p < 0.05
    assert 0.85 <= hits / trials <= 0.99
    assert small_p / trials <= 0.15


def test_bootstrap_detects_a_real_effect(monkeypatch):
    monkeypatch.setattr(A, "NBOOT", 400)
    rng = np.random.default_rng(2)
    detected = sum(A.hboot([rng.normal(-1.0, 0.5, 6) for _ in range(5)], rng)[2] < 0 for _ in range(100))
    assert detected >= 95


def test_bootstrap_resamples_seeds_not_just_tasks(monkeypatch):
    """One seed far from the others must widen the interval: seed-level variance is part of the uncertainty."""
    monkeypatch.setattr(A, "NBOOT", 2000)
    rng = np.random.default_rng(3)
    tight = [np.full(6, -1.0) for _ in range(5)]
    spread = [np.full(6, -1.0) for _ in range(4)] + [np.full(6, 2.0)]
    _, lo_t, hi_t, _ = A.hboot(tight, rng)
    _, lo_s, hi_s, _ = A.hboot(spread, rng)
    assert hi_t - lo_t < 1e-12 and hi_s - lo_s > 1.0


@pytest.mark.parametrize(("lo", "hi", "verdict"), [(-0.5, -0.1, "win"), (0.1, 0.4, "loss"), (-0.2, 0.3, "tie")])
def test_verdict_rule(lo, hi, verdict):
    assert ("win" if hi < 0 else ("loss" if lo > 0 else "tie")) == verdict


def test_stationary_initializers_are_never_chosen_as_opponent():
    def run(seed):
        E = {"qmaml": [-0.5] + [-1.0] * 10, "zero": [-1.0] * 11, "uniform": [-0.5] * 5 + [-1.0] * 6}
        r = {
            "seed": seed, "cfg": {"adapt_iters": 10}, "train_tasks": [[0.0]], "name": "toy", "qubits": 2, "depth": 1,
            "n_params": {"all": 6}, "desc_dim": 1, "n_hidden": 2,
            "pretrain": {"loss": [0.0] * 12, "evals": 10}, "warm_collect": {"evals": 20},
            "test": [{"E0": -1.0, "scale": 1.0, "runs": {s: {"E": e, "grad0": 1.0} for s, e in E.items()}}],
        }  # fmt: skip
        r["test"][0]["runs"]["zero"]["grad0"] = 0.0  # exact stationary point: starts on the answer by construction
        return r

    s = A.score_group([run(k) for k in range(3)], np.random.default_rng(0))
    assert "zero" in s["stationary"]
    assert s["comparisons"]["classical"]["opponent"] == "uniform"


def test_every_pair_of_committed_comparisons_has_holm_p_at_least_raw_p():
    import json
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "results", "unified_summary.json")
    summary = json.load(open(path))
    pairs = [(c["p"], c["p_holm"]) for g in summary.values() for c in g["comparisons"].values()]
    assert pairs and all(ph >= p - 1e-15 for p, ph in pairs)
    assert all(a <= 1.0 for a in itertools.chain.from_iterable(pairs))
