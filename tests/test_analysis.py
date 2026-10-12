"""The fixed scoring rule: iterations-to-threshold, hierarchical bootstrap, Holm correction, verdicts."""

import numpy as np

from qmaml_hep.analysis import TAU, hboot, holm, score_group, task_metrics


def _run(seed, E_by_scheme, E0=-1.0, B=10):
    return {
        "seed": seed,
        "cfg": {"adapt_iters": B},
        "train_tasks": [[0.0]],
        "name": "toy",
        "qubits": 2,
        "depth": 1,
        "n_params": {"all": 6},
        "desc_dim": 1,
        "n_hidden": 2,
        "pretrain": {"loss": [0.0] * 12, "evals": 10},
        "warm_collect": {"evals": 20},
        "test": [{"E0": E0, "scale": 1.0, "runs": {s: {"E": E, "grad0": 1.0} for s, E in E_by_scheme.items()}}],
    }


def test_iterations_to_threshold_and_censoring():
    fast = [-0.5, -0.99, -0.99999, -1.0]  # reaches 1e-3 at t = 2
    never = [-0.5] * 4
    m = task_metrics(_run(0, {"qmaml": fast, "zero": never}, B=3))
    assert m["qmaml"]["T"] == [2]
    assert m["zero"]["T"] == [4]  # censored at B + 1
    assert abs(m["zero"]["final"][0] - 0.5) < 1e-12
    assert TAU == 1e-3


def test_holm_is_monotone_and_bounded():
    adj = holm(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all(adj >= [0.01, 0.04, 0.03, 0.5]) and np.all(adj <= 1)
    assert adj[0] == 0.04  # 4 x 0.01


def test_bootstrap_ci_contains_mean_and_signs_effect():
    rng = np.random.default_rng(0)
    m, lo, hi, p = hboot([np.full(5, -1.0) + rng.normal(0, 0.01, 5) for _ in range(4)], rng)
    assert lo <= m <= hi < 0 and p < 0.05


def test_win_when_qmaml_converges_first_on_every_seed():
    E = {"qmaml": [-0.5, -1.0] + [-1.0] * 9, "uniform": [-0.5] * 6 + [-1.0] * 5, "ws_mean": [-0.5] * 8 + [-1.0] * 3}
    s = score_group([_run(k, E) for k in range(4)], np.random.default_rng(0))
    assert s["comparisons"]["classical"]["verdict"] == "win"
    assert s["comparisons"]["warm"]["verdict"] == "win"
