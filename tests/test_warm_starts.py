"""The four warm-start baselines, the continuation chain that collects their training optima, and the splits."""

import numpy as np
import torch

from qmaml_hep.benchmark import WarmStarts, _nn_chain, split_tasks
from qmaml_hep.tracks import TRACKS


def _optima(n=12, d=2, shape=(2, 3), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    W = rng.normal(size=(d, int(np.prod(shape))))
    T = (X @ W).reshape(n, *shape)  # optima exactly linear in the descriptor
    return X, T


def test_nearest_and_mean_reproduce_training_optima():
    X, T = _optima()
    ws = WarmStarts({"all": (X, T)})
    for i in range(len(X)):
        assert torch.allclose(ws("ws_nearest", "all", X[i]), torch.tensor(T[i]))
    assert torch.allclose(ws("ws_mean", "all", X[0]), torch.tensor(T.mean(0)))


def test_linear_and_kernel_fit_linear_optima():
    X, T = _optima()
    ws = WarmStarts({"all": (X, T)})
    x_new = np.array([0.3, -0.2])
    lin = ws("ws_linear", "all", x_new)
    exact = WarmStarts({"all": (X, T)}).m["all"]
    assert lin.dtype == torch.float64 and lin.shape == T.shape[1:]
    # ridge (1e-3) on z-scored, exactly linear data: essentially exact at training points
    assert np.abs(ws("ws_linear", "all", X[0]).numpy() - T[0]).max() < 1e-2
    assert np.abs(ws("ws_kernel", "all", X[0]).numpy() - T[0]).max() < 5e-2
    assert exact["ell"] > 0


def test_nn_chain_visits_every_task_once():
    X = np.random.default_rng(1).normal(size=(15, 3))
    order = _nn_chain(X)
    assert sorted(order) == list(range(15))


def test_extrapolation_split_separates_train_and_test():
    tr = TRACKS["A_schwinger"]()
    train, test = split_tasks(tr, "extrap", seed=0)
    assert len(train) == tr["cfg"]["n_train"] and len(test) == tr["cfg"]["n_test"]
    assert max(t[0] for t in train) < min(t[0] for t in test)


def test_interpolation_split_uses_independent_draws():
    tr = TRACKS["A_schwinger"]()
    train, test = split_tasks(tr, "interp", seed=0)
    assert not set(train) & set(test)
