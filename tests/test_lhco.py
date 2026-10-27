"""LHCO (m_X, m_Y) task family: feature definitions, grid splits, and the conditioning arms' data paths.

None of these tests needs the datasets."""

import numpy as np
import pandas as pd
import pytest
import torch

from qmaml_hep.classification.lhco import FEATS, features
from qmaml_hep.experiments.lhco_routes import ARMS, Arm, adapt, fs_modes, grid_split, nearest_of, rejection

GRID = [(a, b) for a in range(50, 601, 50) for b in range(50, 601, 50)]


def _two_jets(p1, m1, t1, p2, m2, t2):
    row = {}
    for j, (p, m, t) in zip("12", ((p1, m1, t1), (p2, m2, t2))):
        row.update({f"pxj{j}": p[0], f"pyj{j}": p[1], f"pzj{j}": p[2], f"mj{j}": m})
        row.update({f"tau{k}j{j}": t[k - 1] for k in (1, 2, 3)})
    return pd.DataFrame([row])


def test_dijet_mass_is_the_invariant_mass():
    p1, p2 = np.array([1500.0, 200.0, -300.0]), np.array([-1400.0, -150.0, 800.0])
    d = _two_jets(p1, 100.0, (0.3, 0.1, 0.05), p2, 500.0, (0.5, 0.2, 0.1))
    e1, e2 = np.sqrt(p1 @ p1 + 100.0**2), np.sqrt(p2 @ p2 + 500.0**2)
    p = p1 + p2
    assert features(d)["mjj"].iloc[0] == pytest.approx(np.sqrt((e1 + e2) ** 2 - p @ p), rel=1e-12)


def test_features_do_not_depend_on_jet_order():
    a = _two_jets([1500.0, 0, 0], 100.0, (0.3, 0.1, 0.05), [-1500.0, 0, 0], 500.0, (0.5, 0.2, 0.1))
    b = _two_jets([-1500.0, 0, 0], 500.0, (0.5, 0.2, 0.1), [1500.0, 0, 0], 100.0, (0.3, 0.1, 0.05))
    fa, fb = features(a)[FEATS].to_numpy(), features(b)[FEATS].to_numpy()
    np.testing.assert_allclose(fa, fb)
    f = features(a).iloc[0]
    assert f.mj_lo == 100.0 and f.dm == 400.0
    assert f.tau21_lo == pytest.approx(0.1 / 0.3) and f.tau21_hi == pytest.approx(0.2 / 0.5)


def test_tau_ratios_cancel_a_common_normalisation():
    a = _two_jets([1500.0, 0, 0], 100.0, (0.3, 0.1, 0.05), [-1500.0, 0, 0], 500.0, (0.5, 0.2, 0.1))
    b = _two_jets([1500.0, 0, 0], 100.0, (300.0, 100.0, 50.0), [-1500.0, 0, 0], 500.0, (500.0, 200.0, 100.0))
    np.testing.assert_allclose(features(a)[FEATS].to_numpy(), features(b)[FEATS].to_numpy(), rtol=1e-6)


@pytest.mark.parametrize("spacing", [100, 150, 200])
def test_grid_split_is_a_partition_with_mirror_pairs_together(spacing):
    tr, va, te = grid_split(GRID, spacing)
    assert len(tr) + len(va) + len(te) == len(GRID)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert set(tr) == {(a, b) for a in range(50, 601, spacing) for b in range(50, 601, spacing)}
    for m in va:
        assert (m[1], m[0]) not in set(te)
    assert 0.2 < len(va) / (len(va) + len(te)) < 0.4


def test_nearest_is_deterministic_on_ties():
    tr = [(50, 50), (150, 50)]
    assert nearest_of((100, 50), tr) == (50, 50)


@pytest.mark.parametrize("arm", [a for a in ARMS if a != "nearest"])
def test_every_arm_gives_two_logits_and_trains_its_conditioning(arm):
    model = Arm(arm, seed=0)
    x = torch.randn(5, 6)
    out = model(x, (100, 400))
    assert out.shape == (5, 2) and out.dtype == torch.float32
    out.sum().backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    if arm in ("qmaml", "qmaml_deep"):
        assert model.theta((100, 400)).dtype == torch.float64
        assert not model.pqc.weights.requires_grad
        assert any(n.startswith("learner") and g.abs().sum() > 0 for n, g in grads.items())


@pytest.mark.parametrize("arm", ["param", "qmaml", "hyper_enc", "param_deep", "qmaml_deep", "classical"])
def test_conditioned_arms_depend_on_the_masses(arm):
    model = Arm(arm, seed=0)
    x = torch.randn(5, 6)
    with torch.no_grad():
        assert not torch.allclose(model(x, (100, 400)), model(x, (500, 150)))


@pytest.mark.parametrize("arm", ["joint", "joint_deep"])
def test_unconditioned_arms_ignore_the_masses(arm):
    model = Arm(arm, seed=0)
    x = torch.randn(5, 6)
    with torch.no_grad():
        assert torch.equal(model(x, (100, 400)), model(x, (500, 150)))


def test_rejection_metric():
    y = np.r_[np.ones(100), np.zeros(1000)]
    auc, sic, rej = rejection(y, np.r_[np.ones(100), np.zeros(1000)])
    assert auc == 1.0 and rej == 1000.0  # perfect separation: capped at 1 / n_background
    rng = np.random.default_rng(0)
    auc, _, rej = rejection(y, rng.random(1100))
    assert 0.4 < auc < 0.6 and rej < 10


class _FakeData:
    """Stands in for lhco_routes.Data: Gaussian signal per mass point and Gaussian QCD."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.sig = {(100, 400): (rng.normal(1.0, 1.0, (300, 6)).astype(np.float32), None)}
        self.q_train = rng.normal(0.0, 1.0, (1000, 6)).astype(np.float32)

    def norm(self, a):
        return torch.tensor(a, dtype=torch.float32)


def test_fewshot_events_depend_only_on_seed_mass_and_k():
    data, m = _FakeData(), (100, 400)
    seen = []
    for arm in ("joint", "classical"):
        orig = data.norm
        data.norm = lambda a, orig=orig: (seen.append(a.copy()), orig(a))[1]
        adapt(Arm(arm, seed=0), data, m, 16, "all", 1e-3, seed=3)
        data.norm = orig
    np.testing.assert_array_equal(seen[0], seen[1])
    assert seen[0].shape == (32, 6)


def test_theta_adaptation_changes_only_theta():
    data, m = _FakeData(), (100, 400)
    model = Arm("qmaml", seed=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    adapted, theta = adapt(model, data, m, 16, "theta", 1e-2, seed=0)
    assert all(torch.equal(p, before[n]) for n, p in adapted.named_parameters())
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())  # the original is untouched
    assert theta is not None and not torch.allclose(theta, model.theta(m).detach())


def test_full_adaptation_changes_the_copy_not_the_original():
    data, m = _FakeData(), (100, 400)
    model = Arm("param_deep", seed=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    adapted, theta = adapt(model, data, m, 16, "all", 1e-2, seed=0)
    assert theta is None
    assert any(not torch.equal(p, before[n]) for n, p in adapted.named_parameters())
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_only_qmaml_has_a_theta_mode():
    assert fs_modes("qmaml") == ["all", "theta"]
    assert all(fs_modes(a) == ["all"] for a in ARMS if a != "qmaml")
