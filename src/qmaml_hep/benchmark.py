"""
Unified benchmark for the ten Hamiltonian tracks (round-2 review protocol).

One harness, one protocol, every track:
  * Learner: the original paper's architecture (descriptor -> 256 -> 256 -> P, ReLU, float64), Algorithm 1
    (energy back-propagated through the circuit, no inner loop), output scaled by s=0.3.
  * Adaptation: identical Adam loop (same lr, same iteration budget) for every initializer on a track.
  * Initializers: Q-MAML; classical Zero / pi / Uniform / Gaussian (plus the track's physics reference state
    where one exists); and four *warm-start* baselines built from optimized training-task parameters:
      ws_mean     mean of the training-task optima
      ws_nearest  optimum of the nearest training task (z-scored descriptor)
      ws_linear   ridge-regularized linear regression of optima on the descriptor
      ws_kernel   RBF kernel ridge regression of optima on the descriptor
    Training-task optima are collected by continuation (tasks ordered along a greedy nearest-neighbour chain
    in descriptor space; the first starts from a Uniform draw, each later one from its predecessor's optimum),
    each with the same adaptation budget as a held-out task. Their cost is recorded for compute accounting.
  * Splits: 'interp' draws training and held-out tasks independently from the same ranges (in-distribution);
    'extrap' draws a pool, sorts it on the first continuous descriptor coordinate, trains on the lowest 60%
    and tests on the highest 25%.
  * Recorded per held-out task and initializer: the full energy trajectory, exact E0, energy and gradient
    norm at the initial point, wall time. Metrics (relative error, iterations-to-threshold, verdicts, CIs)
    are computed offline by qmaml_hep.analysis so the scoring rule is fixed independently of the runs.

All energies are exact statevector expectation values (no shot noise); cost is counted in optimizer
iterations (one energy + gradient evaluation each).

Usage:
    python -m qmaml_hep.benchmark --track A_schwinger --seeds 0,1,2,3,4 --split interp
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from qmaml_hep.learner import Learner
from qmaml_hep.paths import RESULTS
from qmaml_hep.tracks import TRACKS, check_fast, make_fast

RES = os.path.join(RESULTS, "unified")
W0_SCALE = 0.3
CLIP = 5.0


CLASSICAL = ["zero", "pi", "uniform", "gaussian"]
WARM = ["ws_mean", "ws_nearest", "ws_linear", "ws_kernel"]


# ---------------------------------------------------------------------------------------------------------
# Shared loops (identical for every initializer)
# ---------------------------------------------------------------------------------------------------------
def adapt(pqc, theta0, H, iters, lr, cost_kw=None):
    """Algorithm 2. Returns (theta_final, energies[iters+1], grad_norm_at_init). energies[0] is E(theta0)."""
    cost_kw = cost_kw or {}
    theta = nn.Parameter(theta0.clone().detach().double(), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    energies, g0 = [], None
    for it in range(iters):
        opt.zero_grad()
        c = pqc.cost(theta, H, **cost_kw)
        c.backward()
        if it == 0:
            g0 = float(theta.grad.detach().norm())
        energies.append(float(c.item()))
        opt.step()
    with torch.no_grad():
        energies.append(float(pqc.cost(theta, H, **cost_kw).item()))
    return theta.detach(), energies, g0


def pretrain(tr, train_tasks, hams, epochs, lr, n_hidden=2, seed=0):
    """Algorithm 1: one Learner per parameter-shape group (only the nuclear track has >1 group)."""
    torch.manual_seed(seed)
    groups = sorted({tr["group"](t) for t in train_tasks})
    d_in = len(tr["desc"](train_tasks[0]))
    learners = {g: Learner(d_in, tuple(tr["shape"](g)), hidden=256, n_hidden=n_hidden).double() for g in groups}
    params = [p for l in learners.values() for p in l.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    loss_hist, grad_hist = [], []
    for _ in range(epochs):
        el, eg = 0.0, []
        for t, H in zip(train_tasks, hams):
            g = tr["group"](t)
            opt.zero_grad()
            theta = W0_SCALE * learners[g](tr["desc"](t))
            c = tr["pqc"](g).cost(theta, H)
            c.backward()
            eg.append(float(sum(p.grad.pow(2).sum() for p in params if p.grad is not None) ** 0.5))
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            el += c.item()
        loss_hist.append(el / len(train_tasks))
        grad_hist.append(float(np.mean(eg)))
    return learners, {"loss": loss_hist, "grad_norm": grad_hist}


# ---------------------------------------------------------------------------------------------------------
# Warm-start baselines
# ---------------------------------------------------------------------------------------------------------
def _nn_chain(X):
    """Greedy nearest-neighbour ordering starting from the point closest to the centroid."""
    n = len(X)
    left = set(range(n))
    cur = int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))
    order = [cur]
    left.remove(cur)
    while left:
        cand = list(left)
        cur = cand[int(np.argmin(((X[cand] - X[cur]) ** 2).sum(1)))]
        order.append(cur)
        left.remove(cur)
    return order


def collect_train_optima(tr, train_tasks, hams, e0s, iters, lr, seed):
    """Continuation over training tasks (per shape group). Returns {group: (X_desc, Theta)} and bookkeeping."""
    out, book = {}, {"iters": 0, "final_rel_err": []}
    gen = torch.Generator().manual_seed(10_000 + seed)
    for g in sorted({tr["group"](t) for t in train_tasks}):
        idx = [i for i, t in enumerate(train_tasks) if tr["group"](t) == g]
        X = np.stack([tr["desc"](train_tasks[i]).numpy() for i in idx])
        sh = tuple(tr["shape"](g))
        theta = (torch.rand(sh, generator=gen, dtype=torch.float64) * 2 - 1) * 0.05 * np.pi
        thetas = {}
        for j in _nn_chain(X):
            i = idx[j]
            theta, en, _ = adapt(tr["pqc"](g), theta, hams[i], iters, lr)
            book["iters"] += iters
            book["final_rel_err"].append(abs(en[-1] - e0s[i]) / max(abs(e0s[i]), 1e-8))
            thetas[j] = theta.clone()
        out[g] = (X, torch.stack([thetas[j] for j in range(len(idx))]).numpy())
    return out, book


class WarmStarts:
    def __init__(self, optima):
        self.m = {}
        for g, (X, T) in optima.items():
            mu, sd = X.mean(0), X.std(0)
            sd[sd < 1e-12] = 1.0
            Z = (X - mu) / sd
            Y = T.reshape(len(T), -1)
            # linear ridge
            A = np.hstack([Z, np.ones((len(Z), 1))])
            W_lin = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Y)
            # RBF kernel ridge, length scale = median pairwise distance
            D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
            ell = float(np.median(D[D > 0])) if (D > 0).any() else 1.0
            K = np.exp(-(D**2) / (2 * ell**2))
            ybar = Y.mean(0)
            alpha = np.linalg.solve(K + 1e-3 * np.eye(len(K)), Y - ybar)
            self.m[g] = dict(mu=mu, sd=sd, Z=Z, Y=Y, shape=T.shape[1:], W_lin=W_lin, ell=ell, alpha=alpha, ybar=ybar)

    def __call__(self, kind, g, x):
        m = self.m[g]
        z = (x - m["mu"]) / m["sd"]
        if kind == "ws_mean":
            y = m["Y"].mean(0)
        elif kind == "ws_nearest":
            y = m["Y"][int(np.argmin(((m["Z"] - z) ** 2).sum(1)))]
        elif kind == "ws_linear":
            y = np.append(z, 1.0) @ m["W_lin"]
        elif kind == "ws_kernel":
            k = np.exp(-((m["Z"] - z) ** 2).sum(1) / (2 * m["ell"] ** 2))
            y = m["ybar"] + k @ m["alpha"]
        else:
            raise ValueError(kind)
        return torch.tensor(y.reshape(m["shape"]), dtype=torch.float64)


# ---------------------------------------------------------------------------------------------------------
def split_tasks(tr, split, seed):
    n_tr, n_te = tr["cfg"]["n_train"], tr["cfg"]["n_test"]
    if split == "interp":
        return tr["sample"](n_tr, seed), tr["sample"](n_te, seed + 1000)
    pool = tr["sample"](120, seed + 5000)
    # first continuous descriptor coordinate = first coordinate that is not a one-hot entry
    D = np.stack([tr["desc"](t).numpy() for t in pool])
    col = next(j for j in range(D.shape[1]) if len(np.unique(D[:, j])) > 2)
    order = np.argsort(D[:, col])
    lo, hi = order[: int(0.6 * len(pool))], order[int(0.75 * len(pool)) :]
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.choice(lo, n_tr, replace=False)], [
        pool[i] for i in rng.choice(hi, n_te, replace=False)
    ]


def run_seed(track_key, seed, split, n_hidden=2, track_kw=None, epochs=None, overrides=None, only=None):
    tr = TRACKS[track_key](**(track_kw or {}))
    orig_pqc = make_fast(tr)
    cfg = dict(tr["cfg"])
    if epochs:
        cfg["pretrain_epochs"] = epochs
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, test = split_tasks(tr, split, seed)
    htr = [tr["ham"](t) for t in train]
    hte = [tr["ham"](t) for t in test]
    e0tr = [tr["e0"](t, H) for t, H in zip(train, htr)]
    e0te = [tr["e0"](t, H) for t, H in zip(test, hte)]
    if isinstance(orig_pqc, dict):
        for g, o in orig_pqc.items():
            i = next((i for i, t in enumerate(train) if tr["group"](t) == g), None)
            if i is not None:
                check_fast(tr["pqc"](g), o, htr[i], tr["shape"](g))
    elif orig_pqc is not None:
        kws = [{}] + [kwf(test[0]) for _, kwf in tr["extra"].values()]
        check_fast(tr["pqc"]("all"), orig_pqc, htr[0], tr["shape"]("all"), kws)

    t0 = time.time()
    learners, hist = pretrain(tr, train, htr, cfg["pretrain_epochs"], cfg["pretrain_lr"], n_hidden, seed)
    t_pre = time.time() - t0
    drop = set(tr.get("drop", ()))
    schemes = ["qmaml"] + [c for c in CLASSICAL if c not in drop] + list(tr["extra"]) + WARM
    if only:
        schemes = [s for s in schemes if s in only]
    t0 = time.time()
    if any(s in WARM for s in schemes):
        optima, book = collect_train_optima(tr, train, htr, e0tr, cfg["adapt_iters"], cfg["adapt_lr"], seed)
        ws = WarmStarts(optima)
    else:
        book = {"iters": 0, "final_rel_err": []}
    t_ws = time.time() - t0
    per_task = []
    torch.manual_seed(20_000 + seed)
    for t, H, E0 in zip(test, hte, e0te):
        g = tr["group"](t)
        x = tr["desc"](t)
        row = {
            "task": [str(v) if isinstance(v, str) else float(v) for v in t],
            "E0": E0,
            "scale": float(tr["err_scale"](t, E0)) if "err_scale" in tr else abs(E0),
            "runs": {},
        }
        for s in schemes:
            kw = {}
            if s == "qmaml":
                with torch.no_grad():
                    th0 = W0_SCALE * learners[g](x)
            elif s in WARM:
                th0 = ws(s, g, x.numpy())
            elif s in tr["extra"]:
                kind, kwf = tr["extra"][s]
                th0 = tr["classical"](tr["shape"](g), kind)
                kw = kwf(t)
            else:
                th0 = tr["classical"](tr["shape"](g), s)
            ts = time.time()
            _, en, g0 = adapt(tr["pqc"](g), th0, H, cfg["adapt_iters"], cfg["adapt_lr"], kw)
            row["runs"][s] = {"E": en, "grad0": g0, "sec": time.time() - ts}
        per_task.append(row)
        print(
            f"  [{tr['name']} {split} seed={seed}] task done: "
            + " ".join(f"{s}={abs(row['runs'][s]['E'][-1] - E0) / max(row['scale'], 1e-12):.1e}" for s in schemes),
            flush=True,
        )

    n_params = {g: int(np.prod(tr["shape"](g))) for g in sorted({tr["group"](t) for t in train + test})}
    return {
        "track": track_key,
        "name": tr["name"],
        "seed": seed,
        "split": split,
        "qubits": tr["qubits"],
        "depth": tr["depth"],
        "n_params": n_params,
        "desc_dim": int(len(tr["desc"](train[0]))),
        "n_hidden": n_hidden,
        "cfg": cfg,
        "train_tasks": [[str(v) if isinstance(v, str) else float(v) for v in t] for t in train],
        "train_E0": e0tr,
        "pretrain": {**hist, "sec": t_pre, "evals": cfg["pretrain_epochs"] * len(train)},
        "warm_collect": {"sec": t_ws, "evals": book["iters"], "train_final_rel_err": book["final_rel_err"]},
        "test": per_task,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--split", default="interp", choices=["interp", "extrap"])
    ap.add_argument("--n_hidden", type=int, default=2)
    ap.add_argument("--qubits", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--pretrain_lr", type=float, default=None)
    ap.add_argument("--adapt_iters", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated subset of initializers to run")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    kw = {}
    if a.qubits:
        kw["num_qubits"] = a.qubits
    if a.depth:
        kw["depth"] = a.depth
    os.makedirs(RES, exist_ok=True)
    for s in (int(x) for x in a.seeds.split(",")):
        tag = (
            f"{a.track}{'_q' + str(a.qubits) if a.qubits else ''}{'_d' + str(a.depth) if a.depth else ''}"
            f"_{a.split}_h{a.n_hidden}{'_e' + str(a.epochs) if a.epochs else ''}{a.tag}_s{s}"
        )
        path = os.path.join(RES, tag + ".json")
        if os.path.exists(path):
            print("skip (exists)", path)
            continue
        t0 = time.time()
        out = run_seed(
            a.track,
            s,
            a.split,
            a.n_hidden,
            kw,
            a.epochs,
            {"pretrain_lr": a.pretrain_lr, "adapt_iters": a.adapt_iters},
            a.only.split(",") if a.only else None,
        )
        out["wall_sec"] = time.time() - t0
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"saved {path}  ({out['wall_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
