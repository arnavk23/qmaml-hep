"""Pilot: Q-MAML on the LHCO (mX, mY) signal grid. Every arm uses the same classifier:
    angles = P(x) (shared Linear 6->6, or 8->6 for param-VQC), FastPQCModel(6 qubits, depth 2), shared head.
Arms differ only in where the circuit weights theta come from:
    joint      one shared theta, no mass information
    param      one shared theta, (mX, mY)/600 appended to the encoding projection input
    qmaml      theta = Learner(mX/600, mY/600)   (d -> 256 -> 256 -> P, paper architecture; Algorithm 1)
    nearest    a separate classifier trained on the nearest training grid point
    pertask    a separate classifier trained on the target point itself (reference, subset of points)
Few-shot (Algorithm 2): K signal + 10K background events at the target (class-weighted 1:10); adapt theta
only, 50 Adam steps; 16 held-out points x 2 seeds x 2 draws.

Train grid: mX, mY in {50, 150, ..., 550} (36 points). Test: every other grid point (mirrors stay together).
Metric: background rejection at eps_s = 0.3 (and AUC), 100k-event QCD test pool.

Run: python lhco_qmaml_pilot.py <data_dir> <out_json>
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tables
import torch
import torch.nn as nn

sys.path.insert(0, "/home/arnav/qmaml-hep/src")
sys.path.insert(0, str(Path(__file__).parent))
from lhco_feasibility import FEATS, features, in_sr, sic_auc  # noqa: E402
from qmaml_hep.learner import Learner  # noqa: E402
from qmaml_hep.simulators import FastPQCModel  # noqa: E402

torch.set_num_threads(8)
RAW = ["pxj1", "pyj1", "pzj1", "mj1", "tau1j1", "tau2j1", "tau3j1", "pxj2", "pyj2", "pzj2", "mj2", "tau1j2", "tau2j2", "tau3j2"]
BAD = {(50, 750)}
NQ, DEPTH = 6, 2
STEPS, TASKS_PER_STEP, EV_PER_TASK, LR = 3000, 8, 128, 3e-3
SEEDS = (0, 1)
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    STEPS = 5


def load_qcd(data):
    cache = data / "qcd_big_sr_feats.npy"
    if cache.exists():
        return np.load(cache)
    out = []
    with tables.open_file(str(data / "Pythia_QCD_Dijet_Events.h5"), "r") as h:
        node = h.root.particles
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in node.block0_items.read()]
        n = node.block0_values.shape[0]
        for s in range(0, n, 1_000_000):
            b0, lab = node.block0_values[s : s + 1_000_000], node.block2_values[s : s + 1_000_000][:, 0]
            df = pd.DataFrame({nm: b0[:, j].astype(np.float64) for j, nm in enumerate(names) if nm in RAW})
            out.append(in_sr(features(df[lab == 0]))[FEATS].to_numpy())
    q = np.concatenate(out)
    np.save(cache, q)
    return q


class Model(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        self.proj = nn.Linear(8 if kind == "param" else 6, NQ)
        self.pqc = FastPQCModel(NQ, DEPTH, init_type="random")
        self.pqc.weights.data *= 0.1
        if kind == "qmaml":
            self.learner = Learner(2, (DEPTH, NQ, 3), 256, n_hidden=2).double()
            self.pqc.weights.requires_grad = False

    def theta(self, m):
        if self.kind == "qmaml":
            return self.learner(torch.tensor(np.asarray(m) / 600.0, dtype=torch.float64))
        return self.pqc.weights

    def forward(self, x, m, theta=None):
        if self.kind == "param":
            x = torch.cat([x, torch.tensor(np.asarray(m) / 600.0, dtype=torch.float32).expand(len(x), 2)], 1)
        th = self.theta(m) if theta is None else theta
        return self.pqc(self.proj(x), weights_override=th)


def main():
    data, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.time()
    rng = np.random.default_rng(0)
    q = load_qcd(data)
    q = q[rng.permutation(len(q))]
    q_test, q_train = q[:100_000], q[100_000:]
    mu, sd = q_train[:500_000].mean(0), q_train[:500_000].std(0)
    norm = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)  # noqa: E731
    print(f"QCD SR {len(q)} ({time.time()-t0:.0f}s)", flush=True)

    grid = in_sr(features(pd.read_hdf(data / "events_anomalydetection_Z_XY_qq_parametric.h5")))
    masses = sorted({(int(a), int(b)) for a, b in grid[["mx", "my"]].drop_duplicates().to_numpy()} - BAD)
    sig = {}
    for m in masses:
        g = grid[(grid.mx == m[0]) & (grid.my == m[1])][FEATS].to_numpy()
        g = g[rng.permutation(len(g))]
        nte = min(10_000, len(g) // 3)
        sig[m] = (g[: len(g) - nte][:20_000], g[len(g) - nte :])
    del grid
    vals = list(range(50, 601, 100))
    train_m = [m for m in masses if m[0] in vals and m[1] in vals]
    test_m = [m for m in masses if m not in set(train_m)]
    if SMOKE:
        test_m, train_m = test_m[:20], train_m[:4]
    res = {"train": train_m, "test": test_m, "config": dict(steps=STEPS, tasks_per_step=TASKS_PER_STEP, ev_per_task=EV_PER_TASK, lr=LR, nq=NQ, depth=DEPTH)}
    qt_n = norm(q_test)

    @torch.no_grad()
    def evaluate(model, m, theta=None):
        xs = norm(sig[m][1])
        s = []
        for x in (xs, qt_n):
            for i in range(0, len(x), 50_000):
                o = model(x[i : i + 50_000], m, theta)
                s.append((o[:, 1] - o[:, 0]).numpy())
        y = np.r_[np.ones(len(xs)), np.zeros(len(qt_n))]
        return sic_auc(y, np.concatenate(s))

    def batch(m, n, r):
        xs = sig[m][0][r.choice(len(sig[m][0]), n, replace=len(sig[m][0]) < n)]
        xb = q_train[r.choice(len(q_train), n, replace=False)]
        return norm(np.vstack([xs, xb])), torch.cat([torch.ones(n), torch.zeros(n)]).long()

    def pretrain(model, tasks, seed, steps=STEPS):
        r = np.random.default_rng(seed + 7)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], LR)
        for st in range(steps):
            loss = 0.0
            for i in r.choice(len(tasks), min(TASKS_PER_STEP, len(tasks)), replace=False):
                x, y = batch(tasks[i], EV_PER_TASK, r)
                loss = loss + nn.functional.cross_entropy(model(x, tasks[i]), y)
            opt.zero_grad()
            (loss / TASKS_PER_STEP).backward()
            opt.step()
        return model

    # 1. multi-task arms
    models = {}
    for kind in ("joint", "param", "qmaml"):
        for seed in SEEDS:
            models[(kind, seed)] = pretrain(Model(kind, seed), train_m, seed)
            print(f"[pretrain] {kind} seed {seed} {time.time()-t0:.0f}s", flush=True)

    # 2. single-task classifiers on every training point (for nearest transfer) and on a test subset (pertask)
    def single(m, seed, steps=5 if SMOKE else 600):
        mod = Model("joint", seed)
        return pretrain(mod, [m], seed, steps)

    single_train = {(m, s): single(m, s) for m in train_m for s in SEEDS}
    print(f"[single] train points {time.time()-t0:.0f}s", flush=True)
    sub = [test_m[i] for i in np.random.default_rng(1).choice(len(test_m), 16, replace=False)]
    single_test = {(m, s): single(m, s) for m in sub for s in SEEDS}
    print(f"[single] test subset {time.time()-t0:.0f}s", flush=True)

    # 3. zero-shot evaluation on held-out points (and on training points for qmaml / param / joint)
    zs = []
    for m in test_m + train_m:
        src = min(train_m, key=lambda s: (s[0] - m[0]) ** 2 + (s[1] - m[1]) ** 2)
        rec = {"tgt": list(m), "held_out": m in set(test_m), "nearest_src": list(src)}
        for s in SEEDS:
            for kind in ("joint", "param", "qmaml"):
                rec.setdefault(kind, []).append(evaluate(models[(kind, s)], m))
            rec.setdefault("nearest", []).append(evaluate(single_train[(src, s)], m))
            if (m, s) in single_test:
                rec.setdefault("pertask", []).append(evaluate(single_test[(m, s)], m))
            if m in set(train_m):
                rec.setdefault("pertask", []).append(evaluate(single_train[(m, s)], m))
        zs.append(rec)
    res["zeroshot"] = zs
    out_path.write_text(json.dumps(res, indent=1))
    print(f"[zeroshot] {time.time()-t0:.0f}s", flush=True)

    # 4. few-shot adaptation of theta only (Algorithm 2), K signal + 10K background, 3 draws
    few = []
    fs_points = [test_m[i] for i in np.random.default_rng(2).choice(len(test_m), 16, replace=False)]
    for K in (16, 64):
        for m in fs_points:
            src = min(train_m, key=lambda s: (s[0] - m[0]) ** 2 + (s[1] - m[1]) ** 2)
            rec = {"K": K, "tgt": list(m)}
            for s in SEEDS:
                for d in range(2):
                    r = np.random.default_rng(10_000 * K + 100 * s + d)
                    xs = sig[m][0][r.choice(len(sig[m][0]), K, replace=False)]
                    xb = q_train[r.choice(len(q_train), 10 * K, replace=False)]
                    x = norm(np.vstack([xs, xb]))
                    y = torch.cat([torch.ones(K), torch.zeros(10 * K)]).long()
                    for name, mod in (("qmaml", models[("qmaml", s)]), ("joint", models[("joint", s)]), ("param", models[("param", s)]), ("nearest", single_train[(src, s)])):
                        with torch.no_grad():
                            th0 = mod.theta(m).detach().clone()
                        th = nn.Parameter(th0)
                        opt = torch.optim.Adam([th], 0.01)
                        for _ in range(50):
                            loss = nn.functional.cross_entropy(mod(x, m, th), y, weight=torch.tensor([1.0, 10.0]))
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                        rec.setdefault(name, []).append(evaluate(mod, m, th.detach()))
            few.append(rec)
        res["fewshot"] = few
        out_path.write_text(json.dumps(res, indent=1))
        print(f"[fewshot] K={K} {time.time()-t0:.0f}s", flush=True)
    print(f"saved {out_path} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
