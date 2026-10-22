"""Capacity-matched conditioning baselines for the Q-MAML pilot. Same data splits, circuit, budget and
evaluation as lhco_qmaml_pilot.py (the split code is copied verbatim so the test sets are identical).

    hyper_enc   masses -> MLP(2 -> 256 -> 256 -> 6*6 + 6) generates the encoding projection; theta shared
                (conditions the encoding, not theta, at the Learner's capacity)
    param_deep  MLP(features + masses: 8 -> 256 -> 256 -> 6) gives the encoding angles; theta shared
                (a classical parameterised network with a circuit head)
    qmaml_deep  MLP(features: 6 -> 256 -> 256 -> 6) front end, theta = Learner(masses)  (Q-MAML with the same
                deep feature network as param_deep, isolating where the mass information enters)
    joint_deep  the same front end, shared theta, no mass information (1 seed)

Run: python lhco_pilot_matched.py <data_dir> <out_json> [kinds, comma-separated]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "/home/arnav/qmaml-hep/src")
sys.path.insert(0, str(Path(__file__).parent))
from lhco_feasibility import FEATS, features, in_sr, sic_auc  # noqa: E402
from lhco_qmaml_pilot import BAD, DEPTH, EV_PER_TASK, LR, NQ, SEEDS, STEPS, TASKS_PER_STEP, load_qcd  # noqa: E402
from qmaml_hep.learner import Learner  # noqa: E402
from qmaml_hep.simulators import FastPQCModel  # noqa: E402

torch.set_num_threads(8)
KINDS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["hyper_enc", "param_deep"]


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, o))


class Matched(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        if kind == "hyper_enc":
            self.gen = mlp(2, NQ * 6 + NQ)
            with torch.no_grad():  # start near a standard Linear(6, 6) init
                self.gen[-1].weight.mul_(0.01)
                base = nn.Linear(6, NQ)
                self.gen[-1].bias.copy_(torch.cat([base.weight.flatten(), base.bias]))
        elif kind == "param_deep":
            self.front = mlp(8, NQ)
        else:  # qmaml_deep / joint_deep: the same deep front end on the features only
            self.front = mlp(6, NQ)
        self.pqc = FastPQCModel(NQ, DEPTH, init_type="random")
        self.pqc.weights.data *= 0.1
        if kind == "qmaml_deep":
            self.learner = Learner(2, (DEPTH, NQ, 3), 256, n_hidden=2).double()
            self.pqc.weights.requires_grad = False

    def theta(self, m):
        if self.kind == "qmaml_deep":
            return self.learner(torch.tensor(np.asarray(m) / 600.0, dtype=torch.float64))
        return self.pqc.weights

    def forward(self, x, m, theta=None):
        mm = torch.tensor(np.asarray(m) / 600.0, dtype=torch.float32)
        if self.kind == "hyper_enc":
            g = self.gen(mm)
            a = x @ g[: NQ * 6].view(NQ, 6).T + g[NQ * 6 :]
        elif self.kind == "param_deep":
            a = self.front(torch.cat([x, mm.expand(len(x), 2)], 1))
        else:
            a = self.front(x)
        return self.pqc(a, weights_override=self.theta(m) if theta is None else theta)


def main():
    data, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.time()
    rng = np.random.default_rng(0)
    q = load_qcd(data)
    q = q[rng.permutation(len(q))]
    q_test, q_train = q[:100_000], q[100_000:]
    mu, sd = q_train[:500_000].mean(0), q_train[:500_000].std(0)
    norm = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)  # noqa: E731
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
    qt_n = norm(q_test)
    print(f"data ready {time.time()-t0:.0f}s", flush=True)

    @torch.no_grad()
    def evaluate(model, m):
        xs = norm(sig[m][1])
        s = []
        for x in (xs, qt_n):
            for i in range(0, len(x), 50_000):
                o = model(x[i : i + 50_000], m)
                s.append((o[:, 1] - o[:, 0]).numpy())
        return sic_auc(np.r_[np.ones(len(xs)), np.zeros(len(qt_n))], np.concatenate(s))

    def batch(m, n, r):
        xs = sig[m][0][r.choice(len(sig[m][0]), n, replace=len(sig[m][0]) < n)]
        xb = q_train[r.choice(len(q_train), n, replace=False)]
        return norm(np.vstack([xs, xb])), torch.cat([torch.ones(n), torch.zeros(n)]).long()

    def pretrain(model, tasks, seed):
        r = np.random.default_rng(seed + 7)
        opt = torch.optim.Adam(model.parameters(), LR)
        for st in range(STEPS):
            loss = 0.0
            for i in r.choice(len(tasks), TASKS_PER_STEP, replace=False):
                x, y = batch(tasks[i], EV_PER_TASK, r)
                loss = loss + nn.functional.cross_entropy(model(x, tasks[i]), y)
            opt.zero_grad()
            (loss / TASKS_PER_STEP).backward()
            opt.step()
        return model

    res = {"zeroshot": []}
    models = {}
    plan = [(k, s) for k in KINDS for s in (SEEDS if k != "joint_deep" else SEEDS[:1])]
    for kind, s in plan:
        if True:
            models[(kind, s)] = pretrain(Matched(kind, s), train_m, s)
            n = sum(p.numel() for p in models[(kind, s)].parameters())
            res[f"{kind}_params"] = n
            print(f"[pretrain] {kind} seed {s} ({n} params) {time.time()-t0:.0f}s", flush=True)
    for m in test_m + train_m:
        rec = {"tgt": list(m), "held_out": m in set(test_m)}
        for kind in KINDS:
            rec[kind] = [evaluate(models[(k, s)], m) for k, s in plan if k == kind]
        res["zeroshot"].append(rec)
    out_path.write_text(json.dumps(res, indent=1))
    print(f"saved {out_path} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
