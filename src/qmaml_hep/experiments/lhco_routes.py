"""Where should task information enter a variational classifier? Q-MAML and its alternatives on the LHCO
(m_X, m_Y) signal grid.

Every quantum arm uses the same classifier: features -> front end -> 6 encoding angles -> FastPQCModel (6 qubits,
depth 2, RY(pi tanh) encoding, StronglyEntanglingLayers, <Z>) -> linear head. The arms differ only in how the task
descriptor m = (m_X, m_Y) / 600 GeV enters:

    joint        nowhere (one classifier for every mass point)
    param        Linear(8 -> 6) on [x, m] (masses as circuit inputs)
    qmaml        theta = Learner(m), d -> 256 -> 256 -> P (Algorithm 1, no inner loop)
    hyper_enc    MLP(m) generates the Linear(6 -> 6) encoding (conditions the encoding, Learner-sized)
    param_deep   MLP([x, m]) 8 -> 256 -> 256 -> 6 angles (parameterised network with a circuit head)
    qmaml_deep   MLP(x) 6 -> 256 -> 256 -> 6 angles, theta = Learner(m)
    joint_deep   MLP(x) 6 -> 256 -> 256 -> 6 angles, no mass information
    nearest      a single-task classifier (joint architecture) trained on the nearest training mass
    classical    parameterised MLP [x, m] -> 128 -> 128 -> 128 -> 2, no circuit (Baldi et al. 2016)

Protocol
    * Training masses: a regular grid with spacing 100, 150 or 200 GeV starting at 50 GeV. Every other grid point is
      held out, and held-out points are split into validation (about 30%) and test points, keeping m_X <-> m_Y
      mirror pairs together (the tasks are nearly symmetric under the swap).
    * Events: per point, signal is split into train (up to 20k) / evaluation (up to 10k). QCD (2.7M in the signal
      region) is split once into train / validation (100k) / test (200k).
    * Same budget for every multi-task arm: STEPS Adam steps, each on TASKS_PER_STEP training masses with
      EV_PER_TASK signal + EV_PER_TASK QCD events; cross-entropy.
    * Tuning: learning rate in LRS per arm, on validation points, with seeds 100 and 101, at spacing 100. The chosen
      rate is used at every spacing. Reporting seeds 0-4.
    * Metric: background rejection R = 1 / eps_b at eps_s = 0.3 (AUC is saturated at 0.96-0.99); comparisons are
      paired per-point differences in log10 R, hierarchical bootstrap over seeds then points, Holm across the family.

Usage
    python -m qmaml_hep.experiments.lhco_routes run --stage tune --spacing 100 --arm qmaml --lr 3e-3 --seed 100
    python -m qmaml_hep.experiments.lhco_routes jobs --stage tune > jobs.txt     # one command per line
    python -m qmaml_hep.experiments.lhco_routes select                          # learning rates from tuning
    python -m qmaml_hep.experiments.lhco_routes score --spacing 100             # report-stage comparisons
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from qmaml_hep import analysis as A
from qmaml_hep.classification import lhco
from qmaml_hep.learner import Learner
from qmaml_hep.paths import RESULTS
from qmaml_hep.simulators import FastPQCModel

OUT = os.path.join(RESULTS, "lhco_routes")
NQ, DEPTH = 6, 2
STEPS, TASKS_PER_STEP, EV_PER_TASK = 3000, 8, 128
SINGLE_STEPS = 600
LRS = [1e-3, 3e-3, 1e-2]
# Arms whose best lr fell on the grid edge were also tuned one step beyond it (same seeds); the edge value still won.
EDGE_CHECKS = {"joint": 3e-2, "param": 3e-2, "qmaml_deep": 3e-4}
TUNE_SEEDS, REPORT_SEEDS = [100, 101], [0, 1, 2, 3, 4]
SPACINGS = [100, 150, 200]
ARMS = ["joint", "param", "qmaml", "hyper_enc", "param_deep", "qmaml_deep", "joint_deep", "nearest", "classical"]
EPS_S = 0.3
MSCALE = 600.0


# ------------------------------------------------------------------------------------------------- splits
def grid_split(masses, spacing):
    """-> (train, validation, test) mass lists. Validation/test assignment is per mirror pair {a, b}."""
    vals = set(range(50, 601, spacing))
    train = sorted(m for m in masses if m[0] in vals and m[1] in vals)
    rest = [m for m in masses if m not in set(train)]
    pairs = sorted({tuple(sorted(m)) for m in rest})
    rng = np.random.default_rng(spacing)
    val_pairs = {pairs[i] for i in rng.choice(len(pairs), int(round(0.3 * len(pairs))), replace=False)}
    val = sorted(m for m in rest if tuple(sorted(m)) in val_pairs)
    test = sorted(m for m in rest if tuple(sorted(m)) not in val_pairs)
    return train, val, test


class Data:
    def __init__(self):
        sig, qcd = lhco.load()
        rng = np.random.default_rng(0)
        qcd = qcd[rng.permutation(len(qcd))]
        self.q_test, self.q_val, self.q_train = qcd[:200_000], qcd[200_000:300_000], qcd[300_000:]
        self.sig = {}
        for m in sorted(sig):
            g = sig[m][rng.permutation(len(sig[m]))]
            n_eval = min(10_000, len(g) // 3)
            self.sig[m] = (g[: len(g) - n_eval][:20_000], g[len(g) - n_eval :])
        self.mu, self.sd = self.q_train[:500_000].mean(0), self.q_train[:500_000].std(0)
        self.masses = sorted(self.sig)

    def norm(self, a):
        return torch.tensor((a - self.mu) / self.sd, dtype=torch.float32)

    def batch(self, m, n, rng):
        xs = self.sig[m][0][rng.choice(len(self.sig[m][0]), n, replace=len(self.sig[m][0]) < n)]
        xb = self.q_train[rng.choice(len(self.q_train), n, replace=False)]
        return self.norm(np.vstack([xs, xb])), torch.cat([torch.ones(n), torch.zeros(n)]).long()


# ------------------------------------------------------------------------------------------------- models
def mlp(i, o, h=256, n=2):
    layers, w = [], i
    for _ in range(n):
        layers += [nn.Linear(w, h), nn.ReLU()]
        w = h
    return nn.Sequential(*layers, nn.Linear(w, o))


class Arm(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        if kind == "classical":
            self.net = mlp(8, 2, h=128, n=3)
            return
        if kind in ("joint", "qmaml", "nearest"):
            self.front = nn.Linear(6, NQ)
        elif kind == "param":
            self.front = nn.Linear(8, NQ)
        elif kind == "hyper_enc":
            self.gen = mlp(2, NQ * 6 + NQ)
            with torch.no_grad():  # start at a standard Linear(6, 6) initialisation
                self.gen[-1].weight.mul_(0.01)
                base = nn.Linear(6, NQ)
                self.gen[-1].bias.copy_(torch.cat([base.weight.flatten(), base.bias]))
        elif kind == "param_deep":
            self.front = mlp(8, NQ)
        elif kind in ("qmaml_deep", "joint_deep"):
            self.front = mlp(6, NQ)
        else:
            raise ValueError(kind)
        self.pqc = FastPQCModel(NQ, DEPTH, init_type="random")
        self.pqc.weights.data *= 0.1
        if kind in ("qmaml", "qmaml_deep"):
            self.learner = Learner(2, (DEPTH, NQ, 3), 256, n_hidden=2).double()
            self.pqc.weights.requires_grad = False

    def theta(self, m):
        if self.kind in ("qmaml", "qmaml_deep"):
            return self.learner(torch.tensor(np.asarray(m) / MSCALE, dtype=torch.float64))
        return self.pqc.weights

    def forward(self, x, m, theta=None):
        mm = torch.tensor(np.asarray(m) / MSCALE, dtype=torch.float32).expand(len(x), 2)
        if self.kind == "classical":
            return self.net(torch.cat([x, mm], 1))
        if self.kind == "hyper_enc":
            g = self.gen(mm[0])
            a = x @ g[: NQ * 6].view(NQ, 6).T + g[NQ * 6 :]
        elif self.kind in ("param", "param_deep"):
            a = self.front(torch.cat([x, mm], 1))
        else:
            a = self.front(x)
        return self.pqc(a, weights_override=self.theta(m) if theta is None else theta)


def train(model, data, tasks, seed, lr, steps):
    rng = np.random.default_rng(seed + 7)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr)
    k = min(TASKS_PER_STEP, len(tasks))
    for _ in range(steps):
        loss = 0.0
        for i in rng.choice(len(tasks), k, replace=False):
            x, y = data.batch(tasks[i], EV_PER_TASK, rng)
            loss = loss + nn.functional.cross_entropy(model(x, tasks[i]), y)
        opt.zero_grad()
        (loss / k).backward()
        opt.step()
    return model


def rejection(y, s):
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y, s)
    nb = int((y == 0).sum())
    eb = max(float(np.interp(EPS_S, tpr, fpr)), 1.0 / nb)
    return [float(roc_auc_score(y, s)), float(EPS_S / np.sqrt(eb)), float(1.0 / eb)]


@torch.no_grad()
def evaluate(model, data, m, qcd):
    xs = data.norm(data.sig[m][1])
    xb = data.norm(qcd)
    s = []
    for x in (xs, xb):
        for i in range(0, len(x), 50_000):
            o = model(x[i : i + 50_000], m)
            s.append((o[:, 1] - o[:, 0]).numpy())
    return rejection(np.r_[np.ones(len(xs)), np.zeros(len(xb))], np.concatenate(s))


def nearest_of(m, train_m):
    return min(train_m, key=lambda s: ((s[0] - m[0]) ** 2 + (s[1] - m[1]) ** 2, s))


# ------------------------------------------------------------------------------------------------- one job
def run(stage, spacing, arm, lr, seed, threads=2):
    torch.set_num_threads(threads)
    t0 = time.time()
    data = Data()
    tr, va, te = grid_split(data.masses, spacing)
    targets, qcd = (va, data.q_val) if stage == "tune" else (te + tr, data.q_test)
    res = dict(
        stage=stage,
        spacing=spacing,
        arm=arm,
        lr=lr,
        seed=seed,
        train=tr,
        val=va,
        test=te,
        config=dict(
            steps=STEPS,
            tasks_per_step=TASKS_PER_STEP,
            ev_per_task=EV_PER_TASK,
            single_steps=SINGLE_STEPS,
            nq=NQ,
            depth=DEPTH,
            eps_s=EPS_S,
        ),
    )
    if arm == "nearest":
        need = sorted({nearest_of(m, tr) for m in targets})
        models = {s: train(Arm("nearest", seed), data, [s], seed, lr, SINGLE_STEPS) for s in need}
        res["points"] = {f"{m[0]},{m[1]}": evaluate(models[nearest_of(m, tr)], data, m, qcd) for m in targets}
    else:
        model = train(Arm(arm, seed), data, tr, seed, lr, STEPS)
        res["n_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        res["points"] = {f"{m[0]},{m[1]}": evaluate(model, data, m, qcd) for m in targets}
    res["seconds"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{stage}_sp{spacing}_{arm}_lr{lr:g}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(res, f)
    print(f"[lhco_routes] {path} {res['seconds']:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------- driver
def selected_lrs():
    path = os.path.join(OUT, "selected_lr.json")
    return json.load(open(path)) if os.path.isfile(path) else None


def jobs(stage):
    py = f"{sys.executable} -m qmaml_hep.experiments.lhco_routes run"
    if stage == "tune":
        for arm in ARMS:
            for lr in LRS:
                for s in TUNE_SEEDS:
                    print(f"{py} --stage tune --spacing 100 --arm {arm} --lr {lr:g} --seed {s}")
    else:
        lrs = selected_lrs()
        if lrs is None:
            raise SystemExit("run the tuning stage and `select` first")
        for sp in SPACINGS:
            for arm in ARMS:
                for s in REPORT_SEEDS:
                    print(f"{py} --stage report --spacing {sp} --arm {arm} --lr {lrs[arm]:g} --seed {s}")


def mean_log_r(rec):
    return float(np.mean([np.log10(v[2]) for v in rec["points"].values()]))


def select():
    out = {}
    for arm in ARMS:
        best = None
        for lr in LRS:
            files = [glob.glob(os.path.join(OUT, f"tune_sp100_{arm}_lr{lr:g}_s{s}.json")) for s in TUNE_SEEDS]
            if not all(files):
                continue
            score = np.mean([mean_log_r(json.load(open(f[0]))) for f in files])
            print(f"  {arm:11s} lr {lr:g}: mean log10 R on validation points {score:.3f}")
            if best is None or score > best[1]:
                best = (lr, score)
        if best:
            out[arm] = best[0]
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "selected_lr.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def load_report(spacing):
    lrs = selected_lrs()
    runs = {}
    for arm in ARMS:
        fs = sorted(glob.glob(os.path.join(OUT, f"report_sp{spacing}_{arm}_lr{lrs[arm]:g}_s*.json")))
        runs[arm] = {json.load(open(f))["seed"]: json.load(open(f)) for f in fs}
    return runs


def score(spacing, focus="qmaml", rng_seed=0):
    runs = load_report(spacing)
    any_run = next(iter(runs[focus].values()))
    test = [tuple(m) for m in any_run["test"]]
    seeds = sorted(set.intersection(*[set(r) for r in runs.values() if r]))
    rng = np.random.default_rng(rng_seed)
    rows, ps = [], []
    for opp in [a for a in ARMS if a != focus and runs[a]]:
        D = []
        for s in seeds:
            a, b = runs[focus][s]["points"], runs[opp][s]["points"]
            D.append(np.array([np.log10(a[f"{m[0]},{m[1]}"][2]) - np.log10(b[f"{m[0]},{m[1]}"][2]) for m in test]))
        mean, lo, hi, p = A.hboot(D, rng)
        wins = float(np.mean(np.concatenate(D) > 0))
        rows.append(dict(opponent=opp, mean=mean, lo=lo, hi=hi, p=p, frac_focus_better=wins))
        ps.append(p)
    for r, padj in zip(rows, A.holm(np.array(ps))):
        r["p_holm"] = float(padj)
        r["verdict"] = (
            "win" if (padj < 0.05 and r["mean"] > 0) else "loss" if (padj < 0.05 and r["mean"] < 0) else "tie"
        )
    med = {
        a: float(
            np.median([np.mean([np.log10(runs[a][s]["points"][f"{m[0]},{m[1]}"][2]) for s in seeds]) for m in test])
        )
        for a in ARMS
        if runs[a]
    }
    gap = {}
    opath = os.path.join(OUT, "oracle_bdt.json")
    if os.path.isfile(opath):  # reference, not part of the Holm family
        orc = json.load(open(opath))
        for a in [a for a in ARMS if runs[a]]:
            D = [
                np.array(
                    [
                        np.log10(runs[a][s]["points"][k][2]) - np.log10(orc[k][2])
                        for k in (f"{m[0]},{m[1]}" for m in test)
                    ]
                )
                for s in seeds
            ]
            mean, lo, hi, _ = A.hboot(D, rng)
            gap[a] = dict(mean=mean, lo=lo, hi=hi)
    summary = dict(
        spacing=spacing,
        focus=focus,
        seeds=seeds,
        n_test_points=len(test),
        median_log10R=med,
        comparisons=rows,
        gap_to_oracle_bdt=gap,
    )
    if gap:
        print("  gap to per-point BDT: " + "  ".join(f"{a} {g['mean']:+.2f}" for a, g in gap.items()))
    print(f"spacing {spacing}: {len(test)} test points, seeds {seeds}")
    print("  median log10 R: " + "  ".join(f"{a} {v:.2f}" for a, v in med.items()))
    for r in rows:
        print(
            f"  {focus} - {r['opponent']:11s}: {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  p_holm {r['p_holm']:.3g}  {r['verdict']:4s}  {focus} better on {r['frac_focus_better']:.2f}"
        )
    with open(os.path.join(OUT, f"summary_sp{spacing}_{focus}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def oracle():
    """Reference: a BDT trained on each grid point's own training signal (and 4x as many QCD training events),
    evaluated on the same QCD test events as the report stage. One seed; it bounds what any arm can reach."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    data = Data()
    rng = np.random.default_rng(0)
    out = {}
    for m in data.masses:
        xs = data.sig[m][0]
        xb = data.q_train[rng.choice(len(data.q_train), 4 * len(xs), replace=False)]
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(np.vstack([xs, xb]), np.r_[np.ones(len(xs)), np.zeros(len(xb))])
        xt = np.vstack([data.sig[m][1], data.q_test])
        yt = np.r_[np.ones(len(data.sig[m][1])), np.zeros(len(data.q_test))]
        out[f"{m[0]},{m[1]}"] = rejection(yt, clf.predict_proba(xt)[:, 1])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "oracle_bdt.json"), "w") as f:
        json.dump(out, f)
    print(f"[lhco_routes] oracle BDT on {len(out)} points")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stage", choices=["tune", "report"], required=True)
    r.add_argument("--spacing", type=int, choices=SPACINGS, required=True)
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--lr", type=float, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--threads", type=int, default=2)
    j = sub.add_parser("jobs")
    j.add_argument("--stage", choices=["tune", "report"], required=True)
    sub.add_parser("select")
    sub.add_parser("oracle")
    s = sub.add_parser("score")
    s.add_argument("--spacing", type=int, choices=SPACINGS, required=True)
    s.add_argument("--focus", default="qmaml")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.stage, a.spacing, a.arm, a.lr, a.seed, a.threads)
    elif a.cmd == "jobs":
        jobs(a.stage)
    elif a.cmd == "select":
        select()
    elif a.cmd == "oracle":
        oracle()
    else:
        score(a.spacing, a.focus)


if __name__ == "__main__":
    main()
