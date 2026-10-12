"""
Round-2 classification protocol (HIGGS and quark-gluon), addressing the review's points 1(c), 1(d), 3 and 6.

Changes from the round-1 protocol:
  * Three disjoint row splits: train (60%) / validation (20%) / test (20%; the same rows as round 1's test set).
    Meta-training tasks come only from train rows, hyperparameters are selected only on validation tasks,
    accuracy is reported only on test tasks. Tasks are re-sampled per seed.
  * 60 test tasks (6 bins x 10) = 480 query events per seed (round 1: 12 tasks = 96 events).
  * Every trainable arm uses the same optimizer (Adam), the same trainable modules (PQC angles or Learner,
    angle projection at 0.2x lr, linear head), the same loss and the same per-task adaptation at test time;
    for each arm the outer learning rate {1e-3, 5e-3, 2e-2} and the epoch count (1..30) are selected on
    validation tasks with two tuning seeds that are not among the ten reporting seeds.
  * Arms:
      qmaml             Learner(mean support embedding) -> theta0, Algorithm 1 (no inner loop), paper Learner
      joint_{zero,uniform,gaussian}
                        same pipeline without a Learner: one shared theta trained jointly (supervised
                        pretraining on the same tasks) -- isolates what the Learner adds
      fomaml_{5,20}     first-order MAML: K inner Adam steps, query loss at theta0 + sg(theta_K - theta0)
      reptile_legacy    round 1's classical arm: interpolation step theta += lr * (theta_K - theta)
      no_pqc            classical hybrid with the circuit removed (angle projection -> head)
      logreg/mlp_pool   scikit-learn on the pooled meta-training events (the data the meta-learners see)
      logreg_fewshot    logistic regression on each test task's 8 support events only
      logreg/mlp_full   scikit-learn on 50k training rows (large-data reference, not few-shot)
  * The PQC is simulated with qmaml_hep.simulators (exact; verified against PennyLane to 1e-15).

Usage:
    python -m qmaml_hep.classification.protocol --dataset higgs --stage tune
    python -m qmaml_hep.classification.protocol --dataset higgs --stage report
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from qmaml_hep.classification.extractor import TabularFeatureExtractor
from qmaml_hep.classification.tasks import make_tasks
from qmaml_hep.learner import Learner
from qmaml_hep.paths import DATA, RESULTS
from qmaml_hep.simulators import FastPQCModel

RES = os.path.join(RESULTS, "classification_v2")
DEPTH, EMB, W0_SCALE = 2, 512, 0.01
ADAPT_STEPS, ADAPT_LR = 5, 0.02
LRS = [1e-3, 5e-3, 2e-2]
MAX_EPOCHS = 30
TUNE_SEEDS = [100, 101]
REPORT_SEEDS = list(range(10))
ARMS = ["qmaml", "joint_zero", "joint_uniform", "joint_gaussian", "fomaml_5", "fomaml_20", "reptile_legacy", "no_pqc"]


# ---------------------------------------------------------------------------------------------------------
def load(dataset):
    if dataset == "higgs":
        from qmaml_hep.classification.higgs import ALL_FEATURE_COLS, load_higgs

        df = load_higgs(180_000, os.path.join(DATA, "higgs", "higgs_subsample.csv"))
        return df, list(ALL_FEATURE_COLS), "m_bb", [4, 6, 8]
    from qmaml_hep.classification.quarkgluon import FEATURE_COLS, load_quarkgluon

    df = load_quarkgluon(
        60_000, os.path.join(DATA, "quarkgluon", "QG_jets_0.npz"), os.path.join(DATA, "quarkgluon", "qg_features.csv")
    )
    return df, list(FEATURE_COLS), "jet_pt", [6]


def splits(df):
    n = len(df)
    test = df.iloc[: int(0.2 * n)].reset_index(drop=True)  # identical to round 1's test rows
    val = df.iloc[int(0.2 * n) : int(0.4 * n)].reset_index(drop=True)
    train = df.iloc[int(0.4 * n) :].reset_index(drop=True)
    return train, val, test


def tasks_for(train, val, test, cols, bin_col, seed):
    kw = dict(bin_count=6, support_size=8, query_size=8)
    return (
        make_tasks(train, cols, bin_col, tasks_per_bin=3, seed=1000 + seed, **kw),
        make_tasks(val, cols, bin_col, tasks_per_bin=3, seed=2000 + seed, **kw),
        make_tasks(test, cols, bin_col, tasks_per_bin=10, seed=3000 + seed, **kw),
    )


# ---------------------------------------------------------------------------------------------------------
class Arm(nn.Module):
    """Extractor (frozen random backbone, trainable angle projection) + PQC + head, plus the arm's
    source of initial angles (Learner or a shared parameter)."""

    def __init__(self, kind, d_in, nq, norm, seed):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.kind, self.nq = kind, nq
        self.ext = TabularFeatureExtractor(d_in, EMB, nq)
        self.ext.set_norm_stats(*norm)
        for p in self.ext.backbone.parameters():
            p.requires_grad = False
        init = kind.split("_")[1] if kind.startswith("joint_") else "zero"
        self.pqc = FastPQCModel(nq, DEPTH, init_type=init, bound_angles=True)
        if kind == "qmaml":
            self.learner = Learner(EMB, (DEPTH, nq, 3), 256, n_hidden=2).double()
            self.pqc.weights.requires_grad = False
        if kind == "no_pqc":
            self.pqc.weights.requires_grad = False

    def logits(self, x, theta):
        a = self.ext(x)
        if self.kind == "no_pqc":
            return self.pqc.fc(torch.tanh(a))
        return self.pqc(a, weights_override=theta)

    def theta0(self, support_x):
        if self.kind == "qmaml":
            emb = self.ext.embed(support_x).mean(0).double()
            return W0_SCALE * self.learner(emb)
        return self.pqc.weights

    def param_groups(self, lr):
        main = list(self.pqc.fc.parameters())
        if self.kind == "qmaml":
            main += list(self.learner.parameters())
        elif self.kind != "no_pqc":
            main += [self.pqc.weights]
        return [{"params": main, "lr": lr}, {"params": list(self.ext.to_angles.parameters()), "lr": 0.2 * lr}]


def adapt_eval(arm, task, steps=ADAPT_STEPS, lr=ADAPT_LR):
    """Test-time adaptation (Algorithm 2): Adam on the circuit angles only (head for no_pqc), support set only."""
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        th0 = arm.theta0(task["support_X"]).detach().clone()
    if arm.kind == "no_pqc":
        head = copy.deepcopy(arm.pqc.fc)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        a_s = torch.tanh(arm.ext(task["support_X"])).detach()
        for _ in range(steps):
            opt.zero_grad()
            ce(head(a_s), task["support_y"]).backward()
            opt.step()
        with torch.no_grad():
            out = head(torch.tanh(arm.ext(task["query_X"])))
    else:
        th = nn.Parameter(th0)
        opt = torch.optim.Adam([th], lr=lr)
        with torch.no_grad():
            a_s = arm.ext(task["support_X"])
        for _ in range(steps):
            opt.zero_grad()
            ce(arm.pqc(a_s, weights_override=th), task["support_y"]).backward()
            opt.step()
        with torch.no_grad():
            out = arm.logits(task["query_X"], th)
    p = torch.softmax(out, 1)[:, 1]
    acc = float((out.argmax(1) == task["query_y"]).float().mean())
    ll = float(ce(out, task["query_y"]))
    return acc, ll, p.numpy(), task["query_y"].numpy()


def evaluate(arm, tasks, steps=ADAPT_STEPS):
    r = [adapt_eval(arm, t, steps) for t in tasks]
    return float(np.mean([a for a, *_ in r])), float(np.mean([l for _, l, *_ in r])), r


def train_arm(kind, d_in, nq, norm, tr_tasks, val_tasks, lr, epochs, seed, track_val=True):
    arm = Arm(kind, d_in, nq, norm, seed)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(arm.param_groups(lr), lr=lr)
    hist = {"meta_loss": [], "val_acc": [], "val_ll": [], "grad_norm": []}
    trainable = [p for g in arm.param_groups(lr) for p in g["params"]]
    for _ in range(epochs):
        tot, gns = 0.0, []
        for task in tr_tasks:
            opt.zero_grad()
            if kind.startswith("fomaml_") or kind == "reptile_legacy":
                K = 5 if kind == "reptile_legacy" else int(kind.split("_")[1])
                th0 = arm.pqc.weights
                thk = nn.Parameter(th0.detach().clone())
                io = torch.optim.Adam([thk], lr=ADAPT_LR)
                with torch.no_grad():
                    a_s = arm.ext(task["support_X"])
                for _ in range(K):
                    io.zero_grad()
                    ce(arm.pqc(a_s, weights_override=thk), task["support_y"]).backward()
                    io.step()
                if kind == "reptile_legacy":
                    # round-1 baseline: no gradient step at all, only an interpolation toward theta_K
                    with torch.no_grad():
                        th0 += lr * (thk.detach() - th0)
                        loss = ce(arm.logits(task["query_X"], th0), task["query_y"])
                    tot += loss.item()
                    gns.append(0.0)
                    continue
                th_q = th0 + (thk.detach() - th0.detach())
                loss = ce(arm.logits(task["query_X"], th_q), task["query_y"])
            else:
                loss = ce(arm.logits(task["support_X"], arm.theta0(task["support_X"])), task["support_y"])
            loss.backward()
            gns.append(float(sum(p.grad.pow(2).sum() for p in trainable if p.grad is not None) ** 0.5))
            for g in arm.param_groups(lr):
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            opt.step()
            tot += loss.item()
        hist["meta_loss"].append(tot / len(tr_tasks))
        hist["grad_norm"].append(float(np.mean(gns)))
        if track_val:
            va, vl, _ = evaluate(arm, val_tasks)
            hist["val_acc"].append(va)
            hist["val_ll"].append(vl)
    return arm, hist


def sklearn_arms(tr_tasks, test_tasks, train_rows, cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xp = torch.cat([torch.cat([t["support_X"], t["query_X"]]) for t in tr_tasks]).numpy()
    yp = torch.cat([torch.cat([t["support_y"], t["query_y"]]) for t in tr_tasks]).numpy()
    sub = train_rows.sample(n=min(50_000, len(train_rows)), random_state=0)
    Xf, yf = sub[cols].values.astype(np.float32), sub["label"].values
    models = {
        "logreg_pool": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xp, yp),
        "mlp_pool": make_pipeline(
            StandardScaler(), MLPClassifier((64,), alpha=1e-2, max_iter=2000, random_state=0)
        ).fit(Xp, yp),
        "logreg_full": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xf, yf),
        "mlp_full": make_pipeline(
            StandardScaler(), MLPClassifier((64, 64), max_iter=200, early_stopping=True, random_state=0)
        ).fit(Xf, yf),
    }
    out = {
        k: float(np.mean([(m.predict(t["query_X"].numpy()) == t["query_y"].numpy()).mean() for t in test_tasks]))
        for k, m in models.items()
    }
    accs = []
    for t in test_tasks:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
            t["support_X"].numpy(), t["support_y"].numpy()
        )
        accs.append((m.predict(t["query_X"].numpy()) == t["query_y"].numpy()).mean())
    out["logreg_fewshot"] = float(np.mean(accs))
    return out


# ---------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["higgs", "qg"], required=True)
    ap.add_argument("--stage", choices=["tune", "report"], required=True)
    ap.add_argument("--qubits", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(RES, exist_ok=True)
    df, cols, bin_col, nqs = load(a.dataset)
    if a.qubits:
        nqs = [int(q) for q in a.qubits.split(",")]
    tag = f"{a.dataset}_q{''.join(map(str, nqs))}"
    train, val, test = splits(df)
    Xtr = torch.from_numpy(train[cols].values.astype(np.float32))
    norm = (Xtr.mean(0), Xtr.std(0))

    if a.stage == "tune":
        path = os.path.join(RES, f"{tag}_tune.json")
        out = json.load(open(path)) if os.path.exists(path) else {}
        for nq in nqs:
            for kind in ARMS:
                for lr in LRS:
                    key = f"{nq}|{kind}|{lr}"
                    if key in out:
                        continue
                    curves = []
                    for s in TUNE_SEEDS:
                        tr_t, va_t, _ = tasks_for(train, val, test, cols, bin_col, s)
                        t0 = time.time()
                        _, h = train_arm(kind, len(cols), nq, norm, tr_t, va_t, lr, MAX_EPOCHS, s)
                        curves.append(h["val_acc"])
                        print(
                            f"[tune {a.dataset}] q={nq} {kind} lr={lr} seed={s} best val={max(h['val_acc']):.3f} "
                            f"({time.time() - t0:.0f}s)",
                            flush=True,
                        )
                    out[key] = curves
                    json.dump(out, open(path, "w"))
        return

    tune = json.load(open(os.path.join(RES, f"{tag}_tune.json")))
    path = os.path.join(RES, f"{tag}_report.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for nq in nqs:
        # selection: (lr, epochs) maximizing mean validation accuracy over the tuning seeds
        choice = {}
        for kind in ARMS:
            best = max(
                ((lr, ep) for lr in LRS for ep in range(MAX_EPOCHS)),
                key=lambda le: np.mean([c[le[1]] for c in tune[f"{nq}|{kind}|{le[0]}"]]),
            )
            choice[kind] = {"lr": best[0], "epochs": best[1] + 1}
        out.setdefault(str(nq), {"choice": choice, "seeds": {}})
        for s in REPORT_SEEDS:
            if str(s) in out[str(nq)]["seeds"]:
                continue
            tr_t, va_t, te_t = tasks_for(train, val, test, cols, bin_col, s)
            row = {}
            for kind in ARMS:
                c = choice[kind]
                arm, h = train_arm(kind, len(cols), nq, norm, tr_t, va_t, c["lr"], c["epochs"], s, track_val=False)
                acc, ll, _ = evaluate(arm, te_t)
                acc0, _, _ = evaluate(arm, te_t, steps=0)
                acc20, _, _ = evaluate(arm, te_t, steps=20)
                row[kind] = {
                    "acc": acc,
                    "logloss": ll,
                    "acc_noadapt": acc0,
                    "acc_adapt20": acc20,
                    "meta_loss": h["meta_loss"],
                    "grad_norm": h["grad_norm"],
                }
            row.update({k: {"acc": v} for k, v in sklearn_arms(tr_t, te_t, train, cols).items()})
            out[str(nq)]["seeds"][str(s)] = row
            json.dump(out, open(path, "w"))
            print(
                f"[report {a.dataset}] q={nq} seed={s}: " + " ".join(f"{k}={v['acc']:.3f}" for k, v in row.items()),
                flush=True,
            )


if __name__ == "__main__":
    main()
