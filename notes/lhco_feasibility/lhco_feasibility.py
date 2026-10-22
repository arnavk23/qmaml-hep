"""Feasibility check: is the LHCO 2020 (mX, mY) signal grid a usable task family for Q-MAML?

Data: QCD background features (Zenodo 3733787), original R&D signal at (500, 100) (same record) and the
parametric signal grid W' -> X(qq) Y(qq), 12 x 12 masses (Zenodo 11188685).

Checks
  0  normalisation: grid point (500,100) vs the original R&D signal, per feature (KS statistic)
  1  saturation: per-task BDT with ample data -> AUC and SIC on every grid point
  2  few-shot: per-task classifiers trained on K signal events
  3  nearest-mass transfer: oracle BDT of a neighbouring grid point applied to the target
  4  parameterised BDT (features + mX, mY) trained on a coarse grid, evaluated on held-out points

Run: python -I lhco_feasibility.py <data_dir> <out_dir>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)
SR = (3.3e3, 3.7e3)  # signal region in mjj (GeV), the CATHODE/LHCO convention
FEATS = ["mj_lo", "dm", "tau21_lo", "tau21_hi", "tau32_lo", "tau32_hi"]
N_SIG_TRAIN, N_SIG_TEST = 20_000, 10_000
EPS_S = 0.3


def features(d: pd.DataFrame) -> pd.DataFrame:
    out = {}
    e = []
    for j in ("1", "2"):
        p2 = d[f"pxj{j}"] ** 2 + d[f"pyj{j}"] ** 2 + d[f"pzj{j}"] ** 2
        e.append(np.sqrt(p2 + d[f"mj{j}"] ** 2))
    px, py, pz = (d[f"p{c}j1"] + d[f"p{c}j2"] for c in "xyz")
    out["mjj"] = np.sqrt(np.clip((e[0] + e[1]) ** 2 - px**2 - py**2 - pz**2, 0, None))
    m1, m2 = d["mj1"].to_numpy(), d["mj2"].to_numpy()
    lo_is_1 = m1 <= m2
    eps = 1e-9

    def pick(a1, a2, lo):
        return np.where(lo_is_1 == lo, a1, a2)

    t21_1, t21_2 = d["tau2j1"] / (d["tau1j1"] + eps), d["tau2j2"] / (d["tau1j2"] + eps)
    t32_1, t32_2 = d["tau3j1"] / (d["tau2j1"] + eps), d["tau3j2"] / (d["tau2j2"] + eps)
    out["mj_lo"] = np.minimum(m1, m2)
    out["dm"] = np.abs(m1 - m2)
    out["tau21_lo"] = pick(t21_1, t21_2, True)
    out["tau21_hi"] = pick(t21_1, t21_2, False)
    out["tau32_lo"] = pick(t32_1, t32_2, True)
    out["tau32_hi"] = pick(t32_1, t32_2, False)
    f = pd.DataFrame(out, index=d.index)
    for c in ("mx", "my"):
        if c in d:
            f[c] = d[c].to_numpy()
    return f


def in_sr(f):
    return f[(f.mjj > SR[0]) & (f.mjj < SR[1])]


def sic_auc(y, s):
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    nb = int((y == 0).sum())
    eb = max(np.interp(EPS_S, tpr, fpr), 1.0 / nb)
    return float(auc), float(EPS_S / np.sqrt(eb)), float(1.0 / eb)


def bdt(seed=0):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)


def main():
    DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    qcd = in_sr(features(pd.read_hdf(DATA / "LHCORD_qcd_features.h5")))
    ref = in_sr(features(pd.read_hdf(DATA / "LHCORD_sig_features.h5")))
    grid_raw = pd.read_hdf(DATA / "events_anomalydetection_Z_XY_qq_parametric.h5")
    print("grid columns:", list(grid_raw.columns), grid_raw.shape, flush=True)
    grid = in_sr(features(grid_raw))
    del grid_raw
    masses = sorted({(int(a), int(b)) for a, b in grid[["mx", "my"]].drop_duplicates().to_numpy()})
    print(f"QCD in SR {len(qcd)}, ref signal in SR {len(ref)}, grid points {len(masses)}", flush=True)

    # QCD pools: train / test, fixed for every task
    qidx = RNG.permutation(len(qcd))
    q_train = qcd.iloc[qidx[: len(qidx) // 2]][FEATS].to_numpy()
    q_test = qcd.iloc[qidx[len(qidx) // 2 :]][FEATS].to_numpy()

    res = {"n_qcd_sr": len(qcd), "n_qcd_test": len(q_test), "eps_s": EPS_S, "sr": SR, "feats": FEATS}

    # 0. normalisation check at (500, 100)
    g0 = grid[(grid.mx == 500) & (grid.my == 100)]
    res["ks_500_100"] = {
        c: float(ks_2samp(g0[c], ref[c]).statistic) for c in FEATS + ["mjj"]
    }
    res["ks_reference_qcd_vs_ref"] = {c: float(ks_2samp(qcd[c].sample(20000, random_state=0), ref[c]).statistic) for c in FEATS}
    print("KS grid(500,100) vs original R&D signal:", res["ks_500_100"], flush=True)

    # split signal per task
    sig = {}
    for m in masses:
        g = grid[(grid.mx == m[0]) & (grid.my == m[1])][FEATS].to_numpy()
        g = g[RNG.permutation(len(g))]
        sig[m] = (g[:N_SIG_TRAIN], g[N_SIG_TRAIN : N_SIG_TRAIN + N_SIG_TEST], len(g))

    def eval_on(m, scorer):
        xt = np.vstack([sig[m][1], q_test])
        yt = np.r_[np.ones(len(sig[m][1])), np.zeros(len(q_test))]
        return sic_auc(yt, scorer(xt))

    # 1. oracle per task
    oracle_models, oracle = {}, {}
    nb_tr = len(q_train)
    for m in masses:
        xs = sig[m][0]
        xb = q_train[RNG.choice(nb_tr, size=min(nb_tr, 4 * len(xs)), replace=False)]
        clf = bdt().fit(np.vstack([xs, xb]), np.r_[np.ones(len(xs)), np.zeros(len(xb))])
        oracle_models[m] = clf
        oracle[m] = eval_on(m, lambda x, c=clf: c.predict_proba(x)[:, 1]) + (sig[m][2],)
    res["oracle"] = {f"{a},{b}": v for (a, b), v in oracle.items()}
    print(f"[1] oracle done {time.time()-t0:.0f}s", flush=True)

    # 2. few-shot per task (K signal events, 10K background events), 5 draws, on a subset of tasks
    few = {}
    sub = [m for m in masses if m[0] % 100 == 0 and m[1] % 100 == 0]  # 6x6 = 36 tasks
    for K in (8, 64, 512):
        for m in sub:
            vals_lr, vals_bdt = [], []
            for r in range(5):
                rr = np.random.default_rng(1000 * K + r)
                xs = sig[m][0][rr.choice(len(sig[m][0]), K, replace=False)]
                xb = q_train[rr.choice(nb_tr, 10 * K, replace=False)]
                X, y = np.vstack([xs, xb]), np.r_[np.ones(K), np.zeros(10 * K)]
                sc = StandardScaler().fit(X)
                lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
                vals_lr.append(eval_on(m, lambda x: lr.decision_function(sc.transform(x))))
                if K >= 64:
                    b = HistGradientBoostingClassifier(max_iter=100, random_state=r, class_weight="balanced").fit(X, y)
                    vals_bdt.append(eval_on(m, lambda x, b=b: b.predict_proba(x)[:, 1]))
            few[(K, m)] = {"lr": np.mean(vals_lr, 0).tolist(), "bdt": np.mean(vals_bdt, 0).tolist() if vals_bdt else None}
    res["fewshot"] = {f"{K}|{a},{b}": v for (K, (a, b)), v in few.items()}
    print(f"[2] few-shot done {time.time()-t0:.0f}s", flush=True)

    # 3. nearest-mass transfer: source offset by d along mX, mY or both
    trans = {}
    mset = set(masses)
    for d in (50, 100, 150, 200, 300):
        for m in masses:
            for off in ((d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, -d)):
                src = (m[0] + off[0], m[1] + off[1])
                if src in mset:
                    c = oracle_models[src]
                    trans.setdefault(d, []).append(
                        (m, src, eval_on(m, lambda x, c=c: c.predict_proba(x)[:, 1]), oracle[m][:3])
                    )
    res["transfer"] = {
        str(d): [{"tgt": list(t), "src": list(s), "got": g, "oracle": o} for t, s, g, o in v] for d, v in trans.items()
    }
    print(f"[3] transfer done {time.time()-t0:.0f}s", flush=True)

    # 4. parameterised BDT on coarse grids; background gets masses drawn from the training grid
    par = {}
    for step in (100, 150):
        vals = list(range(50, 601, step))
        train_m = [m for m in masses if m[0] in vals and m[1] in vals]
        test_m = [m for m in masses if m not in set(train_m)]
        xs = np.vstack([np.c_[sig[m][0][:5000], np.tile(m, (5000, 1))] for m in train_m])
        nb = len(xs)
        bm = np.array(train_m)[RNG.integers(len(train_m), size=nb)]
        xb = np.c_[q_train[RNG.choice(nb_tr, nb, replace=nb > nb_tr)], bm]
        clf = HistGradientBoostingClassifier(max_iter=400, random_state=0).fit(
            np.vstack([xs, xb]), np.r_[np.ones(nb), np.zeros(nb)]
        )
        out = []
        for m in test_m:
            got = eval_on(m, lambda x, m=m: clf.predict_proba(np.c_[x, np.tile(m, (len(x), 1))])[:, 1])
            # nearest training point by Euclidean distance in (mX, mY)
            src = min(train_m, key=lambda s: (s[0] - m[0]) ** 2 + (s[1] - m[1]) ** 2)
            c = oracle_models[src]
            nn = eval_on(m, lambda x, c=c: c.predict_proba(x)[:, 1])
            out.append({"tgt": list(m), "param": got, "nearest": nn, "nearest_src": list(src), "oracle": oracle[m][:3]})
        par[step] = {"n_train_tasks": len(train_m), "results": out}
        print(f"[4] parameterised step {step}: {len(train_m)} train tasks, {time.time()-t0:.0f}s", flush=True)
    res["param"] = {str(k): v for k, v in par.items()}

    (OUT / "lhco_feasibility.json").write_text(json.dumps(res, indent=1))
    print(f"saved {OUT/'lhco_feasibility.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
