"""Second feasibility round on the LHCO (mX, mY) grid.

  A  oracle per-task BDT (reference), on the chosen QCD sample
  B  parameterised MLP (features + mX, mY) on coarse grids, vs parameterised BDT and nearest-mass transfer;
     evaluated on held-out grid points and on its own training points
  C  circuit capacity at 4 grid points: linear projection -> FastPQCModel (RY(tanh(a) pi) encoding,
     StronglyEntanglingLayers, <Z>, linear head), depth 2 and 4; its no-circuit twin; MLP; logistic regression

Run: python lhco_check2.py <data_dir> <out_json> <qcd: small|big> [--skip-param] [--skip-vqc]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tables
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/arnav/qmaml-hep/src")
sys.path.insert(0, str(Path(__file__).parent))
from lhco_feasibility import FEATS, SR, features, in_sr, sic_auc  # noqa: E402
from qmaml_hep.simulators import FastPQCModel  # noqa: E402

DATA, OUT, QCD = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
SKIP_PARAM, SKIP_VQC = "--skip-param" in sys.argv, "--skip-vqc" in sys.argv
RNG = np.random.default_rng(0)
torch.set_num_threads(8)
RAW = ["pxj1", "pyj1", "pzj1", "mj1", "tau1j1", "tau2j1", "tau3j1", "pxj2", "pyj2", "pzj2", "mj2", "tau1j2", "tau2j2", "tau3j2"]
BAD = {(50, 750)}
VQC_POINTS = [(100, 100), (300, 300), (150, 450), (500, 200)]


def load_big_qcd(path, chunk=1_000_000):
    """Fixed-format pandas frame: read row slices with PyTables, keep only the signal region."""
    out, labels = [], set()
    with tables.open_file(str(path), "r") as h:
        node = h.get_node("/" + [n._v_name for n in h.root][0])
        cols = []
        for b in range(int(node._v_attrs.nblocks)):
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in getattr(node, f"block{b}_items").read()]
            cols.append((getattr(node, f"block{b}_values"), names))
        n = cols[0][0].shape[0]
        for s in range(0, n, chunk):
            parts = {}
            for arr, names in cols:
                blk = arr[s : s + chunk]
                for j, nm in enumerate(names):
                    if nm in RAW or nm == "label":
                        parts[nm] = blk[:, j].astype(np.float64)
            df = pd.DataFrame(parts)
            labels.update(np.unique(df["label"]).tolist())
            out.append(in_sr(features(df[df["label"] == 0])))
    print("labels seen in big QCD file:", sorted(labels), flush=True)
    return pd.concat(out, ignore_index=True), n


def rej(y, s):
    return sic_auc(y, s)


def main():
    t0 = time.time()
    res = {"qcd": QCD}
    if QCD == "big":
        qcd, ntot = load_big_qcd(DATA / "Pythia_QCD_Dijet_Events.h5")
        res["n_qcd_total"] = int(ntot)
        small = in_sr(features(pd.read_hdf(DATA / "LHCORD_qcd_features.h5")))
        from scipy.stats import ks_2samp

        res["ks_big_vs_small_qcd"] = {c: float(ks_2samp(qcd[c].sample(100_000, random_state=0), small[c]).statistic) for c in FEATS + ["mjj"]}
        print("KS big vs small QCD (SR):", res["ks_big_vs_small_qcd"], flush=True)
    else:
        qcd = in_sr(features(pd.read_hdf(DATA / "LHCORD_qcd_features.h5")))
    q = qcd[FEATS].to_numpy()
    q = q[RNG.permutation(len(q))]
    n_test = min(len(q) // 2, 1_000_000)
    q_test, q_train = q[:n_test], q[n_test:]
    res["n_qcd_sr"], res["n_qcd_test"] = len(q), n_test
    print(f"QCD SR {len(q)}  test {n_test}  train {len(q_train)}  ({time.time()-t0:.0f}s)", flush=True)

    grid = in_sr(features(pd.read_hdf(DATA / "events_anomalydetection_Z_XY_qq_parametric.h5")))
    masses = sorted({(int(a), int(b)) for a, b in grid[["mx", "my"]].drop_duplicates().to_numpy()} - BAD)
    sig = {}
    for m in masses:
        g = grid[(grid.mx == m[0]) & (grid.my == m[1])][FEATS].to_numpy()
        g = g[RNG.permutation(len(g))]
        nte = min(10_000, len(g) // 3)
        sig[m] = (g[: len(g) - nte][:20_000], g[len(g) - nte :])
    del grid

    def ev(m, f):
        xt = np.vstack([sig[m][1], q_test])
        return rej(np.r_[np.ones(len(sig[m][1])), np.zeros(len(q_test))], f(xt))

    # A. oracle
    oracle, omod = {}, {}
    for m in masses:
        xs = sig[m][0]
        xb = q_train[RNG.choice(len(q_train), min(len(q_train), 4 * len(xs)), replace=False)]
        c = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(np.vstack([xs, xb]), np.r_[np.ones(len(xs)), np.zeros(len(xb))])
        omod[m], oracle[m] = c, ev(m, lambda x, c=c: c.predict_proba(x)[:, 1])
    res["oracle"] = {f"{a},{b}": v for (a, b), v in oracle.items()}
    Path(OUT).write_text(json.dumps(res, indent=1))
    print(f"[A] oracle {time.time()-t0:.0f}s", flush=True)

    sc = StandardScaler().fit(q_train[:200_000])

    # B. parameterised MLP
    if not SKIP_PARAM:
        res["param"] = {}
        for step in (100, 150):
            vals = list(range(50, 601, step))
            tr = [m for m in masses if m[0] in vals and m[1] in vals]
            te = [m for m in masses if m not in set(tr)]
            xs = np.vstack([np.c_[sig[m][0][:5000], np.tile(m, (5000, 1))] for m in tr])
            nb = len(xs)
            xb = np.c_[q_train[RNG.choice(len(q_train), nb, replace=nb > len(q_train))], np.array(tr)[RNG.integers(len(tr), size=nb)]]
            X = np.vstack([xs, xb])
            y = np.r_[np.ones(nb), np.zeros(nb)]
            Xn = np.c_[sc.transform(X[:, :6]), X[:, 6:] / 600.0]
            pbdt = HistGradientBoostingClassifier(max_iter=400, random_state=0).fit(X, y)
            nets = []
            for seed in (0, 1):
                torch.manual_seed(seed)
                net = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
                opt = torch.optim.Adam(net.parameters(), 1e-3)
                Xt, yt = torch.tensor(Xn, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
                for ep in range(30):
                    perm = torch.randperm(len(Xt))
                    for i in range(0, len(Xt), 1024):
                        b = perm[i : i + 1024]
                        loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[b]).squeeze(1), yt[b])
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                nets.append(net.eval())

            def pnn(x, m, net):
                with torch.no_grad():
                    z = np.c_[sc.transform(x), np.tile(np.array(m) / 600.0, (len(x), 1))]
                    return net(torch.tensor(z, dtype=torch.float32)).squeeze(1).numpy()

            out = []
            for m in tr + te:
                src = min(tr, key=lambda s: (s[0] - m[0]) ** 2 + (s[1] - m[1]) ** 2)
                rec = {
                    "tgt": list(m),
                    "held_out": m in set(te),
                    "oracle": oracle[m],
                    "pbdt": ev(m, lambda x: pbdt.predict_proba(np.c_[x, np.tile(m, (len(x), 1))])[:, 1]),
                    "pnn": [ev(m, lambda x, n=n: pnn(x, m, n)) for n in nets],
                    "nearest": ev(m, lambda x, c=omod[src]: c.predict_proba(x)[:, 1]),
                    "nearest_src": list(src),
                }
                out.append(rec)
            res["param"][str(step)] = {"n_train_tasks": len(tr), "results": out}
            Path(OUT).write_text(json.dumps(res, indent=1))
            print(f"[B] step {step}: {len(tr)} train tasks {time.time()-t0:.0f}s", flush=True)

    # C. circuit capacity
    if not SKIP_VQC:
        res["vqc"] = {}
        for m in VQC_POINTS:
            xs = sig[m][0]
            xb = q_train[RNG.choice(len(q_train), min(len(q_train), 4 * len(xs)), replace=False)]
            X = sc.transform(np.vstack([xs, xb]))
            y = np.r_[np.ones(len(xs)), np.zeros(len(xb))]
            w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
            Xt, yt, wt = (torch.tensor(a, dtype=torch.float32) for a in (X, y, w))
            xtest = sc.transform(np.vstack([sig[m][1], q_test]))
            ytest = np.r_[np.ones(len(sig[m][1])), np.zeros(len(q_test))]
            rec = {"oracle_bdt": oracle[m]}
            lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, y)
            rec["logreg"] = rej(ytest, lr.decision_function(xtest))

            def train(model, epochs=40, lr_=5e-3):
                opt = torch.optim.Adam(model.parameters(), lr_)
                for ep in range(epochs):
                    perm = torch.randperm(len(Xt))
                    for i in range(0, len(Xt), 512):
                        b = perm[i : i + 512]
                        out = model(Xt[b])
                        loss = (nn.functional.cross_entropy(out, yt[b].long(), reduction="none") * wt[b]).mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                with torch.no_grad():
                    s = []
                    for i in range(0, len(xtest), 50_000):
                        o = model(torch.tensor(xtest[i : i + 50_000], dtype=torch.float32))
                        s.append((o[:, 1] - o[:, 0]).numpy())
                return rej(ytest, np.concatenate(s)), sum(p.numel() for p in model.parameters() if p.requires_grad)

            class Hybrid(nn.Module):
                def __init__(self, depth, circuit=True):
                    super().__init__()
                    self.proj = nn.Linear(6, 6)
                    self.pqc = FastPQCModel(6, depth, init_type="random", bound_angles=True)
                    self.pqc.weights.data *= 0.1
                    self.circuit = circuit

                def forward(self, x):
                    a = self.proj(x)
                    return self.pqc(a) if self.circuit else self.pqc.fc(torch.tanh(a))

            for seed in (0, 1):
                for name, mk in (
                    ("vqc_d2", lambda: Hybrid(2)),
                    ("vqc_d4", lambda: Hybrid(4)),
                    ("no_circuit", lambda: Hybrid(2, circuit=False)),
                    ("mlp_32", lambda: nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))),
                ):
                    torch.manual_seed(seed)
                    r, npar = train(mk())
                    rec.setdefault(name, []).append(r)
                    rec[f"{name}_params"] = npar
            res["vqc"][f"{m[0]},{m[1]}"] = rec
            Path(OUT).write_text(json.dumps(res, indent=1))
            print(f"[C] {m} {time.time()-t0:.0f}s  oracle rej {oracle[m][2]:.0f}  vqc_d2 {[round(r[2]) for r in rec['vqc_d2']]}  vqc_d4 {[round(r[2]) for r in rec['vqc_d4']]}  twin {[round(r[2]) for r in rec['no_circuit']]}  mlp {[round(r[2]) for r in rec['mlp_32']]}  lr {rec['logreg'][2]:.0f}", flush=True)

    Path(OUT).write_text(json.dumps(res, indent=1))
    print(f"saved {OUT} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
