"""
Corrected gradient-variance (barren-plateau) scan, replacing notebooks 03/04's single random reference
parameter. Notebook 04 drew a new random reference index per qubit count; at 10 qubits it drew (1,6,2), the
final-layer RZ(omega) of StronglyEntanglingLayers, whose gradient is identically zero (a diagonal gate
followed only by CNOTs and Z measurements), so the reported "1e-35 variance" was round-off, not a plateau.

Here the full gradient vector is computed for every probe sample; parameters whose gradient is identically
zero across all samples and schemes are reported and excluded; the statistic is the median over the remaining
(live) parameters of Var[dC/dtheta_k], plus the variance for one fixed live parameter (layer 0, wire 0, RY
angle). A fully random U[0, 2pi) initialization (McClean et al.'s setting) is included as the reference.

Usage:  python run_barren_plateau_v2.py higgs    |    python run_barren_plateau_v2.py heisenberg
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from qmaml_hep.paths import DATA, RESULTS

OUT = os.path.join(RESULTS, "barren_plateau_v2")
N_PROBE = 60


def summarize(G):
    """G: {scheme: (n_samples, n_params)} -> stats with structurally dead parameters removed."""
    allg = np.concatenate(list(G.values()))
    dead = np.all(np.abs(allg) < 1e-12, axis=0)
    out = {"n_params": int(allg.shape[1]), "n_dead": int(dead.sum())}
    for s, g in G.items():
        v = g[:, ~dead].var(0)
        out[s] = {
            "median_var_live": float(np.median(v)),
            "mean_var_live": float(v.mean()),
            "var_fixed_param": float(g[:, 1].var()),
        }  # flat index 1 = (layer 0, wire 0, theta/RY)
    return out


def higgs():
    from qmaml_hep.classification.extractor import TabularFeatureExtractor
    from qmaml_hep.classification.higgs import ALL_FEATURE_COLS, load_higgs
    from qmaml_hep.classification.tasks import make_tasks
    from qmaml_hep.learner import Learner
    from qmaml_hep.simulators import FastPQCModel

    df = load_higgs(180_000, os.path.join(DATA, "higgs", "higgs_subsample.csv"))
    n = len(df)
    test, train = df.iloc[: n // 5], df.iloc[n // 5 :]
    Xtr = torch.from_numpy(train[ALL_FEATURE_COLS].values.astype(np.float32))
    res = {}
    for depth in (2, 8):
        for nq in (4, 6, 8, 10, 12):
            torch.manual_seed(0)
            np.random.seed(0)
            pre = make_tasks(
                train, ALL_FEATURE_COLS, "m_bb", bin_count=6, support_size=8, query_size=8, tasks_per_bin=3, seed=0
            )
            probe = make_tasks(
                test,
                ALL_FEATURE_COLS,
                "m_bb",
                bin_count=6,
                support_size=8,
                query_size=8,
                tasks_per_bin=N_PROBE // 6,
                seed=500,
            )
            ext = TabularFeatureExtractor(len(ALL_FEATURE_COLS), 512, nq)
            ext.set_norm_stats(Xtr.mean(0), Xtr.std(0))
            pqc = FastPQCModel(nq, depth)
            shape = (depth, nq, 3)
            learner = Learner(512, shape, 256, n_hidden=2).double()
            opt = torch.optim.Adam(learner.parameters(), lr=0.01)
            ce = nn.CrossEntropyLoss()
            for _ in range(25):
                for t in pre:
                    opt.zero_grad()
                    th = 0.01 * learner(ext.embed(t["support_X"]).mean(0).double())
                    ce(pqc(ext(t["support_X"]).detach(), weights_override=th), t["support_y"]).backward()
                    torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0)
                    opt.step()
            G = {}
            for scheme in ("qmaml", "uniform", "gaussian", "random"):
                rows = []
                for t in probe:
                    if scheme == "qmaml":
                        with torch.no_grad():
                            th = 0.01 * learner(ext.embed(t["support_X"]).mean(0).double())
                    elif scheme == "uniform":
                        th = (torch.rand(shape, dtype=torch.float64) * 2 - 1) * 0.05 * np.pi
                    elif scheme == "gaussian":
                        th = torch.normal(0.0, (1.0 / (4 * 2 * (depth + 2))) ** 0.5, size=shape, dtype=torch.float64)
                    else:
                        th = torch.rand(shape, dtype=torch.float64) * 2 * np.pi
                    th = th.clone().requires_grad_(True)
                    with torch.no_grad():
                        a = ext(t["support_X"])
                    ce(pqc(a, weights_override=th), t["support_y"]).backward()
                    rows.append(th.grad.detach().numpy().ravel().copy())
                G[scheme] = np.stack(rows)
            res[f"d{depth}_q{nq}"] = summarize(G)
            print(
                f"[higgs d={depth} q={nq}] dead={res[f'd{depth}_q{nq}']['n_dead']}/{res[f'd{depth}_q{nq}']['n_params']} "
                + " ".join(f"{s}={res[f'd{depth}_q{nq}'][s]['median_var_live']:.2e}" for s in G),
                flush=True,
            )
            json.dump(res, open(os.path.join(OUT, "higgs.json"), "w"), indent=1)


def heisenberg():
    from qmaml_hep.initializers import classical_init
    from qmaml_hep.learner import Learner
    from qmaml_hep.systems.heisenberg import HeisenbergPQC, param_shape, sample_task_space

    res = {}
    for depth in (3,):
        for nq in (4, 6, 8, 10, 12):
            torch.manual_seed(0)
            np.random.seed(0)
            shape = param_shape(nq, depth)
            pqc = HeisenbergPQC(nq, depth)
            learner = Learner(3, shape, 256, n_hidden=2).double()
            opt = torch.optim.Adam(learner.parameters(), lr=0.001)
            for _ in range(15):
                for Jx, Jy, Jz in sample_task_space(16, seed=0):
                    opt.zero_grad()
                    pqc.cost(0.3 * learner(torch.tensor([Jx, Jy, Jz], dtype=torch.float64)), Jx, Jy, Jz).backward()
                    torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0)
                    opt.step()
            G = {}
            probe = sample_task_space(N_PROBE, seed=500)
            for scheme in ("qmaml", "uniform", "gaussian", "random"):
                rows = []
                for Jx, Jy, Jz in probe:
                    if scheme == "qmaml":
                        with torch.no_grad():
                            th = 0.3 * learner(torch.tensor([Jx, Jy, Jz], dtype=torch.float64))
                    elif scheme == "random":
                        th = torch.rand(shape, dtype=torch.float64) * 2 * np.pi
                    else:
                        th = classical_init(shape, scheme, depth)
                    th = th.clone().requires_grad_(True)
                    pqc.cost(th, Jx, Jy, Jz).backward()
                    rows.append(th.grad.detach().numpy().ravel().copy())
                G[scheme] = np.stack(rows)
            res[f"d{depth}_q{nq}"] = summarize(G)
            print(
                f"[heis d={depth} q={nq}] dead={res[f'd{depth}_q{nq}']['n_dead']} "
                + " ".join(f"{s}={res[f'd{depth}_q{nq}'][s]['median_var_live']:.2e}" for s in G),
                flush=True,
            )
            json.dump(res, open(os.path.join(OUT, "heisenberg.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(1)
    {"higgs": higgs, "heisenberg": heisenberg}[sys.argv[1]]()
