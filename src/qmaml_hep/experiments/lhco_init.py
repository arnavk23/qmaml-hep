"""Initialise-then-optimise on the LHCO (m_X, m_Y) grid: is the Learner's output a good starting point?

The zero-shot study in lhco_routes evaluates the Learner-conditioned classifier (arm "qmaml" there) without further
optimisation. This stage tests what Q-MAML claims: that the Learner's output is a good place to *start* optimising.

Model. At each target mass the same 92-parameter classifier is optimised: Linear(6 -> 6) front end, 36 circuit
angles theta (6 qubits, depth 2) and Linear(6 -> 2) head (lhco_routes.Arm("joint") architecture). Every weight is
trained. Only the starting point differs:

    learner   front end and head of the trained Learner-conditioned classifier, theta = Learner(m) at the target
    nearest   the single-mass classifier trained at the nearest training mass (lhco_routes "nearest")
    shared    the classifier trained on all training masses without mass information (lhco_routes "joint")
    fresh     a new random initialisation (seeded like every Arm)

Source models are retrained exactly as in the lhco_routes report stage (same seed and selected learning rate; on
the same CPU their zero-shot numbers reproduce the committed report files, which each job checks and records).

Adaptation (fixed before the runs). Adam on all 92 weights, one batch per step of EV_PER_TASK signal events from the
target's training split and as many QCD training events. The batches depend only on (seed, mass), so every start
sees identical events. After each checkpoint step in CHECKPOINTS, R = 1 / eps_b at eps_s = 0.3 is computed on the
target's evaluation signal and the QCD test events (validation QCD in the tune stage). R_oracle is the per-point
BDT (oracle_bdt.json, same QCD test events). steps = the first checkpoint with R >= FRAC * R_oracle, or None if not
reached by MAX_STEPS. Without --early-stop every point runs the full schedule; with it (the six CPU report runs of
2026-11-10, and all tune runs) a point stops at that checkpoint, so its trajectory ends there. The batches depend only
on (seed, mass), so a stopped trajectory is the prefix of the full one.

Scoring (fixed after the tune stage, before any report result was looked at). Fewer than a quarter of validation
points reach the threshold for any start, so steps-to-threshold cannot separate the starts. The primary quantity is
g_k = log10(R_k / R_oracle) at the step budgets BUDGETS; comparisons are learner minus each other start, paired by
seed and point, hierarchical bootstrap (seeds, then points), Holm across the 3 x len(BUDGETS) family. A trajectory
that stopped early is carried forward at its last value (it was already within FRAC of the oracle); score() reports
how many points that affects. Fraction reached and median steps are reported as secondary.

Tuning. The adaptation learning rate is chosen per start from INIT_LRS on the validation masses (100 GeV grid,
seed 100), by the largest fraction of points reached, then the lowest median steps. Test masses never enter the
choice. Reporting: spacing 100, seeds 0-4.

Runs on a GPU with QMAML_DEVICE=cuda (notebooks/lhco_init_colab.ipynb). Results are then not bitwise equal to CPU
runs, and the reproduction check records the difference instead of identity.

Usage
    python -m qmaml_hep.experiments.lhco_init jobs --stage tune     # one command per line
    python -m qmaml_hep.experiments.lhco_init select
    python -m qmaml_hep.experiments.lhco_init jobs --stage report
    python -m qmaml_hep.experiments.lhco_init score
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from qmaml_hep.experiments import lhco_routes as L

OUT = os.path.join(L.RESULTS, "lhco_init")
STARTS = ["learner", "nearest", "shared", "fresh"]
SOURCE_ARM = {"learner": "qmaml", "nearest": "nearest", "shared": "joint"}
CHECKPOINTS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400]
MAX_STEPS, FRAC = CHECKPOINTS[-1], 0.9
BUDGETS = [0, 25, 100, 400]
INIT_LRS = [1e-3, 3e-3, 1e-2]
SPACING, TUNE_SEED, REPORT_SEEDS = 100, 100, [0, 1, 2, 3, 4]


def cpu_info():
    flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    flags = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    gpu = torch.cuda.get_device_name(L.DEVICE) if L.DEVICE.type == "cuda" else None
    return dict(cpu=flags or platform.processor(), torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
                mkl_enable_instructions=os.environ.get("MKL_ENABLE_INSTRUCTIONS"), torch=torch.__version__,
                device=str(L.DEVICE), gpu=gpu, cuda=torch.version.cuda)  # fmt: skip


def start_model(start, m, seed, sources, train_m):
    """-> a fresh Arm("joint") (92 trainable weights) holding the starting point for target mass m."""
    model = L.Arm("joint", seed)
    if start == "fresh":
        return model
    if start == "shared":
        model.load_state_dict(sources["shared"].state_dict())
    elif start == "nearest":
        model.load_state_dict(sources["nearest"][L.nearest_of(m, train_m)].state_dict())
    elif start == "learner":
        src = sources["learner"]
        model.front.load_state_dict(src.front.state_dict())
        model.pqc.fc.load_state_dict(src.pqc.fc.state_dict())
        with torch.no_grad():
            model.pqc.weights.copy_(src.theta(m).to(model.pqc.weights.dtype))
    model.pqc.weights.requires_grad_(True)
    return model


def steps_to_threshold(model, data, m, qcd, lr, seed, r_oracle, early_stop=False):
    rng = np.random.default_rng([seed, m[0], m[1], 11])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr)
    traj, step, first = [], 0, None
    for c in CHECKPOINTS:
        while step < c:
            x, y = data.batch(m, L.EV_PER_TASK, rng)
            loss = nn.functional.cross_entropy(model(x, m), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        r = L.evaluate(model, data, m, qcd)[2]
        traj.append([c, r])
        if first is None and r >= FRAC * r_oracle:
            first = c
            if early_stop:
                break
    return first, traj


def run(stage, start, lr, seed, threads=1, early_stop=False):
    torch.set_num_threads(threads)
    t0 = time.time()
    data = L.Data()
    tr, va, te = L.grid_split(data.masses, SPACING)
    targets, qcd = (va, data.q_val) if stage == "tune" else (te, data.q_test)
    oracle = json.load(open(os.path.join(L.OUT, "oracle_bdt.json")))
    lrs = L.selected_lrs()
    sources, check = {}, {}
    if start in ("learner", "shared"):
        arm = SOURCE_ARM[start]
        sources[start] = L.train(L.Arm(arm, seed), data, tr, seed, lrs[arm], L.STEPS)
    elif start == "nearest":
        need = sorted({L.nearest_of(m, tr) for m in targets})
        sources["nearest"] = {s: L.train(L.Arm("nearest", seed), data, [s], seed, lrs["nearest"], L.SINGLE_STEPS)
                              for s in need}  # fmt: skip
    # On the report stage, the zero-shot learner/shared/nearest models must reproduce the committed report files.
    if stage == "report" and start != "fresh":
        arm = SOURCE_ARM[start]
        ref = json.load(open(os.path.join(L.OUT, f"report_sp{SPACING}_{arm}_lr{lrs[arm]:g}_s{seed}.json")))
        m0 = targets[0]
        src = sources["nearest"][L.nearest_of(m0, tr)] if start == "nearest" else sources[start]
        r0 = L.evaluate(src, data, m0, qcd)[2]
        check = dict(mass=m0, r=r0, committed=ref["points"][f"{m0[0]},{m0[1]}"][2])
        check["identical"] = check["r"] == check["committed"]
    points = {}
    for m in targets:
        key = f"{m[0]},{m[1]}"
        model = start_model(start, m, seed, sources, tr)
        steps, traj = steps_to_threshold(model, data, m, qcd, lr, seed, oracle[key][2], early_stop)
        points[key] = dict(steps=steps, r_oracle=oracle[key][2], trajectory=traj)
    res = dict(stage=stage, spacing=SPACING, start=start, lr=lr, seed=seed, train=tr, test=targets,
               n_params=sum(p.numel() for p in L.Arm("joint", seed).parameters()), checkpoints=CHECKPOINTS,
               frac=FRAC, early_stop=early_stop, source_lrs={k: lrs[v] for k, v in SOURCE_ARM.items()}, reproduction_check=check,
               environment=cpu_info(), points=points, seconds=time.time() - t0)  # fmt: skip
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{stage}_{start}_lr{lr:g}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(res, f)
    print(f"[lhco_init] {path} {res['seconds']:.0f}s", flush=True)


def selected():
    path = os.path.join(OUT, "selected_lr.json")
    if not os.path.isfile(path):
        raise SystemExit("run the tune stage and `select` first")
    return json.load(open(path))


def jobs(stage):
    py = f"{sys.executable} -m qmaml_hep.experiments.lhco_init run"
    if stage == "tune":
        for start in STARTS:
            for lr in INIT_LRS:
                print(f"{py} --stage tune --start {start} --lr {lr:g} --seed {TUNE_SEED} --early-stop")
        return
    lrs = selected()
    for start in STARTS:
        for s in REPORT_SEEDS:
            print(f"{py} --stage report --start {start} --lr {lrs[start]:g} --seed {s}")


def _summary(rec):
    st = [p["steps"] for p in rec["points"].values()]
    reached = [s for s in st if s is not None]
    capped = [s if s is not None else np.inf for s in st]
    return dict(frac_reached=len(reached) / len(st), median_steps=float(np.median(capped)))


def select():
    out, table = {}, {}
    for start in STARTS:
        rows = []
        for lr in INIT_LRS:
            f = os.path.join(OUT, f"tune_{start}_lr{lr:g}_s{TUNE_SEED}.json")
            if os.path.isfile(f):
                rows.append((lr, _summary(json.load(open(f)))))
        if len(rows) < len(INIT_LRS):
            raise SystemExit(f"tuning incomplete for {start}: {len(rows)} of {len(INIT_LRS)}")
        best = max(rows, key=lambda r: (r[1]["frac_reached"], -r[1]["median_steps"], -r[0]))
        out[start], table[start] = best[0], {f"{lr:g}": s for lr, s in rows}
    with open(os.path.join(OUT, "selected_lr.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT, "tune_summary.json"), "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps(out), json.dumps(table, indent=1))


def _gaps(rec):
    """-> (keys, array (n_points, len(CHECKPOINTS)) of log10(R / R_oracle), number of carried-forward points)."""
    keys = sorted(rec["points"])
    g, carried = np.empty((len(keys), len(CHECKPOINTS))), 0
    for i, k in enumerate(keys):
        p = rec["points"][k]
        r = dict(p["trajectory"])
        last = None
        for j, c in enumerate(CHECKPOINTS):
            last = r.get(c, last)
            g[i, j] = np.log10(max(last, 1.0) / p["r_oracle"])
        carried += len(r) < len(CHECKPOINTS)
    return keys, g, carried


def score(rng_seed=0):
    """Gap to the oracle at each budget per start, learner-minus-start paired differences (hierarchical bootstrap,
    Holm), plus fraction reaching FRAC * R_oracle and median steps. Uses the seeds each pair has in common."""
    from qmaml_hep import analysis as A

    lrs = selected()
    runs = {a: {} for a in STARTS}
    for a in STARTS:
        for s in REPORT_SEEDS:
            f = os.path.join(OUT, f"report_{a}_lr{lrs[a]:g}_s{s}.json")
            if os.path.isfile(f):
                runs[a][s] = json.load(open(f))
    if not runs["learner"]:
        raise SystemExit("no learner runs")
    bi = [CHECKPOINTS.index(b) for b in BUDGETS]
    out = dict(lrs=lrs, frac=FRAC, budgets=BUDGETS, checkpoints=CHECKPOINTS, starts={}, comparisons=[])
    G = {a: {s: _gaps(r) for s, r in rs.items()} for a, rs in runs.items()}
    for a in STARTS:
        if not G[a]:
            continue
        allg = np.concatenate([g for _, g, _ in G[a].values()])
        st = [p["steps"] for r in runs[a].values() for p in r["points"].values()]
        reached = [x for x in st if x is not None]
        out["starts"][a] = dict(seeds=sorted(G[a]), n_points=len(allg), carried_forward=sum(c for *_, c in G[a].values()),
                                mean_gap=allg.mean(0).tolist(), median_gap=np.median(allg, 0).tolist(),
                                frac_reached=len(reached) / len(st), frac_at_step0=float(np.mean([x == 0 for x in st])),
                                median_steps=float(np.median([x if x is not None else np.inf for x in st])),
                                devices=sorted({r["environment"].get("device", "cpu") for r in runs[a].values()}))  # fmt: skip
    rng = np.random.default_rng(rng_seed)
    for a in STARTS[1:]:
        seeds = sorted(set(G["learner"]) & set(G[a]))
        if not seeds:
            continue
        assert all(G["learner"][s][0] == G[a][s][0] for s in seeds)
        for b, j in zip(BUDGETS, bi):
            D = [G["learner"][s][1][:, j] - G[a][s][1][:, j] for s in seeds]
            m, lo, hi, p = A.hboot(D, rng)
            out["comparisons"].append(dict(other=a, budget=b, seeds=seeds, diff=m, ci=[lo, hi], p=p,
                                           learner_better=float(np.mean(np.concatenate(D) > 0))))  # fmt: skip
    if out["comparisons"]:
        adj = A.holm([c["p"] for c in out["comparisons"]])
        for c, q in zip(out["comparisons"], adj):
            c["p_holm"] = float(q)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("gap to oracle, mean log10(R/R_oracle) at steps " + " ".join(map(str, BUDGETS)))
    for a, v in out["starts"].items():
        gaps = " ".join(f"{v['mean_gap'][j]:+.2f}" for j in bi)
        print(f"  {a:8s} {gaps}   reached {v['frac_reached']:.2f}  seeds {v['seeds']}  carried {v['carried_forward']}")
    print("learner minus start (positive = learner closer to the oracle)")
    for c in out["comparisons"]:
        print(f"  vs {c['other']:8s} @{c['budget']:3d}: {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
              f"  p_holm {c['p_holm']:.3g}  learner better on {c['learner_better']:.0%}  seeds {c['seeds']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stage", choices=["tune", "report"], required=True)
    r.add_argument("--start", choices=STARTS, required=True)
    r.add_argument("--lr", type=float, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--early-stop", action="store_true")
    j = sub.add_parser("jobs")
    j.add_argument("--stage", choices=["tune", "report"], required=True)
    sub.add_parser("select")
    sub.add_parser("score")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.stage, a.start, a.lr, a.seed, a.threads, a.early_stop)
    elif a.cmd == "jobs":
        jobs(a.stage)
    elif a.cmd == "select":
        select()
    else:
        score()


if __name__ == "__main__":
    main()
