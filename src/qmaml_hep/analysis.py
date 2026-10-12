"""
Scoring for the round-2 unified benchmark (results/unified/*.json). The rule is fixed here, in code, before
the full result set exists; it is applied identically to every track, split and size.

Per held-out task and initializer:
  err_t   = |E_t - E0| / scale    (scale = |E0|; for SUSY QM, where E0 = 0 for unbroken SUSY, scale = m)
  T       = first iteration t with err_t <= TAU (TAU = 1e-3); if never reached within the budget B, T = B + 1
            (censored). Iterations are optimizer steps; each is one exact energy + gradient evaluation.
Primary comparison of Q-MAML against a competitor C, per task: d = log10(T_Q + 1) - log10(T_C + 1)
  (negative = Q-MAML reaches the threshold sooner). If neither Q-MAML nor C reaches TAU on any task of the
  track, d is replaced by log10(err_final_Q / err_final_C) (fallback, reported as such).
Competitors: the best classical initializer and the best warm-start baseline, each chosen post hoc as the
  one with the lowest mean d-statistic proxy (mean log10(T+1), or mean log10 final error under the fallback)
  pooled over all seeds -- i.e. the strongest opponent in hindsight. Initializers whose gradient at the
  starting point is < 1e-8 on every task (exact stationary points, e.g. Zero on several tracks) are excluded.
Uncertainty: hierarchical bootstrap (resample seeds, then tasks within each seed), 10 000 replicates; 95%
  percentile CI of mean d; two-sided bootstrap p = 2 min(P(d* <= 0), P(d* >= 0)).
Verdict: Win if the CI lies entirely below 0, Loss if entirely above 0, Tie otherwise; significance after
  Holm correction over all comparisons in the family is reported alongside.
Compute: pretraining costs epochs x n_train evaluations, warm-start collection n_train x B. Break-even number
  of held-out tasks N* = one-off cost / mean per-task saving (saving measured in iterations-to-threshold;
  censored runs make the saving a lower bound when the competitor is censored).
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

from qmaml_hep.paths import RESULTS

RES = os.path.join(RESULTS, "unified")
TAU = 1e-3
NBOOT = 10_000
CLASSICAL = ["zero", "pi", "uniform", "gaussian", "gauss_vacuum", "basis_paper"]
WARM = ["ws_mean", "ws_nearest", "ws_linear", "ws_kernel"]


def load(pattern):
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        r = json.load(open(p))
        key = os.path.basename(p).rsplit("_s", 1)[0]
        runs[key].append(r)
    return runs


def task_metrics(run):
    """-> {scheme: dict(T=[...], final=[...], init=[...], grad0=[...])} for one seed."""
    B = run["cfg"]["adapt_iters"]
    out = defaultdict(lambda: defaultdict(list))
    for row in run["test"]:
        sc = max(row["scale"], 1e-12)
        for s, r in row["runs"].items():
            err = np.abs(np.array(r["E"]) - row["E0"]) / sc
            hit = np.nonzero(err <= TAU)[0]
            out[s]["T"].append(int(hit[0]) if len(hit) else B + 1)
            out[s]["final"].append(float(max(err[-1], 1e-16)))
            out[s]["init"].append(float(max(err[0], 1e-16)))
            out[s]["grad0"].append(float(r["grad0"]))
    return out


def hboot(D, rng):
    """D: list over seeds of arrays over tasks. Returns (mean, lo, hi, p)."""
    D = [np.asarray(d, float) for d in D]
    stats = np.empty(NBOOT)
    ns = len(D)
    for b in range(NBOOT):
        ss = rng.integers(0, ns, ns)
        stats[b] = np.mean(np.concatenate([D[s][rng.integers(0, len(D[s]), len(D[s]))] for s in ss]))
    m = float(np.mean(np.concatenate(D)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = float(min(1.0, 2 * min((stats <= 0).mean(), (stats >= 0).mean())))
    return m, float(lo), float(hi), p


def score_group(runs, rng):
    M = [task_metrics(r) for r in runs]
    schemes = list(dict.fromkeys(s for m in M for s in m))
    B = runs[0]["cfg"]["adapt_iters"]
    stationary = {s for s in schemes if all(g < 1e-8 for m in M if s in m for g in m[s]["grad0"])}

    def logT(s, MM=None):
        return np.mean([np.log10(np.array(m[s]["T"]) + 1) for m in (MM or M) if s in m])

    def logF(s, MM=None):
        return np.mean([np.log10(m[s]["final"]) for m in (MM or M) if s in m])

    def reached(s, MM):
        return any(t <= B for m in MM for t in m[s]["T"])

    summary = {"n_seeds": len(runs), "budget": B, "stationary": sorted(stationary), "schemes": {}}
    for s in schemes:
        T = np.concatenate([m[s]["T"] for m in M if s in m])
        summary["schemes"][s] = {
            "n_seeds": sum(s in m for m in M),
            "frac_reached": float((T <= B).mean()),
            "median_T": float(np.median(T)),
            "geo_mean_final_err": float(10 ** logF(s)),
            "geo_mean_init_err": float(10 ** np.mean([np.log10(m[s]["init"]) for m in M if s in m])),
            "median_grad0": float(np.median(np.concatenate([m[s]["grad0"] for m in M if s in m]))),
        }
    comps = {}
    qvars = [q for q in ("qmaml", "qmaml_cm") if q in schemes]
    for q, fam, pool in [(q, f, p) for q in qvars for f, p in (("classical", CLASSICAL), ("warm", WARM))]:
        Mq = [m for m in M if q in m]
        cand = [s for s in schemes if s in pool and s not in stationary]
        if not cand or not Mq:
            continue
        use_T = reached(q, Mq) or any(reached(c, Mq) for c in cand)
        best = min(cand, key=(lambda c: logT(c, Mq)) if use_T else (lambda c: logF(c, Mq)))
        use_T = reached(q, Mq) or reached(best, Mq)
        if use_T:
            D = [np.log10(np.array(m[q]["T"]) + 1) - np.log10(np.array(m[best]["T"]) + 1) for m in Mq]
        else:
            D = [np.log10(m[q]["final"]) - np.log10(m[best]["final"]) for m in Mq]
        mean, lo, hi, p = hboot(D, rng)
        per_seed = [float(np.mean(d)) for d in D]
        key = fam if q == "qmaml" else f"{fam}|cm"
        comps[key] = {
            "q": q,
            "n_seeds": len(Mq),
            "opponent": best,
            "metric": "log10_iters_to_tau" if use_T else "log10_final_err_ratio",
            "mean_d": mean,
            "ci": [lo, hi],
            "p": p,
            "per_seed_mean_d": per_seed,
            "verdict": "win" if hi < 0 else ("loss" if lo > 0 else "tie"),
        }
        # compute accounting
        tq = np.concatenate([m[q]["T"] for m in Mq])
        tc = np.concatenate([m[best]["T"] for m in Mq])
        comps[key]["mean_iter_saving_per_task"] = float(np.mean(tc - tq))
    pre = runs[0]["pretrain"]["evals"]
    ws = runs[0]["warm_collect"]["evals"]
    summary["compute"] = {
        "pretrain_evals": pre,
        "warm_collect_evals": ws,
        "pretrain_final_loss_rel_change_last10": float(
            np.mean(
                [
                    abs(r["pretrain"]["loss"][-1] - r["pretrain"]["loss"][-10])
                    / max(abs(r["pretrain"]["loss"][-10]), 1e-12)
                    for r in runs
                ]
            )
        ),
    }
    cm_pre = next((r["_cm_pretrain_evals"] for r in runs if "_cm_pretrain_evals" in r), None)
    summary["compute"]["pretrain_evals_cm"] = cm_pre
    for c in comps.values():
        sav = c["mean_iter_saving_per_task"]
        cost = cm_pre if c["q"] == "qmaml_cm" else pre
        c["break_even_tasks"] = (cost / sav) if sav > 0 else None
    # the warm-start pipeline's own break-even against the best classical initializer
    cl = [s for s in schemes if s in CLASSICAL and s not in stationary]
    wm = [s for s in schemes if s in WARM]
    if cl and wm:
        bc, bw = min(cl, key=logT), min(wm, key=logT)
        sav = float(np.mean(np.concatenate([np.array(m[bc]["T"]) - np.array(m[bw]["T"]) for m in M])))
        summary["compute"]["warm_vs_classical"] = {
            "warm": bw,
            "classical": bc,
            "mean_iter_saving_per_task": sav,
            "break_even_tasks": (ws / sav) if sav > 0 else None,
        }
    summary["comparisons"] = comps
    summary["meta"] = {k: runs[0][k] for k in ("name", "qubits", "depth", "n_params", "desc_dim", "n_hidden")}
    summary["meta"]["train_tasks"] = len(runs[0]["train_tasks"])
    summary["meta"]["test_tasks"] = len(runs[0]["test"])
    summary["meta"]["cfg"] = runs[0]["cfg"]
    return summary


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def merge_compute_matched(groups):
    """Files '<base>_e<B>_s*' that contain only the qmaml initializer are the compute-matched Q-MAML runs
    (pretraining epochs = adaptation budget). Inject them into the base group as scheme 'qmaml_cm'
    (same seeds -> same tasks), and drop them as separate groups."""
    for key in list(groups):
        runs = groups[key]
        if "_e" not in key or set(runs[0]["test"][0]["runs"]) != {"qmaml"}:
            continue
        base = key.rsplit("_e", 1)[0]
        if base not in groups:
            continue
        by_seed = {r["seed"]: r for r in runs}
        for r in groups[base]:
            cm = by_seed.get(r["seed"])
            if cm is None:
                continue
            for row, row_cm in zip(r["test"], cm["test"]):
                assert row["task"] == row_cm["task"]
                row["runs"]["qmaml_cm"] = row_cm["runs"]["qmaml"]
            r["_cm_pretrain_evals"] = cm["pretrain"]["evals"]
        del groups[key]
    return groups


def main(pattern="*.json"):
    rng = np.random.default_rng(0)
    groups = merge_compute_matched(load(pattern))
    out = {}
    for key, runs in sorted(groups.items()):
        out[key] = score_group(runs, rng)
    fam = [(k, f) for k, v in out.items() for f in v["comparisons"]]
    adj = holm(np.array([out[k]["comparisons"][f]["p"] for k, f in fam])) if fam else []
    for (k, f), a in zip(fam, adj):
        out[k]["comparisons"][f]["p_holm"] = float(a)
    json.dump(out, open(os.path.join(RESULTS, "unified_summary.json"), "w"), indent=1)
    for k, v in out.items():
        line = f"{k:42s} n={v['n_seeds']} "
        for f, c in v["comparisons"].items():
            line += (
                f"| {f}:{c['opponent']:<10s} d={c['mean_d']:+.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] "
                f"{c['verdict']:<4s} p_h={c.get('p_holm', float('nan')):.3f} "
            )
        print(line)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "*.json")
