"""Summarise check2_big.json and pilot.json. Rejection R = 1/eps_b at eps_s = 0.3; differences in log10 R.
Paired per-point differences with a 95% bootstrap CI over grid points (seed-averaged first)."""

import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
L = np.log10
RNG = np.random.default_rng(0)


def ci(x, n=4000):
    x = np.asarray(x)
    b = np.median(x[RNG.integers(len(x), size=(n, len(x)))], axis=1)
    return f"{np.median(x):+.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]"


def lr(v):  # list of [auc, sic, rej] per seed -> mean log10 rej
    return float(np.mean([L(r[2]) for r in v]))


def auc(v):
    return float(np.mean([r[0] for r in v]))


if (D / "check2_big.json").exists():
    r = json.loads((D / "check2_big.json").read_text())
    o = r["oracle"]
    print(f"== check2 ({r['qcd']} QCD; SR {r.get('n_qcd_sr')}, test {r.get('n_qcd_test')})")
    a = np.array([v[0] for v in o.values()])
    R = np.array([v[2] for v in o.values()])
    print(f"oracle BDT over {len(o)} pts: AUC {a.min():.3f}-{a.max():.3f} (median {np.median(a):.3f}); rejection {R.min():.0f}-{R.max():.0f} (median {np.median(R):.0f})")
    for step, v in r.get("param", {}).items():
        res = v["results"]
        for ho in (True, False):
            rr = [x for x in res if x["held_out"] == ho]
            if not rr:
                continue
            pb = [L(x["pbdt"][2]) - L(x["oracle"][2]) for x in rr]
            pn = [lr(x["pnn"]) - L(x["oracle"][2]) for x in rr]
            nn = [L(x["nearest"][2]) - L(x["oracle"][2]) for x in rr]
            d = [a_ - b_ for a_, b_ in zip(pn, nn)]
            print(f" step {step} {'held-out' if ho else 'train   '} n={len(rr):3d}: vs oracle  pNN {ci(pn)}  pBDT {ci(pb)}  nearest {ci(nn)} | pNN - nearest {ci(d)}  pNN wins {np.mean(np.array(d) > 0):.2f}")
    for m, v in r.get("vqc", {}).items():
        row = " ".join(
            f"{k} {np.mean([x[2] for x in v[k]]):.0f}/{np.mean([x[0] for x in v[k]]):.3f}"
            for k in ("vqc_d2", "vqc_d4", "no_circuit", "mlp_32")
            if k in v
        )
        print(f" vqc @{m}: oracle {v['oracle_bdt'][2]:.0f}/{v['oracle_bdt'][0]:.3f}  logreg {v['logreg'][2]:.0f}/{v['logreg'][0]:.3f}  {row}   (rejection/AUC; params d2 {v.get('vqc_d2_params')} d4 {v.get('vqc_d4_params')} twin {v.get('no_circuit_params')} mlp {v.get('mlp_32_params')})")

if (D / "pilot.json").exists():
    p = json.loads((D / "pilot.json").read_text())
    print(f"\n== Q-MAML pilot: {len(p['train'])} train points, {len(p['test'])} held-out; {p['config']}")
    zs = p.get("zeroshot", [])
    for ho in (True, False):
        rr = [x for x in zs if x["held_out"] == ho]
        if not rr:
            continue
        tag = "held-out" if ho else "train"
        med = {k: np.median([lr(x[k]) for x in rr]) for k in ("joint", "param", "qmaml", "nearest")}
        meda = {k: np.median([auc(x[k]) for x in rr]) for k in ("joint", "param", "qmaml", "nearest")}
        print(f" zero-shot {tag} n={len(rr)}: median log10 R  " + "  ".join(f"{k} {v:.2f} (AUC {meda[k]:.3f})" for k, v in med.items()))
        for opp in ("param", "nearest", "joint"):
            d = [lr(x["qmaml"]) - lr(x[opp]) for x in rr]
            print(f"   qmaml - {opp:7s}: {ci(d)}  qmaml better on {np.mean(np.array(d) > 0):.2f}")
        pt = [x for x in rr if "pertask" in x]
        if pt:
            for k in ("qmaml", "param", "nearest"):
                d = [lr(x[k]) - lr(x["pertask"]) for x in pt]
                print(f"   {k:7s} - pertask (n={len(pt)}): {ci(d)}")
    fs = p.get("fewshot", [])
    for K in sorted({x["K"] for x in fs}):
        rr = [x for x in fs if x["K"] == K]
        med = {k: np.median([lr(x[k]) for x in rr]) for k in ("qmaml", "param", "joint", "nearest")}
        print(f" few-shot K={K} n={len(rr)}: median log10 R  " + "  ".join(f"{k} {v:.2f}" for k, v in med.items()))
        for opp in ("param", "nearest", "joint"):
            d = [lr(x["qmaml"]) - lr(x[opp]) for x in rr]
            print(f"   qmaml - {opp:7s}: {ci(d)}  qmaml better on {np.mean(np.array(d) > 0):.2f}")
        zsm = {tuple(x["tgt"]): x for x in zs}
        d = [lr(x["qmaml"]) - lr(zsm[tuple(x["tgt"])]["qmaml"]) for x in rr if tuple(x["tgt"]) in zsm]
        if d:
            print(f"   qmaml adapted - qmaml zero-shot: {ci(d)}")

if (D / "pilot_matched.json").exists() and (D / "pilot.json").exists():
    mt = json.loads((D / "pilot_matched.json").read_text())
    p = json.loads((D / "pilot.json").read_text())
    zs = {tuple(x["tgt"]): x for x in p["zeroshot"]}
    print(f"\n== capacity-matched conditioning (params: hyper_enc {mt.get('hyper_enc_params')}, param_deep {mt.get('param_deep_params')})")
    for ho in (True, False):
        rr = [x for x in mt["zeroshot"] if x["held_out"] == ho]
        tag = "held-out" if ho else "train"
        print(f" {tag} n={len(rr)}: median log10 R  hyper_enc {np.median([lr(x['hyper_enc']) for x in rr]):.2f}  param_deep {np.median([lr(x['param_deep']) for x in rr]):.2f}  qmaml {np.median([lr(zs[tuple(x['tgt'])]['qmaml']) for x in rr]):.2f}")
        for k in ("hyper_enc", "param_deep"):
            d = [lr(zs[tuple(x["tgt"])]["qmaml"]) - lr(x[k]) for x in rr]
            print(f"   qmaml - {k:10s}: {ci(d)}  qmaml better on {np.mean(np.array(d) > 0):.2f}")

if (D / "pilot_deep.json").exists() and (D / "pilot_matched.json").exists():
    dp = {tuple(x["tgt"]): x for x in json.loads((D / "pilot_deep.json").read_text())["zeroshot"]}
    mt = {tuple(x["tgt"]): x for x in json.loads((D / "pilot_matched.json").read_text())["zeroshot"]}
    zs = {tuple(x["tgt"]): x for x in json.loads((D / "pilot.json").read_text())["zeroshot"]}
    print("\n== deep feature front end (same 6->256->256->6 network on the features)")
    for ho in (True, False):
        ks = [k for k, x in dp.items() if x["held_out"] == ho]
        tag = "held-out" if ho else "train"
        print(f" {tag} n={len(ks)}: median log10 R  qmaml_deep {np.median([lr(dp[k]['qmaml_deep']) for k in ks]):.2f}  joint_deep {np.median([lr(dp[k]['joint_deep']) for k in ks]):.2f}  param_deep {np.median([lr(mt[k]['param_deep']) for k in ks]):.2f}")
        for name, f in (
            ("qmaml_deep - param_deep", lambda k: lr(dp[k]["qmaml_deep"]) - lr(mt[k]["param_deep"])),
            ("qmaml_deep - joint_deep", lambda k: lr(dp[k]["qmaml_deep"]) - lr(dp[k]["joint_deep"])),
            ("qmaml_deep - qmaml     ", lambda k: lr(dp[k]["qmaml_deep"]) - lr(zs[k]["qmaml"])),
        ):
            d = [f(k) for k in ks]
            print(f"   {name}: {ci(d)}  first better on {np.mean(np.array(d) > 0):.2f}")
