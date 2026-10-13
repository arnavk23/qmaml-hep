"""Figures, from saved results only (python scripts/make_figures.py; writes figures/*.pdf).
  fig_r2_scoreboard.pdf   per track: mean log10 iteration ratio (Q-MAML / opponent) with 95% hierarchical-bootstrap CI,
                          against the best classical and the best warm-start initializer; plain and compute-matched Q-MAML
  fig_r2_traj.pdf         median relative error vs iteration (all seeds x tasks), Q-MAML / compute-matched / best classical /
                          best warm start
  fig_r2_grad0.pdf        gradient norm at the initial point, per track and initializer
  fig_r2_size.pdf         the comparison vs qubit count (size extensions)
  fig_r2_class.pdf        classification accuracy, all arms, with classical baselines
  fig_r2_bp.pdf           corrected gradient-variance scan (live parameters only)
  fig_r2_replication.pdf  paper-scale replication (12-qubit Heisenberg, 10-qubit LiH), plotted as in the paper's Fig. 4
Colour: Okabe-Ito; every series also has its own marker/line style (colour is never the only cue).
"""

import glob
import json
import os

# fixed PDF timestamps, so regenerating unchanged figures produces byte-identical files
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)
C = {
    "qmaml": "#D55E00",
    "qmaml_cm": "#D55E00",
    "classical": "#0072B2",
    "warm": "#009E73",
    "grey": "#7F7F7F",
    "zero": "#7F7F7F",
    "pi": "#0072B2",
    "uniform": "#56B4E9",
    "gaussian": "#CC79A7",
    "ws_mean": "#009E73",
    "ws_nearest": "#009E73",
    "ws_linear": "#009E73",
    "ws_kernel": "#009E73",
    "gauss_vacuum": "#E69F00",
    "basis_paper": "#E69F00",
}
LS = {"qmaml": "-", "qmaml_cm": "--", "classical": "-.", "warm": ":"}
ORDER = [
    "A_schwinger",
    "B_z2",
    "B_z2_hva",
    "C_scalar",
    "D_su2",
    "E_neutrino",
    "F_susyqm",
    "G_d8",
    "H_nuclear",
    "I_yukawa",
    "J_lmg",
]
LABEL = {
    "A_schwinger": "A Schwinger",
    "B_z2": "B $\\mathbb{Z}_2$ (HEA)",
    "B_z2_hva": "B $\\mathbb{Z}_2$ (HVA)",
    "C_scalar": "C $\\phi^4$",
    "D_su2": "D SU(2)",
    "E_neutrino": "E neutrinos",
    "F_susyqm": "F SUSY QM",
    "G_d8": "G $\\mathbb{D}_8$",
    "H_nuclear": "H nuclei",
    "I_yukawa": "I Yukawa",
    "J_lmg": "J LMG",
}
TAU = 1e-3


def summary():
    return json.load(open(os.path.join(R, "unified_summary.json")))


def scoreboard(S, split="interp"):
    rows = [k for k in ORDER if f"{k}_{split}_h2" in S]
    fig, ax = plt.subplots(figsize=(6.3, 0.32 * len(rows) + 0.9))
    specs = [
        ("classical", "o", 0.18, "vs best classical"),
        ("warm", "s", -0.18, "vs best warm start"),
        ("classical|cm", "o", 0.06, "compute-matched, vs classical"),
        ("warm|cm", "s", -0.06, "compute-matched, vs warm start"),
    ]
    for y, k in enumerate(rows[::-1]):
        comps = S[f"{k}_{split}_h2"]["comparisons"]
        for key, mk, off, lab in specs:
            if key not in comps:
                continue
            c = comps[key]
            col = C["warm"] if key.startswith("warm") else C["classical"]
            filled = "|cm" not in key
            ax.plot(c["ci"], [y + off] * 2, color=col, lw=1.0, alpha=0.9 if filled else 0.6)
            ax.plot(
                c["mean_d"],
                y + off,
                mk,
                ms=4.5,
                color=col,
                mfc=col if filled else "white",
                mew=0.9,
                label=lab if y == 0 else None,
            )
        n = S[f"{k}_{split}_h2"]["n_seeds"]
        ax.text(1.02, y, f"n={n}", transform=ax.get_yaxis_transform(), fontsize=6, va="center", color="0.35")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[k] for k in rows[::-1]])
    ax.set_xlabel(
        r"$\langle\log_{10}(T_{\mathrm{Q\text{-}MAML}}+1)-\log_{10}(T_{\mathrm{opp}}+1)\rangle$"
        r"   ($T$ = iterations to $10^{-3}$ relative error; $<0$: Q-MAML faster)"
    )
    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_r2_scoreboard_{split}.pdf"))
    plt.close(fig)


def load_runs(key):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, "unified", f"{key}_s*.json")))]
    cm = {
        json.load(open(p))["seed"]: json.load(open(p))
        for p in glob.glob(os.path.join(R, "unified", f"{key}_e*_s*.json"))
    }
    for r in runs:
        if r["seed"] in cm and set(cm[r["seed"]]["test"][0]["runs"]) == {"qmaml"}:
            for row, rc in zip(r["test"], cm[r["seed"]]["test"]):
                row["runs"]["qmaml_cm"] = rc["runs"]["qmaml"]
    return runs


def err_curves(runs, s):
    out = []
    for r in runs:
        for row in r["test"]:
            if s in row["runs"]:
                out.append(np.abs(np.array(row["runs"][s]["E"]) - row["E0"]) / max(row["scale"], 1e-12))
    return np.array(out) if out else None


def trajectories(S, split="interp"):
    rows = [k for k in ORDER if f"{k}_{split}_h2" in S]
    nc = 4
    nr = int(np.ceil(len(rows) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(7.0, 1.55 * nr + 0.4), squeeze=False)
    for ax, k in zip(axes.flat, rows):
        s = S[f"{k}_{split}_h2"]
        runs = load_runs(f"{k}_{split}_h2")
        series = [
            ("qmaml", "Q-MAML"),
            ("qmaml_cm", "Q-MAML (compute-matched)"),
            (s["comparisons"].get("classical", {}).get("opponent"), "best classical"),
            (s["comparisons"].get("warm", {}).get("opponent"), "best warm start"),
        ]
        for (sch, lab), style in zip(series, ["qmaml", "qmaml_cm", "classical", "warm"]):
            E = err_curves(runs, sch) if sch else None
            if E is None:
                continue
            med = np.median(np.log10(np.maximum(E, 1e-16)), 0)
            q1, q3 = np.percentile(np.log10(np.maximum(E, 1e-16)), [25, 75], axis=0)
            col = C[style] if style in C else C[sch]
            ax.plot(med, color=col, ls=LS[style], lw=1.1, label=lab if k == rows[0] else None)
            ax.fill_between(range(len(med)), q1, q3, color=col, alpha=0.10, lw=0)
        ax.axhline(np.log10(TAU), color="0.5", lw=0.5, ls=":")
        ax.set_title(f"{LABEL[k]} (n={s['n_seeds']})", fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(rows) :]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}$ rel. error", fontsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel("iteration", fontsize=6.5)
    fig.legend(loc="lower right", bbox_to_anchor=(0.98, 0.04), fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_r2_traj_{split}.pdf"))
    plt.close(fig)


def grad0(S, split="interp"):
    rows = [k for k in ORDER if f"{k}_{split}_h2" in S]
    schemes = ["qmaml", "qmaml_cm", "zero", "pi", "uniform", "gaussian", "ws_nearest", "ws_kernel"]
    mk = {
        "qmaml": "D",
        "qmaml_cm": "d",
        "zero": "x",
        "pi": "^",
        "uniform": "v",
        "gaussian": "o",
        "ws_nearest": "s",
        "ws_kernel": "P",
    }
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    for i, k in enumerate(rows):
        sc = S[f"{k}_{split}_h2"]["schemes"]
        for j, s in enumerate(schemes):
            if s in sc:
                v = max(sc[s]["median_grad0"], 1e-12)
                ax.plot(
                    i + (j - len(schemes) / 2) * 0.08,
                    v,
                    mk[s],
                    ms=3.6,
                    color=C.get(s, "k"),
                    mfc="white" if s in ("qmaml_cm", "ws_kernel") else C.get(s, "k"),
                    label=s.replace("_", " ") if i == 0 else None,
                )
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([LABEL[k] for k in rows], rotation=35, ha="right")
    ax.set_ylabel(r"median $\|\nabla_\theta E(\theta_0)\|$")
    ax.legend(ncol=8, fontsize=5.6, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.1, columnspacing=0.8)
    ax.grid(axis="y", color="0.93", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_r2_grad0.pdf"))
    plt.close(fig)


def size_trend(S):
    # (result key, x position, depth label); Schwinger depths at one size are spread horizontally
    fam = {
        "A Schwinger": [("A_schwinger_interp_h2", 4, "")]
        + [(f"A_schwinger_q6_d{d}_interp_h2", 6 + dx, f"L={d}") for d, dx in ((3, -0.5), (6, 0), (9, 0.5))]
        + [(f"A_schwinger_q8_d{d}_interp_h2_e100", 8 + dx, f"L={d}") for d, dx in ((12, -0.3), (18, 0.3))],
        "D SU(2)": [("D_su2_interp_h2", 4, ""), ("D_su2_q6_interp_h2", 6, ""), ("D_su2_q8_interp_h2", 8, "")],
        "E neutrinos": [("E_neutrino_interp_h2", 4, ""), ("E_neutrino_q6_interp_h2", 6, "")],
    }
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), sharey=True)
    for ax, (name, keys) in zip(axes, fam.items()):
        for key, x, lab in keys:
            if key not in S:
                continue
            for f, off, mk in (("classical", -0.1, "o"), ("warm", 0.1, "s")):
                c = S[key]["comparisons"].get(f)
                if not c:
                    continue
                col = C["classical"] if f == "classical" else C["warm"]
                ax.plot([x + off] * 2, c["ci"], color=col, lw=0.9)
                ax.plot(x + off, c["mean_d"], mk, color=col, ms=4, label=f"vs best {f}" if key == keys[0][0] else None)
            if lab:
                ax.text(x, -0.95, lab, fontsize=5, ha="center", color="0.35")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(name, fontsize=7)
        ax.set_xlabel("qubits")
        ax.set_xticks([4, 6, 8])
    axes[0].set_ylabel("mean log10 iteration ratio\n(<0: Q-MAML faster)")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_r2_size.pdf"))
    plt.close(fig)


def classification():
    files = sorted(glob.glob(os.path.join(R, "classification_v2", "*_report.json")))
    if not files:
        return
    panels = []
    for f in files:
        d = json.load(open(f))
        ds = os.path.basename(f).split("_q")[0]
        for nq, v in d.items():
            panels.append((f"{'HIGGS' if ds == 'higgs' else 'quark-gluon'}, {nq} qubits", v))
    arms = [
        "qmaml",
        "joint_zero",
        "joint_uniform",
        "joint_gaussian",
        "fomaml_5",
        "fomaml_20",
        "reptile_legacy",
        "no_pqc",
        "logreg_pool",
        "mlp_pool",
        "logreg_fewshot",
        "logreg_full",
        "mlp_full",
    ]
    lab = {
        "qmaml": "Q-MAML",
        "joint_zero": "joint (zero)",
        "joint_uniform": "joint (unif.)",
        "joint_gaussian": "joint (gauss.)",
        "fomaml_5": "FO-MAML K=5",
        "fomaml_20": "FO-MAML K=20",
        "reptile_legacy": "Reptile (round 1)",
        "no_pqc": "no circuit",
        "logreg_pool": "LogReg (pool)",
        "mlp_pool": "MLP (pool)",
        "logreg_fewshot": "LogReg (8-shot)",
        "logreg_full": "LogReg (50k)",
        "mlp_full": "MLP (50k)",
    }
    col = lambda a: (
        C["qmaml"]
        if a == "qmaml"
        else (
            C["classical"]
            if a.startswith(("joint", "fomaml", "reptile"))
            else ("#E69F00" if a == "no_pqc" else C["grey"])
        )
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(1.75 * len(panels) + 0.6, 2.6), sharey=True, squeeze=False)
    for ax, (title, v) in zip(axes[0], panels):
        seeds = v["seeds"]
        for i, a in enumerate(arms):
            acc = np.array([s[a]["acc"] for s in seeds.values() if a in s])
            if not len(acc):
                continue
            ax.barh(i, acc.mean(), color=col(a), alpha=0.75, height=0.7, lw=0)
            ax.errorbar(acc.mean(), i, xerr=1.96 * acc.std(ddof=1) / np.sqrt(len(acc)), color="k", lw=0.6, capsize=1.5)
        ax.axvline(0.5, color="k", lw=0.4, ls=":")
        ax.set_title(f"{title} (n={len(seeds)})", fontsize=6.5)
        ax.set_xlim(0.45, 0.8)
        ax.set_xlabel("test accuracy", fontsize=6.5)
    axes[0][0].set_yticks(range(len(arms)))
    axes[0][0].set_yticklabels([lab[a] for a in arms], fontsize=6)
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_r2_class.pdf"))
    plt.close(fig)


def bp():
    panels = []
    for name, f in (("HIGGS cross-entropy", "higgs.json"), ("Heisenberg XYZ energy", "heisenberg.json")):
        p = os.path.join(R, "barren_plateau_v2", f)
        if os.path.exists(p):
            panels.append((name, json.load(open(p))))
    if not panels:
        return
    mk = {"qmaml": "D", "uniform": "v", "gaussian": "o", "random": "s"}
    cc = {"qmaml": C["qmaml"], "uniform": "#56B4E9", "gaussian": "#CC79A7", "random": "k"}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 1.0, 2.2), squeeze=False)
    for ax, (name, d) in zip(axes[0], panels):
        depths = sorted({k.split("_")[0] for k in d})
        for dep, ls in zip(depths, ["-", "--"]):
            ks = sorted([k for k in d if k.startswith(dep + "_")], key=lambda k: int(k.split("_q")[1]))
            qs = [int(k.split("_q")[1]) for k in ks]
            for s in mk:
                ax.plot(
                    qs,
                    [d[k][s]["median_var_live"] for k in ks],
                    ls=ls,
                    marker=mk[s],
                    ms=3,
                    color=cc[s],
                    lw=0.9,
                    label=f"{s}, L={dep[1:]}",
                )
        ax.set_yscale("log")
        ax.set_xlabel("qubits")
        ax.set_title(name, fontsize=7)
        ax.set_ylabel(r"median over live $k$ of Var$[\partial C/\partial\theta_k]$")
    axes[0][-1].legend(fontsize=5.4, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_r2_bp.pdf"))
    plt.close(fig)


def replication():
    items = [
        ("heisenberg_q12_L20.json", "Heisenberg XYZ, 12 qubits, L=20"),
        ("molecule_LiH_q10_L7.json", "LiH, 10 qubits, 7 SEL layers"),
    ]
    items = [(os.path.join(R, "paper_replication", f), t) for f, t in items]
    items = [(f, t) for f, t in items if os.path.exists(f)]
    if not items:
        return
    cc = {"qmaml": C["qmaml"], "zero": "#7F7F7F", "pi": "#009E73", "uniform": "#0072B2", "gaussian": "k"}
    lsd = {"qmaml": "-", "zero": "-", "pi": ":", "uniform": "--", "gaussian": "-."}
    fig, axes = plt.subplots(1, len(items), figsize=(3.3 * len(items), 2.2), squeeze=False)
    for ax, (f, title) in zip(axes[0], items):
        d = json.load(open(f))
        E0 = np.array(d["E0"])
        for s, curves in d["curves"].items():
            if not curves:
                continue
            g = np.log(np.abs(np.array(curves) - E0[: len(curves), None]))  # natural log, as the paper's axis label
            m, sd = g.mean(0), g.std(0)
            ax.plot(m, color=cc[s], ls=lsd[s], lw=1.0, label=s)
            ax.fill_between(range(len(m)), m - sd / 2, m + sd / 2, color=cc[s], alpha=0.12, lw=0)
        ax.set_title(f"{title} ({len(d['curves']['qmaml'])} tasks)", fontsize=7)
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\ln|E_{gs}-\langle H\rangle|$")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_r2_replication.pdf"))
    plt.close(fig)


TRACK_MARK = dict(zip(ORDER, "osDv^<>pPhX"))
TRACK_COL = dict(
    zip(ORDER, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000", "#999999",
                "#882255", "#117733", "#AA4499"])
)  # fmt: skip


def interpolator(S, split="interp"):
    """Initial relative error of Q-MAML against that of the track's best warm start, one point per held-out task."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    frac = []
    for k in ORDER:
        key = f"{k}_{split}_h2"
        if key not in S or "warm" not in S[key]["comparisons"]:
            continue
        opp = S[key]["comparisons"]["warm"]["opponent"]
        q, w = [], []
        for r in load_runs(key):
            for row in r["test"]:
                sc = max(row["scale"], 1e-12)
                q.append(max(abs(row["runs"]["qmaml"]["E"][0] - row["E0"]) / sc, 1e-9))
                w.append(max(abs(row["runs"][opp]["E"][0] - row["E0"]) / sc, 1e-9))
        q, w = np.array(q), np.array(w)
        a.scatter(q, w, s=9, marker=TRACK_MARK[k], facecolor="none", edgecolor=TRACK_COL[k], lw=0.6,
                  label=LABEL[k])  # fmt: skip
        frac.append((k, float((w < q).mean()), len(q)))
    lim = [1e-9, 3]
    a.plot(lim, lim, color="k", lw=0.6)
    a.axhline(TAU, color=C["grey"], ls=":", lw=0.6)
    a.axvline(TAU, color=C["grey"], ls=":", lw=0.6)
    a.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Q-MAML initial relative error",
          ylabel="best warm start, initial relative error")  # fmt: skip
    a.text(2e-9, 1e-1, "warm start\nstarts higher", fontsize=6, va="top")
    a.text(2e-3, 3e-9, "warm start starts lower", fontsize=6)
    a.legend(fontsize=5.5, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2, columnspacing=0.5)
    ks, fs, ns = zip(*frac)
    y = np.arange(len(ks))[::-1]
    b.barh(y, fs, color=[TRACK_COL[k] for k in ks], height=0.65)
    b.axvline(0.5, color="k", lw=0.6, ls="--")
    b.set(yticks=y, yticklabels=[LABEL[k] for k in ks], xlim=(0, 1),
          xlabel="fraction of held-out tasks where\nthe warm start begins closer")  # fmt: skip
    for yi, f, n in zip(y, fs, ns):
        inside = f > 0.75
        b.text(f - 0.02 if inside else f + 0.02, yi, f"{f:.0%} (n={n})", va="center", fontsize=5.5,
               ha="right" if inside else "left", color="white" if inside else "black")  # fmt: skip
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_interpolator_{split}.pdf"))
    plt.close(fig)


def cross_platform():
    """Per-seed and pooled comparison of Q-MAML between two numerical environments (results/cross_platform)."""
    path = os.path.join(R, "cross_platform", "summary.json")
    if not os.path.exists(path):
        return
    X = json.load(open(path))
    keys = [k for k in ORDER if f"{k}_interp_h2" in X]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.42 * len(keys) + 0.9), sharey=True)
    for ax, fam, title in zip(axes, ("classical", "warm"), ("vs best classical", "vs best warm start")):
        for i, k in enumerate(keys):
            c = X[f"{k}_interp_h2"]["comparisons"][fam]
            yi = len(keys) - 1 - i
            for j, (mk, col) in enumerate((("o", C["classical"]), ("s", C["qmaml"]))):
                off = 0.14 if j == 0 else -0.14
                ax.scatter(c["per_seed_mean_d"][j], np.full(len(c["per_seed_mean_d"][j]), yi + off), s=7,
                           marker=mk, facecolor="none", edgecolor=col, lw=0.5, alpha=0.8)  # fmt: skip
                lo, hi = c["ci"][j]
                ax.plot([lo, hi], [yi + off] * 2, color=col, lw=1.2)
                ax.plot(c["mean_d"][j], yi + off, marker=mk, color=col, ms=4,
                        label=("environment 1 (stored)" if j == 0 else "environment 2 (rerun)") if i == 0 else None)  # fmt: skip
                ax.text(1.02, yi + off, c["verdict"][j][0].upper(), transform=ax.get_yaxis_transform(), fontsize=6,
                        va="center", color=col)  # fmt: skip
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("$\\Delta$ = log$_{10}$ iteration ratio (Q-MAML / opponent)")
    axes[0].set_yticks(range(len(keys))[::-1], [LABEL[k] for k in keys])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_cross_platform.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    S = summary()
    for split in ("interp", "extrap"):
        if any(k.endswith(f"_{split}_h2") for k in S):
            scoreboard(S, split)
            trajectories(S, split)
    grad0(S)
    size_trend(S)
    classification()
    bp()
    replication()
    interpolator(S)
    cross_platform()
    print("figures written to", OUT)
