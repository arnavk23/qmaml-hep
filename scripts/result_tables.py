"""Result tables for round 2 (imported by make_round2_tables.py)."""

import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures", "tables")
TRACK_ORDER = [
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
NAME = {
    "A_schwinger": r"Schwinger, $L{=}2$",
    "B_z2": r"$\mathbb{Z}_2$ LGT, penalty (HEA)",
    "B_z2_hva": r"$\mathbb{Z}_2$ LGT, gauge-inv.\ (HVA)",
    "C_scalar": r"$\phi^4$ scalar field",
    "D_su2": r"SU(2) LGT, $N{=}2$",
    "E_neutrino": r"Neutrinos, $N{=}4$",
    "F_susyqm": "SUSY QM",
    "G_d8": r"$\mathbb{D}_8$ LGT",
    "H_nuclear": "Light nuclei",
    "I_yukawa": "Scalar Yukawa",
    "J_lmg": r"LMG, $J{=}3/2$",
}
SNAME = {
    "zero": "Zero",
    "pi": r"$\pi$",
    "uniform": "Unif.",
    "gaussian": "Gauss.",
    "gauss_vacuum": "vacuum",
    "basis_paper": "basis",
    "ws_mean": "mean",
    "ws_nearest": "nearest",
    "ws_linear": "linear",
    "ws_kernel": "kernel",
}


def fmt_e(x):
    if x is None or not np.isfinite(x):
        return "--"
    m, e = f"{x:.1e}".split("e")
    e = int(e)
    return m if e == 0 else rf"${m}\!\times\!10^{{{e}}}$"


def _summary():
    return json.load(open(os.path.join(R, "unified_summary.json")))


def _cell(c, f):
    if f not in c:
        return "--"
    x = c[f]
    mark = {"win": "W", "loss": "L", "tie": "T"}[x["verdict"]]
    star = r"$^\ast$" if x.get("p_holm", 1) < 0.05 else ""
    fb = r"$^\dagger$" if x["metric"] != "log10_iters_to_tau" else ""
    return f"{SNAME[x['opponent']]} {x['mean_d']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] {mark}{star}{fb}"


def verdicts(split="interp"):
    S = _summary()
    rows = []
    for k in TRACK_ORDER:
        key = f"{k}_{split}_h2"
        if key not in S:
            continue
        v = S[key]
        sc = v["schemes"]
        c = v["comparisons"]
        rows.append(
            f"{k[0]} & {NAME[k]} & {v['n_seeds']} & {sc['qmaml']['frac_reached']:.2f} & "
            f"{fmt_e(sc['qmaml']['geo_mean_final_err'])} & {_cell(c, 'classical')} & {_cell(c, 'warm')} & "
            f"{_cell(c, 'classical|cm')} & {_cell(c, 'warm|cm')} \\\\"
        )
    tex = (
        "\\begin{tabular}{@{}llcclllll@{}}\n\\toprule\n"
        r"Tr. & System & $n_s$ & reach & $\varepsilon_B$ & vs best classical & vs best warm start & "
        r"matched: vs classical & matched: vs warm \\"
        "\n\\midrule\n" + "\n".join(rows) + "\n\\botrule\n\\end{tabular}\n"
    )
    open(os.path.join(OUT, f"tab_verdicts_{split}.tex"), "w", encoding="utf-8").write(tex)


def compute_table():
    S = _summary()
    rows = []
    be = lambda x: "never" if x is None else f"{x:.0f}"
    for k in TRACK_ORDER:
        key = f"{k}_interp_h2"
        if key not in S:
            continue
        v = S[key]
        cp = v["compute"]
        c = v["comparisons"]
        wc = cp.get("warm_vs_classical", {})
        sav = f"{c['classical']['mean_iter_saving_per_task']:+.0f}" if "classical" in c else "--"
        wsav = f"{wc['mean_iter_saving_per_task']:+.0f}" if wc else "--"
        rows.append(
            f"{k[0]} & {NAME[k]} & {cp['pretrain_evals']} & {cp.get('pretrain_evals_cm') or '--'} & "
            f"{cp['warm_collect_evals']} & {sav} & {be(c.get('classical', {}).get('break_even_tasks'))} & "
            f"{wsav} & {be(wc.get('break_even_tasks')) if wc else '--'} & "
            f"{100 * cp['pretrain_final_loss_rel_change_last10']:.1f} \\\\"
        )
    tex = (
        "\\begin{tabular}{@{}llrrrrrrrr@{}}\n\\toprule\n"
        r"Tr. & System & pretrain & matched & warm coll. & Q-MAML saving & break-even & warm saving & break-even & "
        r"$\Delta\ell_{10}$ (\%) \\"
        "\n\\midrule\n" + "\n".join(rows) + "\n\\botrule\n\\end{tabular}\n"
    )
    open(os.path.join(OUT, "tab_compute.tex"), "w", encoding="utf-8").write(tex)


ARMS = [
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
ARM_LABEL = {
    "qmaml": "Q-MAML",
    "joint_zero": "joint, zero",
    "joint_uniform": "joint, uniform",
    "joint_gaussian": "joint, Gaussian",
    "fomaml_5": r"FO-MAML $K{=}5$",
    "fomaml_20": r"FO-MAML $K{=}20$",
    "reptile_legacy": "Reptile step (round 1)",
    "no_pqc": "no circuit",
    "logreg_pool": "LogReg, pooled",
    "mlp_pool": "MLP, pooled",
    "logreg_fewshot": "LogReg, 8-shot",
    "logreg_full": "LogReg, 50k rows",
    "mlp_full": "MLP, 50k rows",
}


def classification_data():
    from scipy import stats

    cols, data, tests = [], {}, []
    for f in sorted(glob.glob(os.path.join(R, "classification_v2", "*_report.json"))):
        d = json.load(open(f))
        ds = "HIGGS" if os.path.basename(f).startswith("higgs") else "QG"
        for nq, v in sorted(d.items()):
            col = f"{ds} {nq}q"
            cols.append(col)
            seeds = sorted(v["seeds"], key=int)
            for a in ARMS:
                data[(col, a)] = np.array([v["seeds"][s][a]["acc"] for s in seeds if a in v["seeds"][s]])
            for a in ARMS[1:]:
                x, y = data[(col, "qmaml")], data[(col, a)]
                if len(x) == len(y) and len(x) > 2:
                    tests.append((col, a, float(stats.ttest_rel(x, y).pvalue), float(np.mean(x - y))))
    ps = np.array([t[2] for t in tests])
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return cols, data, {(t[0], t[1]): (float(a), t[3]) for t, a in zip(tests, adj)}


def classification():
    cols, data, padj = classification_data()
    if not cols:
        return
    rows = []
    for a in ARMS:
        cells = []
        for col in cols:
            x = data.get((col, a))
            if x is None or not len(x):
                cells.append("--")
                continue
            ci = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            mark = ""
            if (col, a) in padj and padj[(col, a)][0] < 0.05:
                mark = r"$^{\uparrow}$" if padj[(col, a)][1] > 0 else r"$^{\downarrow}$"
            cells.append(f"{x.mean():.3f}$\\pm${ci:.3f}{mark}")
        rows.append(f"{ARM_LABEL[a]} & " + " & ".join(cells) + r" \\")
    ns = [len(data[(c, "qmaml")]) for c in cols]
    tex = (
        "\\begin{tabular}{@{}l"
        + "c" * len(cols)
        + "@{}}\n\\toprule\nArm & "
        + " & ".join(f"{c} ($n{{=}}{n}$)" for c, n in zip(cols, ns))
        + r" \\"
        + "\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\botrule\n\\end{tabular}\n"
    )
    open(os.path.join(OUT, "tab_class.tex"), "w", encoding="utf-8").write(tex)
