"""Round-2 tables, generated from code and saved results (no hand-transcribed numbers):
tab_design.tex    per-track design: qubits, Hilbert dim, ansatz, depth, #angles P, descriptor, ranges,
                  n_train/n_test, budget B, learning rates (read from the qmaml_hep.tracks adapters)
tab_physics.tex   per-track physics question answered by the ground state, and the pre-use verification
tab_verdicts.tex  round-2 verdicts (iterations to 1e-3, hierarchical-bootstrap CI, Holm) from unified_summary.json
tab_class.tex     classification (HIGGS 4/6/8, quark-gluon 6) from results/classification_v2
tab_lhco.tex      LHCO (m_X, m_Y) routes: Q-MAML minus each arm per grid spacing, and each arm's gap to a per-point
                  BDT, from results/lhco_routes
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
R = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures", "tables")
sys.path.insert(0, HERE)

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
    "A_schwinger": r"Schwinger model, $L{=}2$",
    "B_z2": r"$\mathbb{Z}_2$ LGT, penalty (HEA)",
    "B_z2_hva": r"$\mathbb{Z}_2$ LGT, gauge-inv.\ (HVA)",
    "C_scalar": r"$\phi^4$ scalar field",
    "D_su2": r"SU(2) LGT, $N{=}2$",
    "E_neutrino": r"Collective neutrinos, $N{=}4$",
    "F_susyqm": "SUSY QM",
    "G_d8": r"$\mathbb{D}_8$ LGT",
    "H_nuclear": r"Light nuclei (pionless EFT)",
    "I_yukawa": "Scalar Yukawa",
    "J_lmg": r"LMG, $J{=}3/2$",
}
ANSATZ = {
    "A_schwinger": "HEA",
    "B_z2": "HEA",
    "B_z2_hva": "HVA",
    "C_scalar": "HEA",
    "D_su2": "HEA",
    "E_neutrino": "HEA",
    "F_susyqm": "HEA",
    "G_d8": "HEA",
    "H_nuclear": "UCCSD",
    "I_yukawa": "HEA",
    "J_lmg": "HEA",
}
RANGES = {
    "A_schwinger": r"$m_0\in[0,1.5]$, $g\in[0.3,2]$",
    "B_z2": r"$J,m,\mu\in[0.1,2]$; $V{=}10$",
    "B_z2_hva": r"$J,m,\mu\in[0.1,2]$; $V{=}10$",
    "C_scalar": r"$m_0^2\in[-2.5,1]$, $\lambda_0\in[0.5,1.5]$",
    "D_su2": r"$\tilde m\in[0.1,2]$, $x\in[0.2,5]$",
    "E_neutrino": r"$\theta\in[0.05,0.7]$, $\mu\in[0.1,3]$",
    "F_susyqm": r"$W\in\{$HO,AHO,DW$\}$; $m,g,\mu\in[0.5,1.5]$",
    "G_d8": r"$M,\lambda_E,\epsilon\in[0.2,2]$",
    "H_nuclear": r"$^2$H/$^3$H/$^3$He; $t\in[0.5,2]$, $C\in[-2,-0.2]$, $D\in[-1,1]$",
    "I_yukawa": r"$\eta\in[0.2,1]$, $m\in[1,2]$",
    "J_lmg": r"$\epsilon\in[0.5,2]$, $V\in[-1.5,1.5]$",
}
PHYSICS = {
    "A_schwinger": (
        "Vacuum of 1+1D QED (lattice Schwinger model, gauge field eliminated); its energy vs.\\ $(m_0,g)$ "
        "tracks the chiral condensate/confinement regime.",
        "Hermiticity; Coulomb term checked against a brute-force dense construction ($n\\le8$).",
    ),
    "B_z2": (
        "Vacuum of a single-plaquette $\\mathbb{Z}_2$ gauge theory with matter in the physical (charge-free) "
        "Gauss sector, enforced by an energy penalty.",
        "$G_l^2=\\mathbb{1}$ and $[H,G_l]=0$ for all four sites; exact ground state lies in the $q_l{=}{+}1$ sector.",
    ),
    "B_z2_hva": (
        "Same Hamiltonian; ansatz is gauge-invariant by construction (Trotterized physical terms from the "
        "Gauss vacuum).",
        "As above; every ansatz generator commutes with every $G_l$.",
    ),
    "C_scalar": (
        "Ground state of 2-site $\\phi^4$ theory across the symmetric/broken-symmetry crossover.",
        "$\\Phi,\\Pi$ Hermitian; free-field spectrum matches $\\mu(n+\\frac12)$ to $<5\\%$ for the lowest levels.",
    ),
    "D_su2": (
        "Vacuum of SU(2) LGT with staggered fermions (gauge field integrated out), the reference state for "
        "hadron masses.",
        "Hermiticity at several $(\\tilde m,x)$; 4-body colour-exchange term checked against "
        "an independent dense $\\sigma^\\pm$ construction ($10^{-16}$).",
    ),
    "E_neutrino": (
        "Adiabatic (co-rotating) ground state of the collective two-flavour neutrino Hamiltonian; "
        "reformulation of a real-time problem.",
        "Hermiticity; $\\mu{=}0$ spectrum equals the analytic single-particle one; $\\theta{=}\\mu{=}0$ ground "
        "state is $|1\\ldots1\\rangle$.",
    ),
    "F_susyqm": (
        "Whether SUSY is broken: $E_0{=}0$ (unbroken, HO/AHO) vs.\\ $E_0{>}0$ (broken, DW).",
        "Hermiticity; HO spectrum; DW potential has two minima (sign convention checked on a grid).",
    ),
    "G_d8": (
        "Ground state of a non-Abelian $\\mathbb{D}_8$ gauge theory with fermionic matter (2 sites, 1 link).",
        "Representation check $D(g_1)D(g_2)=D(g_1g_2)$ over all 64 pairs (caught a real bug); unitarity; "
        "$j{=}2$ projector idempotent of rank 4; JW anticommutation.",
    ),
    "H_nuclear": (
        "Binding energies of $^2$H, $^3$H, $^3$He in lattice pionless EFT.",
        "Hermiticity; $[H,N_p]=[H,N_n]=0$; bound (negative-energy) state exists in each sector.",
    ),
    "I_yukawa": (
        "Dressed fermion-vacuum ground state under scalar Yukawa coupling; reformulation of a quench problem.",
        "Vacuum-sector energy vs.\\ closed form $-(\\eta/2)^2/m$ ($<1\\%$ in range); ground state "
        "has zero fermion occupation.",
    ),
    "J_lmg": (
        "Collective ground state of the LMG model (nuclear structure benchmark, not HEP).",
        "Hermiticity; su(2) algebra of the generators; $V{=}0$ limit $E_0=-\\epsilon J$.",
    ),
}


def design():
    import qmaml_hep.tracks as U

    rows = []
    for k in TRACK_ORDER:
        tr = U.TRACKS[k]()
        c = tr["cfg"]
        tasks = tr["sample"](4, 0)
        d = len(tr["desc"](tasks[0]))
        groups = sorted({tr["group"](t) for t in tr["sample"](60, 0)})
        P = "/".join(str(int(np.prod(tr["shape"](g)))) for g in groups)
        rows.append(
            f"{k[0]} & {NAME[k]} & {tr['qubits']} & $2^{{{tr['qubits']}}}$ & {ANSATZ[k]} & {tr['depth']} & {P} & "
            f"{d} & {RANGES[k]} & {c['n_train']}/{c['n_test']} & {c['adapt_iters']} & {c['adapt_lr']} & "
            f"{c['pretrain_epochs']}/{c['pretrain_lr']} \\\\"
        )
    body = "\n".join(rows)
    tex = (
        r"\begin{tabular}{@{}llcclccclcccc@{}}"
        "\n\\toprule\n"
        r"Tr. & System & $n$ & $\dim\mathcal{H}$ & Ansatz & $L$ & $P$ & $d$ & Task range (uniform) & train/test & $B$ & $\eta'$ & epochs/$\eta$ \\"
        "\n\\midrule\n" + body + "\n\\botrule\n\\end{tabular}\n"
    )
    open(os.path.join(OUT, "tab_design.tex"), "w", encoding="utf-8").write(tex)
    rows = [f"{k[0]} & {NAME[k]} & {PHYSICS[k][0]} & {PHYSICS[k][1]} \\\\" for k in TRACK_ORDER]
    tex = (
        r"\begin{tabular}{@{}llp{0.34\textwidth}p{0.35\textwidth}@{}}"
        "\n\\toprule\n"
        r"Tr. & System & Question answered by the ground state & Verification before use \\"
        "\n\\midrule\n" + "\n".join(rows) + "\n\\botrule\n\\end{tabular}\n"
    )
    open(os.path.join(OUT, "tab_physics.tex"), "w", encoding="utf-8").write(tex)


LHCO_ARMS = {  # arm: (label, where the masses enter)
    "qmaml": ("Q-MAML", r"$\theta=h_W(m)$"),
    "joint": ("joint", "nowhere"),
    "param": ("param", r"linear encoding of $[x,m]$"),
    "nearest": ("nearest", "nearest training mass"),
    "hyper_enc": ("hyper-enc", r"MLP$(m)$ sets the encoding"),
    "qmaml_deep": ("Q-MAML, deep", r"$\theta=h_W(m)$, MLP front end"),
    "joint_deep": ("joint, deep", "nowhere, MLP front end"),
    "param_deep": ("param-deep", r"MLP$([x,m])\to$ angles"),
    "classical": ("classical MLP", r"MLP$([x,m])$, no circuit"),
}


def lhco():
    """tab_lhco.tex: the LHCO routes study, Q-MAML against every arm at each grid spacing."""
    import glob
    import json

    D = os.path.join(R, "lhco_routes")
    if not os.path.exists(os.path.join(D, "selected_lr.json")):
        return
    lrs = json.load(open(os.path.join(D, "selected_lr.json")))
    S = {sp: json.load(open(f)) for sp in (100, 150, 200) if os.path.exists(f := os.path.join(D, f"summary_sp{sp}_qmaml.json"))}  # fmt: skip
    rows = []
    for a, (label, route) in LHCO_ARMS.items():
        f = sorted(glob.glob(os.path.join(D, f"report_sp100_{a}_lr{lrs[a]:g}_s*.json")))[0]
        n = json.load(open(f)).get("n_params")
        cells = []
        for s in S.values():
            r = next((r for r in s["comparisons"] if r["opponent"] == a), None)
            if r is None:
                cells.append("--")
                continue
            v = {"win": r"\textbf{W}", "loss": r"\textbf{L}", "tie": "T"}[r["verdict"]]
            cells.append(f"${r['mean']:+.2f}$\\,{v}")  # the intervals are in fig_lhco_routes
        gaps = " / ".join(f"${s['gap_to_oracle_bdt'][a]['mean']:+.2f}$" for s in S.values())
        n = n if n else "92/mass"  # nearest: one joint-architecture circuit per training mass
        rows.append(f"{label} & {route} & {n} & {lrs[a]:g} & " + " & ".join(cells) + f" & {gaps} \\\\")
    ns = [f"{sp} GeV" for sp in S]
    tex = (
        r"\begin{tabular}{@{}llrr" + "r" * len(S) + "c@{}}\n\\toprule\n"
        r"Arm & Masses enter via & Params & $\eta$ & "
        + " & ".join(ns)
        + r" & gap to BDT \\"
        + "\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\botrule\n\\end{tabular}\n"
    )
    open(os.path.join(OUT, "tab_lhco.tex"), "w", encoding="utf-8").write(tex)


from result_tables import classification, compute_table, verdicts  # noqa: E402

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    design()
    lhco()
    if os.path.exists(os.path.join(R, "unified_summary.json")):
        verdicts("interp")
        verdicts("extrap")
        compute_table()
    classification()
