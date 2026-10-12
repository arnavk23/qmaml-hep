# qmaml-hep

**Does a meta-learned initializer help variational quantum circuits on high-energy-physics Hamiltonians, once it
faces the right baselines?**

[![CI](https://github.com/arnavk23/qmaml-hep/actions/workflows/ci.yml/badge.svg)](https://github.com/arnavk23/qmaml-hep/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This repository benchmarks **Q-MAML** (Lee, Cho & Kim, AAAI 2025, [arXiv:2501.05906](https://arxiv.org/abs/2501.05906)).
Q-MAML trains a classical network that maps a Hamiltonian's parameters to the starting angles of a variational
circuit. The benchmark covers ten lattice and field-theory systems and two few-shot classification tasks. Every
initializer runs under one protocol, and the scoring rule is fixed in code before the results exist.

## Findings

- **Against task-independent starts** (zero, π, uniform, Gaussian), Q-MAML converges faster on the
  gauge-invariant Z₂ ansatz (significant after Holm correction) and on SU(2) and SUSY QM (uncorrected). It ties on
  every other system and loses on none.
- **Against warm starts built from the same training tasks** (the mean, nearest-neighbour, linear and kernel
  regression of stored optima), Q-MAML never wins. It loses to nearest-task transfer on the Schwinger model,
  collective neutrinos and the Yukawa model. On these 1–3-parameter families the Learner behaves as an
  interpolator of optimal parameters.
- **Wins at base size do not survive to larger systems.** SU(2) at 8 qubits is a clear loss.
- **On few-shot classification** (HIGGS, quark–gluon jets), Q-MAML is statistically indistinguishable from the same
  model without a Learner, and logistic regression on the same events is better.
- **Equal compute does not help.** Giving Q-MAML's pretraining the same compute as the warm-start collection
  changes no verdict.

These are negative results, reported as such. The useful output is the controlled comparison itself, the
protocol, and a benchmark suite where every circuit is checked against PennyLane.

<p align="center"><img src="figures/scoreboard_interp.png" width="760" alt="Per-track comparison of Q-MAML against the best classical and the best warm-start initializer"></p>

<sub>Mean of log₁₀(T<sub>Q-MAML</sub>+1) − log₁₀(T<sub>opponent</sub>+1) per held-out task, where T is the number of
iterations to reach 10⁻³ relative error. Negative means Q-MAML is faster. Bars are 95% hierarchical-bootstrap
intervals over 5 seeds × 6–9 tasks. The opponent is the best initializer of its family, chosen in hindsight.</sub>

| Track | System | Qubits | vs best classical | vs best warm start |
|---|---|:-:|---|---|
| A | Schwinger model (1+1D QED), L=2 | 4 | tie | **loss**\* (nearest) |
| B | Z₂ lattice gauge theory, penalty HEA | 8 | tie† | tie† |
| B | Z₂ lattice gauge theory, gauge-invariant HVA | 8 | **win**\* (uniform) | tie |
| C | φ⁴ scalar field | 6 | tie† | win† (mean) |
| D | SU(2) lattice gauge theory, N=2 | 4 | win (uniform) | tie |
| E | Collective neutrino oscillations, N=4 | 4 | tie | **loss**\* (nearest) |
| F | Supersymmetric QM | 4 | win (uniform) | loss (nearest) |
| G | D₈ lattice gauge theory | 7 | tie† | tie† |
| H | Light nuclei, pionless EFT (UCCSD) | 8 | tie | tie |
| I | Scalar Yukawa | 4 | tie | **loss**\* (kernel) |
| J | Lipkin–Meshkov–Glick, J=3/2 | 2 | tie | tie |

<sub>\* significant after Holm correction across all comparisons. † No initializer reaches 10⁻³ within the budget,
so the track is scored on final error and is not informative about initialization. In-distribution split. The
full table, with the extrapolation split, compute-matched variants and break-even task counts, is in
[`figures/tables`](figures/tables).</sub>

## How it works

```
Algorithm 1 (pretrain, once per track)          Algorithm 2 (adapt, per held-out task; identical for every initializer)
  x = task parameters                             θ ← initializer(x)        Q-MAML | zero | π | uniform | Gaussian |
  θ₀ = 0.3 · Learner_W(x)                                                   ws_mean | ws_nearest | ws_linear | ws_kernel
  E = ⟨ψ(θ₀)|H(x)|ψ(θ₀)⟩                         repeat B times:  θ ← Adam step on E(θ)
  W ← Adam step on ∂E/∂W  (through the circuit)
```

- **Exact and double precision.** All energies are exact statevector expectation values in float64/complex128, and
  cost is counted in optimizer iterations. The circuits run on pure-torch simulators. Each run first checks them
  against the PennyLane circuit (energy and gradient) and aborts on disagreement.
- **Same budget for everyone.** Warm starts are collected by solving each training task with the same budget, in
  continuation order along a nearest-neighbour chain. Their cost is charged alongside Q-MAML's pretraining.
- **Fixed scoring.** Scoring lives in [`src/qmaml_hep/analysis.py`](src/qmaml_hep/analysis.py): iterations to 10⁻³,
  a hierarchical bootstrap over seeds and then tasks, and Holm correction. Initializers sitting at an exact
  stationary point are excluded as opponents.
- **Documented data path.** [`docs/gradient_flow.md`](docs/gradient_flow.md) traces every tensor's dtype from the
  Learner to the energy, including the float32 ↔ float64 boundaries in the classification model. Tests assert each
  step.

## Quickstart

```bash
git clone https://github.com/arnavk23/qmaml-hep && cd qmaml-hep
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plots]"

make test                      # fast suite, 2–4 min: physics checks, simulators vs PennyLane, gradient flow, scoring
python -m qmaml_hep.benchmark --track J_lmg --seeds 0      # one track, one seed, about 3 min on a laptop CPU
python -m qmaml_hep.analysis                                # rescore everything in results/unified
python scripts/make_figures.py && python scripts/make_tables.py
```

## Reproducing the results

| Experiment | Command | Output |
|---|---|---|
| Ten-track benchmark, in-distribution | `python -m qmaml_hep.benchmark --track <A_schwinger … J_lmg> --seeds 0,1,2,3,4` | `results/unified/` |
| Extrapolation split | `… --split extrap --seeds 0,1,2` | `results/unified/` |
| Compute-matched Q-MAML | `… --epochs <B> --only qmaml` | `results/unified/*_e<B>_*` |
| Larger systems | `… --track D_su2 --qubits 8`, `--track A_schwinger --qubits 8 --depth 18`, … | `results/unified/` |
| Scoring | `python -m qmaml_hep.analysis` | `results/unified_summary.json` |
| Classification | `python -m qmaml_hep.classification.protocol --dataset higgs --stage tune`, then `--stage report` | `results/classification_v2/` |
| Gradient-variance scan | `python -m qmaml_hep.experiments.barren_plateau higgs` (or `heisenberg`) | `results/barren_plateau_v2/` |
| Paper-scale replication | `python -m qmaml_hep.experiments.paper_replication heisenberg` (or `molecule`) | `results/paper_replication/` |

Track keys are listed in [`src/qmaml_hep/tracks.py`](src/qmaml_hep/tracks.py). Datasets download on first use into
`data/`: a streamed subsample of UCI HIGGS, and the Komiske–Metodiev–Thaler quark–gluon jets from Zenodo. Every
raw result file behind the paper is committed, and the generated tables match the manuscript's exactly.

**Reproducibility caveat.** Q-MAML's pretraining is chaotic. The classical and warm-start runs reproduce exactly
across machines, but Learner pretraining amplifies rounding differences between BLAS builds. A rerun on different
hardware therefore gives different per-seed Q-MAML numbers. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Repository layout

```
src/qmaml_hep/
  learner.py            the Q-MAML Learner (paper architecture, float64)
  initializers.py       zero / π / uniform / Gaussian
  simulators.py         exact torch statevector simulators (SEL, HVA, UCCSD, Ising brick, classifier circuit)
  tracks.py             one adapter per system: task space, descriptor, Hamiltonian, circuit, budgets
  benchmark.py          Algorithms 1 and 2, warm starts, splits, the run harness and CLI
  analysis.py           the scoring rule: iterations to threshold, hierarchical bootstrap, Holm
  systems/              Hamiltonians and reference circuits: schwinger, z2lgt, scalar, su2lgt, neutrino,
                        susyqm, d8lgt, nuclear, yukawa, lmg, heisenberg, molecule
  classification/       HIGGS and quark–gluon loaders, few-shot tasks, the classification protocol
  experiments/          gradient-variance scan, paper-scale replication
tests/                  physics, simulators vs PennyLane, gradient flow and dtypes, warm starts, scoring
scripts/                figures and LaTeX tables, generated from results/ only
results/                raw per-seed trajectories and summaries (see results/README.md)
docs/                   gradient flow and dtypes, reproducibility
```

## Citation

If you use this benchmark, please cite it via [`CITATION.cff`](CITATION.cff), and cite the method:

> J. Lee, J. Cho, S. Kim. *Q-MAML: Quantum Model-Agnostic Meta-Learning for Variational Quantum Algorithms.*
> AAAI 2025. arXiv:2501.05906.

## Acknowledgements

Developed as a volunteer contribution to the ML4SCI Q-MAML project, with review from Junyong Lee (Yonsei
University). It follows on from the project's GSoC 2025 codebase by Arnav Singhal. All code here is new; the
benchmark, simulators and analysis were written from scratch.
