# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-11-19

### Added
- LHCO initialise-then-optimise experiment (`qmaml_hep.experiments.lhco_init`, `results/lhco_init/`): the
  92-weight per-mass classifier is optimised at each test mass from four starts (Learner output, nearest mass,
  shared θ, fresh) with identical batches and step sizes tuned on validation masses. Score: gap log10(R/R_BDT)
  after 0, 25, 100 and 400 steps, Learner minus each start, hierarchical bootstrap and Holm over 12 comparisons.
  Report runs: 5 seeds × 4 starts, full 400-step schedule, no early stopping, on one AVX-512 Intel CPU. The
  Learner start is ahead of every other start at every budget (all 12 Holm-significant), with a shrinking lead
  (over nearest: 0.39 decades at step 0, 0.13 at step 400).
- `QMAML_DEVICE=cuda` option and a Colab notebook (`notebooks/lhco_init_colab.ipynb`); the CPU path is unchanged.
- `scripts/heisenberg_sector_floor.py` (`results/paper_replication/heisenberg_sector_floor.txt`): the
  stated-settings 12-qubit Heisenberg replication conserves ∏Z and ∏X, which bounds the energy 1.6–6% above E0 on
  10 of 16 test tasks.
- Compact (upright) variants of the design, physics and verdict tables for the LNCS layout.

### Changed
- The LHCO zero-shot Learner arm is labelled "LCC" (Learner-conditioned classifier) in figures and tables; result
  keys stay `qmaml`.
- Main-text figures are drawn at the LNCS text width (4.8 in), so figure text prints at 7–8 pt; the LHCO figure
  shares one row axis between its two panels.

## [0.4.0] - 2026-10-28

### Added
- LHCO (m_X, m_Y) routes study (`qmaml_hep.classification.lhco`, `qmaml_hep.experiments.lhco_routes`,
  `results/lhco_routes/`, `fig_lhco_routes`, `tab_lhco`): where should the task descriptor enter a variational
  classifier? Nine arms on a 6-qubit circuit, three training-grid spacings, learning rates tuned on validation
  points, five seeds, hierarchical bootstrap and Holm. Q-MAML beats nearest-mass transfer and masses as linear
  inputs on the 100 and 150 GeV grids, ties an encoding hypernetwork, and loses about 2× to feeding the masses in
  with the features (`param_deep`) and to the classical parameterised MLP. On the 200 GeV grid most comparisons tie.
- Few-shot stage (`fstune`, `fewshot`, `select-fs`, `score-fewshot`): every arm adapted on K = 16 and 64 events
  at each test point (Q-MAML adapts only θ, Algorithm 2), step size tuned on validation points. Adaptation changes
  no verdict against nearest, `param_deep` or the classical network on the 100 and 150 GeV grids; only arms
  without the masses gain much.
- `windows` check: every route's score peaks at the signal's jet mass (median error 5-12 GeV); the better routes cut
  narrower windows.
- `notes/lhco_feasibility/`: the feasibility checks and pilot that motivated the study.

## [0.3.0] - 2026-10-13

### Added
- Cross-platform reproducibility study (`qmaml_hep.experiments.cross_platform`, `results/cross_platform/`,
  `fig_cross_platform`): Q-MAML rerun in a second numerical environment for six systems × five seeds. 11 of 12
  verdicts are unchanged, one flips (LMG vs warm start: tie → loss), and per-seed results move by up to 10×.
  Changing only the thread count also changes Q-MAML's trajectories, while every other initializer reproduces
  bit for bit.
- Figure showing the Learner as an interpolator: Q-MAML's initial error against the best warm start's, per
  held-out task (`fig_interpolator_interp`).
- Property-based tests (Hypothesis): state normalization, energies within the exact spectrum, 2π periodicity,
  and the parameter-shift rule against autograd.
- Symmetry tests: Gauss's law for the Z₂ HVA, particle-number conservation for UCCSD, SUSY E₀ → 0 with the
  cutoff.
- Statistical calibration tests: bootstrap interval coverage under the null, power under an effect, seed-level
  resampling, Holm against a reference implementation, and family-wise error control.
- CLI tests, and a regression test that rescoring the committed results reproduces every verdict and Holm p-value.
- `qmaml-benchmark` and `qmaml-score` console entry points.
- Documentation site (MkDocs Material, API reference), CI on Linux and macOS with Python 3.10–3.13 and coverage,
  weekly slow-test workflow, tag-driven release workflow, Dependabot, pre-commit, and issue and PR templates.

### Fixed
- `docs/reproducibility.md` attributed the cross-environment difference to MKL vs OpenBLAS; both environments
  use MKL, and the difference is the OS, PyTorch build and thread count.
- `check_fast` no longer converts tensors that require grad to Python floats (removes a torch warning).

## [0.2.0] - 2026-10-13

### Added
- First release of the round-2 benchmark: ten lattice and field-theory systems plus two calibration systems,
  exact float64/complex128 torch simulators cross-checked against PennyLane on every run, the Q-MAML Learner,
  classical and warm-start initializers, the fixed scoring rule, the few-shot classification protocol, the
  gradient-variance scan and the paper-scale replication.
- Every raw result behind the round-2 analysis, and scripts that regenerate all figures and tables from them.

[Unreleased]: https://github.com/arnavk23/qmaml-hep/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/arnavk23/qmaml-hep/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/arnavk23/qmaml-hep/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/arnavk23/qmaml-hep/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/arnavk23/qmaml-hep/releases/tag/v0.2.0
