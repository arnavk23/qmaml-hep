# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- LHCO (m_X, m_Y) routes study (`qmaml_hep.classification.lhco`, `qmaml_hep.experiments.lhco_routes`,
  `results/lhco_routes/`, `fig_lhco_routes`, `tab_lhco`): where should the task descriptor enter a variational
  classifier? Nine arms on a 6-qubit circuit, three training-grid spacings, learning rates tuned on validation
  points, five seeds, hierarchical bootstrap and Holm. Q-MAML beats nearest-mass transfer and masses as linear
  inputs on the 100 and 150 GeV grids, ties an encoding hypernetwork, and loses about 2× to feeding the masses in
  with the features (`param_deep`) and to the classical parameterised MLP. On the 200 GeV grid most comparisons tie.
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

[0.3.0]: https://github.com/arnavk23/qmaml-hep/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/arnavk23/qmaml-hep/releases/tag/v0.2.0
