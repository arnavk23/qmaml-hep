# Reproducibility

## What is deterministic

Everything is seeded: task draws, splits, Learner initialization, classical initializers and the warm-start
chain. In a fixed numerical environment (same operating system, PyTorch build and thread count), a rerun
reproduces every stored trajectory bit for bit. This package was checked against the original research scripts on
the LMG track: all 9 initializers × 6 held-out tasks were identical, with a maximum |ΔE| of 0.

Across environments, the following reproduce exactly:

- task sampling, exact ground energies, splits
- all four classical initializers (zero, π, uniform, Gaussian)
- all four warm starts (their training optima come from deterministic Adam runs on fixed starts)

`tests/test_reproducibility.py` (marked `slow`) reruns LMG seed 0 and checks these against the committed result
file to 1e-8.

## What is not: Q-MAML pretraining

Learner pretraining (Algorithm 1) is a chaotic optimization. Adam with gradient clipping amplifies floating-point
rounding differences of about 1e-16 per operation until the trained Learners differ. Three environments, all on
the same laptop CPU (Intel i5-1135G7) and all using MKL:

| Environment | OS | Python | PyTorch | Threads |
|---|---|---|---|---|
| 1 (stored results) | Windows | 3.11 | 2.13.0 | default |
| 2a | Ubuntu 26.04 | 3.14 | 2.12.1 | 4 |
| 2b | Ubuntu 26.04 | 3.14 | 2.12.1 | 1 |

LMG, interpolation split, seed 0:

| | env. 1 | env. 2a | env. 2b |
|---|---|---|---|
| pretraining loss, epochs 1–2 | −1.7410, −1.4269 | −1.7410, −1.4269 | −1.7410, −1.4269 |
| pretraining loss, epoch 5 | −1.8383 | −1.6559 | −1.6559 |
| Q-MAML iterations to 1e-3, six test tasks | 13, 28, 11, 23, 21, 11 | 48, 49, 51, 32, 52, 61 | 0, 59, 58, 36, 52, 22 |
| π initializer, same six tasks | 8, 8, 2, 32, 26, 11 | 8, 8, 2, 32, 26, 11 | 8, 8, 2, 32, 26, 11 |

Environments 1 and 2 diverge after two epochs. Environments 2a and 2b agree for longer but still end with
different Learners: **changing only the number of threads changes the result.**

## Do verdicts survive? The cross-platform study

Because only Q-MAML depends on the environment, a rerun only needs to repeat Q-MAML. We did this for six systems,
five seeds each, in environment 2b (`qmaml-benchmark --track ... --only qmaml`), and rescored with the unchanged rule
(`python -m qmaml_hep.experiments.cross_platform`; results in `results/cross_platform/`).

- **11 of 12 verdicts are unchanged.** Pooling over seeds and tasks is what makes them stable.
- **One flips:** on LMG, Q-MAML against the best warm start goes from a tie (Δ = +0.08) to a loss (Δ = +0.40).
- **Per-seed results move a lot.** The median iterations to threshold on one neutrino seed go from 35 to 135, and
  from 159 to 24 on another.

A per-seed Q-MAML result is therefore one draw from a distribution that includes the numerical environment.
Single-seed or single-environment comparisons of learned initializers should not be trusted. To reproduce Q-MAML
numbers bit for bit, use the same OS, PyTorch build and thread count, and record them:

```bash
python -c "import torch, numpy, platform; print(platform.platform(), torch.__version__, torch.get_num_threads(), numpy.__version__)"
```

## Precision

All circuit states are complex128 and all energies float64. The PennyLane reference circuits on
`lightning.qubit` evaluate `expval(H)` in single precision (relative error about 1e-7), so they agree with the
torch simulators only to about 1e-8. That is why the simulators, not lightning, are used for every run, and why
round-1 results below about 1e-7 relative error carried no information.
