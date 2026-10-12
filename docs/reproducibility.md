# Reproducibility

## What is deterministic

Everything is seeded: task draws, splits, Learner initialization, classical initializers and the warm-start
chain. On one machine and one software stack, a rerun reproduces every stored trajectory bit for bit. This
package was checked against the original research scripts on the LMG track: all 9 initializers × 6 held-out
tasks were identical, with a maximum |ΔE| of 0.

Across machines, the following reproduce exactly:

- task sampling, exact ground energies, splits
- all four classical initializers (zero, π, uniform, Gaussian)
- all four warm starts (their training optima come from deterministic Adam runs on fixed starts)

`tests/test_reproducibility.py` (marked `slow`) reruns LMG seed 0 and checks these against the committed result
file to 1e-8.

## What is not: Q-MAML pretraining

Learner pretraining (Algorithm 1) is a chaotic optimization. Adam with gradient clipping amplifies floating-point
rounding differences, about 1e-16 per operation, coming from different BLAS builds (MKL on Windows, OpenBLAS on
Linux) or different torch versions. Measured on LMG, interpolation split, seed 0:

| | stored run (Windows, torch 2.13) | rerun (Linux, torch 2.12) |
|---|---|---|
| pretraining loss, epochs 1–2 | −1.7410, −1.4269 | −1.7410, −1.4269 |
| Learner gradient norm, epoch 2 | 11.195577 | 11.195480 |
| pretraining loss, epoch 5 | −1.8383 | −1.6559 |
| Q-MAML iterations to 1e-3, six test tasks | 13, 28, 11, 23, 21, 11 | 48, 49, 51, 32, 52, 61 |
| π initializer, same six tasks | 8, 8, 2, 32, 26, 11 | 8, 8, 2, 32, 26, 11 |

The two runs diverge after the first two epochs. Per-seed Q-MAML results are therefore one draw from a
distribution that includes the numerical environment, not a fixed number. This is one reason every verdict
pools seeds with a hierarchical bootstrap rather than relying on any single run. A rerun on new hardware should
reproduce the verdicts statistically, not the per-seed trajectories.

To get bitwise reproduction of Q-MAML numbers, rerun on the same stack. Record it with
`python -c "import torch, numpy; print(torch.__version__, torch.__config__.show(), numpy.__version__)"`.

## Precision

All circuit states are complex128 and all energies float64. The PennyLane reference circuits on
`lightning.qubit` evaluate `expval(H)` in single precision (relative error about 1e-7), so they agree with the
torch simulators only to about 1e-8. That is why the simulators, not lightning, are used for every run, and why
round-1 results below about 1e-7 relative error carried no information.
