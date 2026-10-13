# Protocol

The rule below is implemented in `qmaml_hep.analysis` and was fixed before the full result set existed. It is
applied identically to every track, split and size.

## Initializers

| Family | Initializer | Definition |
|---|---|---|
| Learned | Q-MAML | θ₀ = 0.3 · Learner(x); Learner x → 256 → 256 → P, ReLU, float64; Algorithm 1, no inner loop |
| Task-independent | zero, π | All angles 0, or all π |
| | uniform | U[−0.05π, 0.05π] (reduced domain, Wang et al. 2023) |
| | Gaussian | N(0, γ²), γ² = 1/(4S(L+2)), S = 2 (Zhang et al. 2022) |
| | physics state | Gauss-law vacuum (Z₂), the source paper's basis state (SUSY QM) |
| Warm start | mean | Mean of the training optima |
| | nearest | Optimum of the nearest training task (z-scored descriptor) |
| | linear | Ridge regression of optima on the descriptor (λ = 10⁻³) |
| | kernel | RBF kernel ridge regression (length scale = median pairwise distance, λ = 10⁻³) |

Training optima are collected by solving each training task with the same budget as a held-out task. The tasks
are visited in continuation order, along a greedy nearest-neighbour chain in descriptor space, so the optima lie on
one branch and can be interpolated. That cost is charged to the warm starts.

## Adaptation

Every initializer runs the identical loop: Adam on θ for the track's budget B at its learning rate, with exact
energies and gradients.

## Splits

- **Interpolation (in-distribution):** training and held-out tasks are independent draws from the same ranges.
- **Extrapolation:** a pool of 120 tasks is sorted by the first continuous descriptor coordinate. Training tasks
  come from the lowest 60% and held-out tasks from the top 25%.

## Metric and verdict

- **Per-task error:** for a held-out task and an initializer, ε_t = |E_t − E₀| / |E₀|. SUSY QM, where E₀ = 0,
  divides by the mass m instead.
- **Metric T:** the first iteration with ε_t ≤ 10⁻³. If the threshold is never reached, T = B + 1 (censored).
- **Per-task statistic:** Δ = log₁₀(T_Q + 1) − log₁₀(T_opp + 1). If nothing reaches the threshold on a track, Δ is
  the log-ratio of final errors (marked † in tables).
- **Opponent:** the best classical and the best warm-start initializer, each chosen in hindsight. Initializers whose
  gradient vanishes at the start on every task (exact stationary points) are excluded.
- **Uncertainty:** a hierarchical bootstrap that resamples seeds, then tasks within seeds, with 10⁴ replicates. It
  gives a 95% percentile interval and a two-sided p.
- **Verdict:** win if the interval lies below 0, loss if above, tie otherwise. Holm correction runs over every
  comparison in the family.

## Compute accounting

Pretraining costs (epochs × training tasks) evaluations, and warm-start collection costs (training tasks × B).
The break-even number of held-out tasks is the one-off cost divided by the mean per-task saving.
