# qmaml-hep

**Does a meta-learned initializer help variational quantum circuits on high-energy-physics Hamiltonians, once it
faces the right baselines?**

Q-MAML (Lee, Cho & Kim, AAAI 2025) trains a classical network, the *Learner*, to map a Hamiltonian's parameters to
the starting angles of a variational circuit. This project benchmarks it against two families of initializers:

- **task-independent** starts: zero, π, uniform and Gaussian;
- **warm starts** built from the optimized parameters of the same training tasks: their mean, the nearest task's
  optimum, and linear or kernel regression on the task parameters.

The benchmark spans ten lattice and field-theory systems and two few-shot classification datasets. It runs under
one protocol whose scoring rule is fixed in code before any result exists.

![Per-track comparison](assets/scoreboard_interp.png)

## In one paragraph

Against task-independent starts, Q-MAML converges faster on the gauge-invariant Z₂ ansatz and, without multiple-
comparison correction, on SU(2) and SUSY QM. It ties on everything else. Against warm starts it never wins, and
transferring the nearest training task's optimum is significantly faster on three systems. The wins do not survive
to larger versions of the same systems. On HEP classification, Q-MAML is indistinguishable from the same model
without a Learner. On these 1–3-parameter families the Learner behaves as an interpolator of optimal parameters,
and a nearest-neighbour lookup is the baseline it has to beat.

## Where to go next

- [Getting started](getting-started.md): install, run a track, rescore, rebuild figures.
- [Protocol](protocol.md): initializers, splits, the metric and the verdict rule.
- [Gradient flow and dtypes](gradient_flow.md): every tensor from the Learner to the energy, with the tests that
  check each step.
- [Reproducibility](reproducibility.md): what reproduces bit for bit, and why Q-MAML pretraining does not.
