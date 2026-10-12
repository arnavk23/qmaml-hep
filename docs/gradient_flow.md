# Data types and gradient flow

This page traces every tensor from the classical network to the circuit's output, for both settings in the
repository, and lists the tests that check each statement. The question it answers is where a hybrid model
could silently break: a dtype downcast, a detached tensor, or a gradient that never reaches the Learner.

## 1. Hamiltonian tracks (Algorithm 1 and Algorithm 2)

```
                    float64                  float64                    complex128                  float64
descriptor x  ──►  Learner f_W  ──►  θ₀ = s·f_W(x)  ──►  |ψ(θ₀)⟩ = U(θ₀)|ψ_init⟩  ──►  E = Re⟨ψ|H|ψ⟩
 (task params)     d→256→256→P        s = 0.3            exact statevector              H dense complex128
      ▲                                                                                       │
      └──────────────────────────── ∂E/∂W by reverse-mode autograd ◄─────────────────────────┘
```

| Stage | Code | dtype | Notes |
|---|---|---|---|
| Task descriptor `x` | `tracks.py`, `desc(task)` | float64 | Raw task parameters; one-hot plus continuous for SUSY QM and nuclei |
| Learner weights `W` | `learner.py` | float64 | Module is cast with `.double()` before training |
| Initial angles `θ₀` | `benchmark.pretrain` | float64 | `θ₀ = 0.3 · f_W(x)` |
| Gate matrices, state | `simulators.py` | complex128 | Built from `θ` with `torch.cos/sin/exp`; differentiable |
| Hamiltonian | `simulators._DenseCache` | complex128 | Dense matrix, built once per task |
| Energy | `simulators._energy` | float64 | `Re(ψ† H ψ)` |

**Algorithm 1 (pretraining).** `benchmark.pretrain` back-propagates the energy straight through the circuit into
the Learner. There is no inner loop. Gradients are clipped at norm 5 and the optimizer is Adam.

**Algorithm 2 (adaptation).** `benchmark.adapt` gives every initializer the same loop: Q-MAML, the four classical
initializers and the four warm starts. `θ` is a fresh float64 `nn.Parameter` copied from `θ₀`, the Learner is no
longer involved, and Adam runs for the track's budget `B` at its learning rate.

**Why not PennyLane in the loop.** Every Hamiltonian track runs on exact torch simulators of the same circuits:
StronglyEntanglingLayers, the gauge-invariant Z₂ HVA, closed-form UCCSD excitations and the Ising brick ansatz.
At the start of every run, `tracks.check_fast` compares energy and gradient against the PennyLane QNode at a random
point and aborts the run if they differ by more than 1e-6. Two kinds of reference are involved:

- **`default.qubit` references** agree to about 1e-14.
- **`lightning.qubit` references** agree only to about 1e-8. These are the Schwinger, Z₂ HEA, SU(2), neutrino and H₂
  circuits. lightning evaluates `expval(H)` in single precision, so in these comparisons the torch simulator is
  the more accurate side.

## 2. Classification (HIGGS, quark–gluon)

Here a neural network feeds the circuit's *encoding*, so the classical and quantum parts are attached twice: at
the encoding angles and at the initial weights.

```
 features (float32)
     │
     ├─► frozen backbone ─► embed (no_grad) ─► mean over support ─► .double() ─► Learner ─► θ₀ (float64)
     │                                                                                       │
     └─► frozen backbone ─► to_angles (trainable, float32) ─► a ─► tanh(a)·π ─► RY encoding ─┤
                                                                   (cast to float64)          ▼
                                                     StronglyEntanglingLayers(θ) (complex128) ─► ⟨Z_i⟩ (float64)
                                                                                                  │
                                                     linear head (float32) ◄── cast to float32 ◄──┘
                                                          │
                                                     cross-entropy
```

| Boundary | What happens | Gradient |
|---|---|---|
| Backbone → embedding | `TabularFeatureExtractor.embed` runs under `no_grad` | None. The backbone is frozen and the task embedding is a constant input to the Learner |
| Embedding → Learner | `.double()` | Learner weights receive gradient through `θ₀` |
| `to_angles` → circuit | float32 → float64 inside `sel_expvals` | The cast is differentiable, so `to_angles` is trained (at 0.2× the main learning rate) |
| Circuit → head | float64 → float32 | The cast is differentiable, so the gradient reaches the circuit |

The only float32 parts are the feature extractor and the linear head. These are classical layers whose precision
does not limit the result. Everything between the encoding angles and the expectation values is float64 or
complex128.

## 3. Tests

| Statement | Test |
|---|---|
| The Learner has the paper's shape, d→256→256→P | `test_gradient_flow.py::test_learner_matches_paper_architecture` |
| The Hamiltonian path is float64/complex128 end to end, and every Learner parameter receives a finite, non-zero gradient | `::test_vqe_path_is_float64_end_to_end` |
| ∂E/∂W through the circuit equals a central finite difference | `::test_vqe_learner_gradient_matches_finite_differences` |
| Circuit gradients pass `torch.autograd.gradcheck` | `::test_circuit_gradients_pass_gradcheck` |
| Classification: dtypes at every boundary; gradient reaches the Learner, `to_angles` and the head, but not the frozen backbone or the shared angles | `::test_classification_path_dtypes_and_gradients` |
| The torch simulators match PennyLane for every track and state preparation | `test_simulators.py` |
