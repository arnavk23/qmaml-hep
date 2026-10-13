# Systems

Every system module in `qmaml_hep.systems` builds its Hamiltonian, samples tasks and computes exact ground
energies. It also ships a construction check that runs in the test suite.

| Track | System | Qubits | Ansatz | Layers | Angles | Task parameters | Train/test | Budget |
|---|---|:-:|---|:-:|:-:|---|:-:|:-:|
| A | Schwinger model, L = 2 | 4 | HEA | 3 | 36 | m₀ ∈ [0, 1.5], g ∈ [0.3, 2] | 16/6 | 300 |
| B | Z₂ LGT, energy penalty | 8 | HEA | 3 | 72 | J, m, μ ∈ [0.1, 2]; V = 10 | 16/6 | 300 |
| B | Z₂ LGT, gauge-invariant | 8 | HVA | 3 | 45 | as above | 16/6 | 300 |
| C | φ⁴ scalar field | 6 | HEA | 3 | 54 | m₀² ∈ [−2.5, 1], λ₀ ∈ [0.5, 1.5] | 16/6 | 300 |
| D | SU(2) LGT, N = 2 | 4 | HEA | 4 | 48 | m̃ ∈ [0.1, 2], x ∈ [0.2, 5] | 16/6 | 250 |
| E | Collective neutrinos, N = 4 | 4 | HEA | 3 | 36 | θ ∈ [0.05, 0.7], μ ∈ [0.1, 3] | 16/6 | 200 |
| F | SUSY QM | 4 | HEA | 3 | 36 | W ∈ {HO, AHO, DW}; m, g, μ ∈ [0.5, 1.5] | 18/9 | 200 |
| G | D₈ LGT | 7 | HEA | 3 | 63 | M, λ_E, ε ∈ [0.2, 2] | 16/6 | 200 |
| H | Light nuclei, pionless EFT | 8 | UCCSD | 2 | 4/8/8 | ²H/³H/³He; t ∈ [0.5, 2], C ∈ [−2, −0.2], D ∈ [−1, 1] | 18/9 | 150 |
| I | Scalar Yukawa | 4 | HEA | 3 | 36 | η ∈ [0.2, 1], m ∈ [1, 2] | 16/6 | 150 |
| J | Lipkin–Meshkov–Glick, J = 3/2 | 2 | HEA | 3 | 18 | ε ∈ [0.5, 2], V ∈ [−1.5, 1.5] | 16/6 | 150 |

HEA is `StronglyEntanglingLayers`; HVA is a Trotterized gauge-invariant ansatz starting from the Gauss-law
vacuum. Two calibration systems are also included: Heisenberg XYZ (`V_heisenberg`) and H₂ (`V_h2`).

## What each ground state answers, and how it is verified

| Track | Question | Verification before use |
|---|---|---|
| A | Vacuum of 1+1D QED; its energy in (m₀, g) tracks the chiral-condensate and confinement regime | Hermiticity; Coulomb term against a brute-force dense construction |
| B | Vacuum of a single-plaquette Z₂ gauge theory with matter in the charge-free Gauss sector | G² = 1 and [H, G] = 0 at every site; ground state in the physical sector; HVA generators commute with every G |
| C | Ground state of two-site φ⁴ across the symmetric/broken-symmetry crossover | Φ, Π Hermitian; free-field spectrum |
| D | SU(2) vacuum with staggered fermions, the reference state for hadron masses | Hermiticity; colour-exchange term against an independent construction |
| E | Adiabatic ground state of the collective two-flavour neutrino Hamiltonian | μ = 0 spectrum against the analytic one; θ = μ = 0 ground state |
| F | Whether supersymmetry is broken: E₀ = 0 (HO, AHO) or E₀ > 0 (DW) | HO spectrum; DW potential on a grid; E₀ → 0 with the cutoff |
| G | Ground state of a non-Abelian D₈ gauge theory with fermionic matter | Group representation over all 64 pairs; unitarity; projector rank; Jordan–Wigner algebra |
| H | Binding energies of ²H, ³H and ³He in lattice pionless EFT | [H, N_p] = [H, N_n] = 0; bound state in each sector; UCCSD conserves N_p and N_n |
| I | Dressed fermion-vacuum ground state under scalar Yukawa coupling | Vacuum-sector energy against the closed form −(η/2)²/m |
| J | Collective ground state of the LMG model (nuclear structure) | su(2) algebra; V = 0 limit E₀ = −εJ |
