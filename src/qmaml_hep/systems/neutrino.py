"""Track E: collective neutrino oscillations -- reformulated as a ground-state problem.

**Why this needs a reformulation, stated up front.** The literature this track is nominally based on
(e.g. arXiv:2102.12556, quantum-simulating collective neutrino oscillations) studies *real-time evolution*:
prepare a flavor eigenstate, time-evolve it under the collective Hamiltonian, and track entanglement/flavor
observables as a function of time. That is not a ground-state/VQE problem and does not fit the Q-MAML
formalism (a Learner predicting a good VQE initial state) without a genuine change of question. This module
makes that change explicitly rather than silently: instead of asking "how does a flavor eigenstate evolve
under H(t)", it asks "what is the ground state of the same class of collective-oscillation Hamiltonian, and
can a Learner initialize a VQE to find it faster". This is not the cited paper's question, and this module
does not claim to reproduce any of its results. It is, however, a well-posed and physically motivated
question in its own right: the ground state of the collective (all-to-all) neutrino Hamiltonian is exactly
the object studied in the *adiabatic*/spectral-split literature on collective flavor transformation (Duan,
Fuller & Qian, Ann. Rev. Nucl. Part. Sci. 2010, and follow-on work treating the "co-rotating"/adiabatic
ground state as the physically relevant collective-oscillation endpoint), so a VQE ground-state search on
this Hamiltonian is a legitimate physics question, just a different one from real-time entanglement growth.
Flagged in the project's own tracks document as "a genuine research step, not just engineering" -- this is
that step, made explicit rather than glossed over.

**The Hamiltonian.** Standard two-flavor "bipolar"/single-angle collective neutrino Hamiltonian, N
neutrinos each represented by one flavor-isospin qubit (|0> = nu_e, |1> = nu_x), with a single-particle
vacuum-oscillation term set by the mixing angle theta and an all-to-all (collective) neutrino-neutrino
forward-scattering term set by interaction strength mu:

  H(theta, mu) = (omega/2) * sum_i ( cos(2*theta) Z_i + sin(2*theta) X_i )
               + (mu/N) * sum_{i<j} ( X_i X_j + Y_i Y_j + Z_i Z_j )

omega (vacuum oscillation frequency scale) is fixed to 1 (not a task parameter, matching how
systems/scalar.py fixes its representation parameter mu_disc); (theta, mu) are the task parameters --
theta is the standard neutrino mixing angle, mu/omega sets whether the system is in the weak-interaction
("bipolar oscillation") or strong-interaction ("synchronized") regime, a genuine physical crossover in the
collective-oscillation literature, analogous in spirit (though not mechanism) to Track C's phase transition.

MVP scale: N=4 neutrinos (4 qubits), matching this track's own "small, near-term" framing in the project's
tracks document.
"""

from typing import Tuple

import numpy as np
import pennylane as qml
import scipy.sparse.linalg
import torch
import torch.nn as nn

OMEGA = 1.0  # fixed representation/scale parameter, not a task parameter (see docstring)


N_NEUTRINOS_DEFAULT = 4


def neutrino_hamiltonian(N: int, theta: float, mu: float, omega: float = OMEGA) -> qml.Hamiltonian:
    """Collective (all-to-all) two-flavor neutrino oscillation Hamiltonian, N qubits.
    Task parameters: mixing angle theta, collective interaction strength mu."""
    coeffs, ops = [], []
    for i in range(N):
        coeffs += [omega / 2.0 * np.cos(2 * theta), omega / 2.0 * np.sin(2 * theta)]
        ops += [qml.PauliZ(i), qml.PauliX(i)]
    for i in range(N):
        for j in range(i + 1, N):
            w = mu / N
            coeffs += [w, w, w]
            ops += [qml.PauliX(i) @ qml.PauliX(j), qml.PauliY(i) @ qml.PauliY(j), qml.PauliZ(i) @ qml.PauliZ(j)]
    return qml.Hamiltonian(coeffs, ops)


def exact_ground_energy(H: qml.Hamiltonian, num_qubits: int) -> float:
    sm = H.sparse_matrix(wire_order=list(range(num_qubits)))
    if sm.shape[0] <= 64:
        return float(np.min(np.linalg.eigvalsh(sm.toarray())))
    val = scipy.sparse.linalg.eigsh(sm, k=1, which="SA", return_eigenvectors=False)
    return float(val[0])


def sample_task_space(n_tasks: int, seed: int, theta_range=(0.05, 0.7), mu_range=(0.1, 3.0)) -> np.ndarray:
    """Task = (theta, mu). theta spans small-to-near-maximal mixing (radians, below pi/4); mu spans
    weak-interaction (bipolar-like) to strong-interaction (synchronized-like) regimes at fixed omega=1."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(*theta_range, size=n_tasks)
    mu = rng.uniform(*mu_range, size=n_tasks)
    return np.stack([theta, mu], axis=1)


def param_shape(depth: int, num_qubits: int) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)


def verify_construction(N: int = N_NEUTRINOS_DEFAULT, atol: float = 1e-8) -> bool:
    """(1) H Hermitian to machine precision. (2) Limiting-case sanity checks, not assumed: at mu=0 (no
    neutrino-neutrino interaction), H reduces to N independent single-qubit terms, so the ground energy
    must equal N * (-omega/2) exactly (each qubit independently minimized along its own field direction) --
    checked against exact diagonalization, not just asserted. (3) At theta=0, mu=0, H = (omega/2) sum Z_i,
    an exactly diagonal Hamiltonian; ground energy must be exactly -N*omega/2 and the ground state must be
    the all-|1> computational basis state (checked via the diagonalized eigenvector, not assumed)."""
    ok = True
    H0 = neutrino_hamiltonian(N, theta=0.3, mu=0.0)
    Hd = H0.sparse_matrix(wire_order=list(range(N))).toarray()
    herm_err = np.max(np.abs(Hd - Hd.conj().T))
    print(f"  Hermiticity (mu=0): max|H-H^dag|={herm_err:.2e}")
    ok = ok and herm_err < atol

    E0_mu0 = exact_ground_energy(H0, N)
    expected_mu0 = -N * OMEGA / 2.0
    err_mu0 = abs(E0_mu0 - expected_mu0)
    print(f"  mu=0 ground energy: {E0_mu0:.6f} vs. analytic -N*omega/2={expected_mu0:.6f}  err={err_mu0:.2e}")
    ok = ok and err_mu0 < atol

    H_diag = neutrino_hamiltonian(N, theta=0.0, mu=0.0)
    Hd2 = H_diag.sparse_matrix(wire_order=list(range(N))).toarray()
    eigvals, eigvecs = np.linalg.eigh(Hd2)
    ground_vec = eigvecs[:, np.argmin(eigvals)]
    all_ones_index = 2**N - 1  # computational basis |1,1,...,1>
    overlap = abs(ground_vec[all_ones_index]) ** 2
    print(f"  theta=0,mu=0 ground state overlap with |11...1>: {overlap:.6f}")
    ok = ok and overlap > 1.0 - atol

    return ok


# ---------------------------------------------------------------------------------------------------------
# Variational circuits (PennyLane reference implementations; the benchmark runs the exact torch
# simulators in qmaml_hep.simulators, cross-checked against these on every run)
# ---------------------------------------------------------------------------------------------------------


NUM_QUBITS = N_NEUTRINOS_DEFAULT


class NeutrinoPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_neutrino(theta_mix,mu)|psi(theta)>, measured directly via qml.expval(H).
    (Naming note: `theta` below is always the PQC's variational weight tensor, never the neutrino mixing
    angle -- the Hamiltonian's mixing angle is always spelled `theta_mix` at call sites in this module to
    avoid the name collision.)"""

    def __init__(self, num_qubits: int = NUM_QUBITS, depth: int = 3, use_lightning: bool = True):
        super().__init__()
        import pennylane as qml

        self.num_qubits = num_qubits
        self.depth = depth
        self._qml = qml
        diff_method = "parameter-shift"
        if use_lightning:
            try:
                self.dev = qml.device("lightning.qubit", wires=num_qubits)
                diff_method = "adjoint"
            except Exception:
                self.dev = qml.device("default.qubit", wires=num_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=num_qubits)
        self._diff_method = diff_method

    def make_circuit(self, H):
        qml = self._qml

        @qml.qnode(self.dev, interface="torch", diff_method=self._diff_method)
        def circuit(weights):
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.expval(H)

        return circuit

    def cost(self, weights: torch.Tensor, H) -> torch.Tensor:
        circuit = self.make_circuit(H)
        return circuit(weights.to(torch.float64))
