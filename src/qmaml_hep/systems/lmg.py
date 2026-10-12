"""Track J: Lipkin-Meshkov-Glick (LMG) model, following Robin & Savage, "Quantum Simulations in Effective
Model Spaces (I): Hamiltonian-Learning VQE using Digital Quantum Computers and Application to the
Lipkin-Meshkov-Glick Model" (arXiv:2301.05976, Phys. Rev. C 108, 024313), their Eqs. 1-2, 9-10.

**Flagged, not core HEP**, per this project's own tracks document: the LMG model is a nuclear-structure
collective-many-body Hamiltonian, not a particle-physics or gauge-theory system.

**Naming collision, worth stating precisely.** The cited paper's method, HL-VQE (Hamiltonian-Learning
VQE), does NOT use a classical neural network to predict circuit parameters the way Q-MAML does, despite
the similar-sounding name. HL-VQE iteratively co-optimizes an *effective Hamiltonian* (via a single-
particle-basis orbital rotation, their Eq. 11-13, parameterized by an angle beta) together with the
variational wavefunction, exploiting the non-commutativity of symmetries and Hilbert-space truncations to
localize entanglement into the truncated space -- a genuinely different mechanism aimed at a similar goal
(better VQE performance in a truncated model space). This module implements only the plain,
un-rotated LMG Hamiltonian (their Eq. 1, beta=0) as a Q-MAML VQE benchmark; it does not reproduce
HL-VQE's orbital-rotation learning step.

**The Hamiltonian** (Eq. 1), restricted to the fully-symmetric J=N/2 collective sector (N = number of
particles), in the angular-momentum basis {|J,M>}, M=-J..J:

  H = epsilon * J_z - (V/2) * (J_+^2 + J_-^2)

J_z, J_+, J_- are standard su(2) generators; matrix elements (Eq. 9-10, with n = M+J so |n> <-> |J,M=n-J>):

  <n|H|n>   = epsilon * (n - J)
  <n+2|H|n> = -(V/2) * sqrt[J(J+1)-(n-J)(n-J+1)] * sqrt[J(J+1)-(n-J+1)(n-J+2)]
  <n-2|H|n> = -(V/2) * sqrt[J(J+1)-(n-J)(n-J-1)] * sqrt[J(J+1)-(n-J-1)(n-J-2)]

Task parameters: single-particle splitting epsilon, interaction strength V.

**Qubit count.** The cited paper's whole point is a mapping where qubit count scales with log2(2J+1)
(effective model space size) rather than particle number N; their own demonstrations use 1- and 2-qubit
effective model spaces. We use J=3/2 (2J+1=4 states, exactly 2 qubits, no padding needed) -- matching their
own smallest/showcase scale directly, not an arbitrary further reduction.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

J_SPIN = 1.5  # total collective spin (N=3 particles' fully-symmetric sector); 2J+1=4 states = 2 qubits


NUM_QUBITS = 2


def _dim(J: float) -> int:
    return int(round(2 * J + 1))


def lmg_hamiltonian(epsilon: float, V: float, J: float = J_SPIN) -> np.ndarray:
    """Dense Hermitian matrix, dimension 2J+1, in the |J,M> basis (n = M+J, n=0..2J).
    Task parameters: single-particle splitting epsilon, interaction strength V."""
    dim = _dim(J)
    H = np.zeros((dim, dim), dtype=complex)
    for n in range(dim):
        M = n - J
        H[n, n] = epsilon * M
    for n in range(dim - 2):
        M = n - J
        cplus = np.sqrt(J * (J + 1) - M * (M + 1)) * np.sqrt(J * (J + 1) - (M + 1) * (M + 2))
        H[n + 2, n] = -(V / 2.0) * cplus
        H[n, n + 2] = -(V / 2.0) * cplus  # real & symmetric for real cplus (J+^2 and J-^2 are h.c. pair)
    return H


def exact_ground_energy(H: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(H)))


def sample_task_space(n_tasks: int, seed: int, epsilon_range=(0.5, 2.0), V_range=(-1.5, 1.5)) -> np.ndarray:
    """Task = (epsilon, V): single-particle splitting, interaction strength."""
    rng = np.random.default_rng(seed)
    epsilon = rng.uniform(*epsilon_range, size=n_tasks)
    V = rng.uniform(*V_range, size=n_tasks)
    return np.stack([epsilon, V], axis=1)


def param_shape(depth: int, num_qubits: int = NUM_QUBITS) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)


def _collective_ops(J: float = J_SPIN):
    dim = _dim(J)
    Jz = np.diag([n - J for n in range(dim)]).astype(complex)
    Jp = np.zeros((dim, dim), dtype=complex)
    for n in range(dim - 1):
        M = n - J
        Jp[n + 1, n] = np.sqrt(J * (J + 1) - M * (M + 1))
    Jm = Jp.conj().T
    return Jz, Jp, Jm


def verify_construction(atol: float = 1e-8) -> bool:
    """(1) H Hermitian at a sample point. (2) su(2) algebra check on the constructed generators:
    [J_z,J_+]=J_+, [J_+,J_-]=2J_z -- exact, well-known relations, checked directly on the matrices this
    module builds from, not assumed. (3) V=0 limit: H is purely diagonal (epsilon*J_z), so the exact
    ground state must be the fully-analytically-known M=-J state with E=-epsilon*J (for epsilon>0) --
    checked against exact diagonalization, not assumed."""
    ok = True
    H = lmg_hamiltonian(epsilon=1.0, V=0.6)
    herm_err = np.max(np.abs(H - H.conj().T))
    print(f"  Hermiticity: max|H-H^dag|={herm_err:.2e}  {'OK' if herm_err < atol else 'FAIL'}")
    ok = ok and herm_err < atol

    Jz, Jp, Jm = _collective_ops()
    comm1 = Jz @ Jp - Jp @ Jz - Jp
    comm2 = Jp @ Jm - Jm @ Jp - 2 * Jz
    c1_err = np.max(np.abs(comm1))
    c2_err = np.max(np.abs(comm2))
    algebra_ok = c1_err < atol and c2_err < atol
    print(
        f"  su(2) algebra: [Jz,J+]-J+ err={c1_err:.2e}, [J+,J-]-2Jz err={c2_err:.2e}  {'OK' if algebra_ok else 'FAIL'}"
    )
    ok = ok and algebra_ok

    H_v0 = lmg_hamiltonian(epsilon=1.3, V=0.0)
    E0_numeric = exact_ground_energy(H_v0)
    E0_analytic = -1.3 * J_SPIN
    v0_err = abs(E0_numeric - E0_analytic)
    v0_ok = v0_err < atol
    print(
        f"  V=0 limit: E0 numeric={E0_numeric:.6f} vs. analytic -epsilon*J={E0_analytic:.6f}  {'OK' if v0_ok else 'FAIL'}"
    )
    ok = ok and v0_ok

    return ok


# ---------------------------------------------------------------------------------------------------------
# Variational circuits (PennyLane reference implementations; the benchmark runs the exact torch
# simulators in qmaml_hep.simulators, cross-checked against these on every run)
# ---------------------------------------------------------------------------------------------------------


class LMGPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_LMG(epsilon,V)|psi(theta)>, measured via qml.Hermitian."""

    def __init__(self, num_qubits: int = NUM_QUBITS, depth: int = 3):
        super().__init__()
        import pennylane as qml

        self.num_qubits = num_qubits
        self.depth = depth
        self._qml = qml
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self._diff_method = "backprop"

    def make_circuit(self, H_matrix: np.ndarray):
        qml = self._qml
        obs = qml.Hermitian(H_matrix, wires=range(self.num_qubits))

        @qml.qnode(self.dev, interface="torch", diff_method=self._diff_method)
        def circuit(weights):
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.expval(obs)

        return circuit

    def cost(self, weights: torch.Tensor, H_matrix: np.ndarray) -> torch.Tensor:
        circuit = self.make_circuit(H_matrix)
        return circuit(weights.to(torch.float64))
