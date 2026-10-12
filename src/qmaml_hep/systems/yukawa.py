"""Track I: scalar Yukawa coupling (zero-dimensional / single-site), following Kaldenbach, Heller, Alber &
Stojanovic, "Digital Quantum Simulation of Scalar Yukawa Coupling" (arXiv:2211.02684), their Eqs. 6-7:

  H_0   = M (a^dag a + c^dag c) + m b^dag b
  H_int = (eta/2) (a^dag a + c^dag c - 1) (b + b^dag)

a = fermion annihilation, c = antifermion annihilation (two independent fermionic modes; H contains only
number operators a^dag a, c^dag c, never a raising/lowering combination of them, so -- as stated, not
re-derived -- there is no Jordan-Wigner string to thread between them: nothing in H anticommutes across the
two modes), b = boson annihilation (truncated Fock space, cutoff Lambda=2**N_b). eta = 4mg*beta^(3/2) is
their effective coupling constant, M the fermion mass, m the boson mass.

**Reformulation, stated up front (same spirit as Track E).** The source paper studies real-time dynamics
after a Yukawa-interaction quench (Loschmidt echo, boson-number growth) starting from the fermion-antifermion
vacuum, not a VQE ground-state search -- though unlike Track E, the reformulation here is direct: the same
Hamiltonian's ground state is a perfectly standard, well-posed VQE target, and the paper's own numerically-
exact classical benchmark (used for their adiabatic-state-preparation section, computing ground- and
first-excited-state energies) gives an independent way to sanity-check results beyond plain diagonalization.

**An analytic cross-check, not available in most other tracks.** Fixing the fermion sector to
(n_a, n_c) = (0, 0) (the vacuum), H reduces exactly to a displaced harmonic oscillator,
H = m*b^dag b - (eta/2)(b + b^dag), whose ground energy is known in closed form on the *untruncated* Fock
space: E_0 = -(eta/2)^2 / m (complete the square: H = m*(b^dag - eta/2m)(b - eta/2m) - (eta/2)^2/m, and the
first term is positive-semi-definite with minimum 0). The (n_a,n_c)=(1,1) sector gives the identical boson
ground energy shifted up by 2M; the two singly-occupied sectors give M with no boson driving at all. For
any physically reasonable M, m, eta > 0, the vacuum sector is therefore the true global ground state --
checked numerically below via the ground state's measured fermion occupation, not assumed. The ground
state here is a displaced-vacuum (coherent-state-like) state with mean boson number ~(eta/2m)^2; at our
truncation (Lambda=4, matching the source paper's own "up to three bosons" showcase scale), the analytic
formula is only approached, not matched exactly, once eta/2m gets large -- our task-parameter range is
chosen to keep eta/(2m) small enough (<~0.5) that the truncation error stays under ~1%, verified directly
below rather than assumed to be small.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

M_FERMION = 1.0  # fixed fermion mass (representation constant, not swept as a task parameter)


N_BOSON_QUBITS = 2  # Lambda = 4 boson levels (up to 3 bosons, matching the paper's own worked example)


NUM_QUBITS = 2 + N_BOSON_QUBITS  # 2 fermion qubits (a, c occupation) + boson register


def _ladder_operators(n_levels: int):
    a = np.zeros((n_levels, n_levels), dtype=complex)
    for n in range(1, n_levels):
        a[n - 1, n] = np.sqrt(n)
    return a, a.conj().T


def yukawa_hamiltonian(eta: float, m: float, M: float = M_FERMION, n_boson_qubits: int = N_BOSON_QUBITS) -> np.ndarray:
    """Dense Hermitian matrix, dimension 4 * Lambda (2 fermion qubits x Lambda boson levels).
    Task parameters: Yukawa coupling eta, boson mass m."""
    n_levels = 2**n_boson_qubits
    b, bdag = _ladder_operators(n_levels)
    n_b = bdag @ b
    I_b = np.eye(n_levels, dtype=complex)

    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    n_op = (I2 - Z) / 2.0  # number operator on a single fermionic (occupation) qubit

    n_a = np.kron(n_op, np.kron(I2, I_b))
    n_c = np.kron(I2, np.kron(n_op, I_b))
    n_fermion_total = n_a + n_c

    b_plus_bdag = np.kron(I2, np.kron(I2, b + bdag))
    n_b_full = np.kron(I2, np.kron(I2, n_b))
    I_full = np.kron(I2, np.kron(I2, I_b))

    H0 = M * n_fermion_total + m * n_b_full
    Hint = (eta / 2.0) * (n_fermion_total - I_full) @ b_plus_bdag
    return H0 + Hint


def exact_ground_energy(H: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(H)))


def analytic_vacuum_sector_ground_energy(eta: float, m: float) -> float:
    """E_0 = -(eta/2)^2 / m for the fermion-vacuum sector (see module docstring)."""
    return -((eta / 2.0) ** 2) / m


def sample_task_space(n_tasks: int, seed: int, eta_range=(0.2, 1.0), m_range=(1.0, 2.0)) -> np.ndarray:
    """Task = (eta, m): Yukawa coupling, boson mass (fermion mass M fixed)."""
    rng = np.random.default_rng(seed)
    eta = rng.uniform(*eta_range, size=n_tasks)
    m = rng.uniform(*m_range, size=n_tasks)
    return np.stack([eta, m], axis=1)


def param_shape(depth: int, num_qubits: int = NUM_QUBITS) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)


def verify_construction(atol: float = 1e-6) -> bool:
    """(1) H Hermitian, always exact. (2) Ground energy approaches the analytic displaced-harmonic-
    oscillator formula (exact only on the untruncated Fock space) within ~1% relative error at small
    eta/(2m), verified directly rather than assumed -- and the error is shown growing with eta/(2m) to make
    the truncation limit's shape visible, not just a single pass/fail point. (3) The numerically found
    ground state's measured fermion occupation is exactly 0 (vacuum sector is truly the global minimum,
    not merely the lowest of a restricted set), checked directly on the eigenvector, not assumed."""
    ok = True
    for eta, m in [(0.3, 1.8), (0.6, 1.5), (1.0, 1.0), (2.0, 1.0)]:
        H = yukawa_hamiltonian(eta, m)
        herm_err = np.max(np.abs(H - H.conj().T))
        E0_numeric = exact_ground_energy(H)
        E0_analytic = analytic_vacuum_sector_ground_energy(eta, m)
        rel_err = abs(E0_numeric - E0_analytic) / abs(E0_analytic)
        alpha = eta / (2 * m)
        # only the small-alpha points (inside our sampled task range) must pass the 1% bar; the eta=2.0
        # point is included specifically to show truncation error growing outside that range, not to pass
        must_pass = alpha <= 0.5
        passed = herm_err < atol and (rel_err < 0.01 or not must_pass)
        print(
            f"  eta={eta} m={m} alpha={alpha:.2f}: Hermiticity err={herm_err:.2e}, E0 numeric={E0_numeric:.6f} "
            f"vs. analytic={E0_analytic:.6f} (rel err={rel_err:.2%}){'  [outside sampled range, informational]' if not must_pass else ''}  {'OK' if passed else 'FAIL'}"
        )
        ok = ok and passed

    n_levels = 2**N_BOSON_QUBITS
    H = yukawa_hamiltonian(eta=1.5, m=1.0)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_vec = eigvecs[:, 0]
    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    n_op = (I2 - Z) / 2.0
    I_b = np.eye(n_levels, dtype=complex)
    n_a = np.kron(n_op, np.kron(I2, I_b))
    n_c = np.kron(I2, np.kron(n_op, I_b))
    occ = np.real(ground_vec.conj() @ (n_a + n_c) @ ground_vec)
    vac_ok = abs(occ) < 1e-6
    print(
        f"  Ground state fermion occupation <n_a+n_c>={occ:.6f} (want 0, vacuum sector)  {'OK' if vac_ok else 'FAIL'}"
    )
    ok = ok and vac_ok

    return ok


# ---------------------------------------------------------------------------------------------------------
# Variational circuits (PennyLane reference implementations; the benchmark runs the exact torch
# simulators in qmaml_hep.simulators, cross-checked against these on every run)
# ---------------------------------------------------------------------------------------------------------


class YukawaPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_Yukawa(eta,m)|psi(theta)>, measured via qml.Hermitian."""

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
