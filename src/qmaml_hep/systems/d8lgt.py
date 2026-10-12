"""Track G: non-Abelian D8 (dihedral group of order 8) lattice gauge theory with dynamical fermionic
matter, following Gaz, Popov, Pardo, Lewenstein, Hauke & Zohar, "Quantum Simulation of non-Abelian
Lattice Gauge Theories: a variational approach to D8" (arXiv:2501.17863), their Eqs. 1-24 (the original,
pre-matter-removal Kogut-Susskind Hamiltonian). Flagged in this project's own tracks document as "not
recommended for now": qudit hardware, and no small-qubit reduction confirmed feasible. Investigated before
building anything (see the accompanying research notebook's opening section): the paper's own qubit-count
reduction ("matter removal", their Eqs. 27-45 and Section III) could not be extracted with enough fidelity
from available tooling to transcribe correctly (several of its intermediate equations were only partially
recoverable). Rather than guess at that reduction, this module implements their ORIGINAL, fully-specified
Hamiltonian (Eqs. 1-24) directly, with explicit fermionic matter (Jordan-Wigner-mapped) instead of their
Hilbert-space-reducing matter-removal trick -- more qubits than their reduced scheme would use, but every
term is transcribed from a completely stated equation, not a partially-recovered one.

**The group.** D8 = {a^p y^q | p=0,1,2,3; q=0,1}, a^4=y^2=e, yay=a^{-1} (paper's Eqs. 2-4). Multiplication:
(a^p1 y^q1)(a^p2 y^q2) = a^{p1 + p2*(-1)^q1 mod 4} y^{(q1+q2) mod 2}.

**Irreps** (paper's Eq. 5-6): four 1-dimensional (j=0, 0bar, 1, 1bar) and one 2-dimensional, faithful
irrep j=2 with generator matrices D^2(a) = -i*sigma_y, D^2(y) = sigma_z. D^2(a^p y^q) = D^2(a)^p D^2(y)^q.

**Link Hilbert space**: dimension 8 (= |D8|), one qubit (q, 2 levels) tensor one 4-level qudit (p, 4
levels), |g> = |a^p y^q> = |p>_qudit x |q>_qubit (paper's Fig. 2). Represented here as 3 qubits (1 for the
2-level q register, 2 for the 4-level p register), matching the paper's own qubit/qudit split exactly in
dimension even though we use qubits throughout rather than a native 4-level qudit.

**Hamiltonian** (paper's Eqs. 20-24), N=2 staggered sites (x=0,1), 1 link, no plaquette term (H_B needs a
closed loop, impossible in a 1D chain -- the paper's own N=4 chain example states this explicitly,
Section V.1):

  H = H_M + H_E + H_GM
  H_M  = M * sum_x (-1)^x * sum_m n_{x,m},                    m = 0,1 (matter species / j=2 flavor index)
  H_E  = lambda_E * Pi_{j=2}(link),                            Pi_{j=2} = projector onto the j=2 irrep sector
  H_GM = epsilon * sum_{m,n=0,1} psi^dag_m(0) U_mn(link) psi_n(1) + h.c.

n_{x,m} = psi^dag_{x,m} psi_{x,m}; psi_{x,m} are fermionic (Jordan-Wigner-mapped over the 4 matter modes,
ordered (x=0,m=0),(x=0,m=1),(x=1,m=0),(x=1,m=1)); U_mn is diagonal in the group-element basis with
U_mn|g> = D^2_mn(g)|g> (paper's Eq. 14).

MVP scope: N=2 (2 sites, 1 link, 4 matter qubits + 3 link qubits = 7 qubits total), vs. the paper's own
smallest worked example at N=4 (3 links). No plaquette term at any N in 1D. Does not reproduce the paper's
matter-removal Hilbert-space reduction, their trapped-ion qudit gate set, or their VarQITE results -- see
the companion notebook's scope note.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

N_SITES = 2


N_MATTER_QUBITS = N_SITES * 2  # 2 fermionic species per site, Jordan-Wigner mapped


N_LINK_QUBITS = 3  # 1 (q, 2-level) + 2 (p, 4-level) per link


N_LINKS = N_SITES - 1  # open chain


NUM_QUBITS = N_MATTER_QUBITS + N_LINKS * N_LINK_QUBITS  # 4 + 3 = 7


I2 = np.eye(2, dtype=complex)


X = np.array([[0, 1], [1, 0]], dtype=complex)


Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


Z = np.array([[1, 0], [0, -1]], dtype=complex)


SIGMA_PLUS = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|... convention: raises 1->0? see _ladder note


SIGMA_MINUS = SIGMA_PLUS.conj().T


def _kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _embed_1q(op: np.ndarray, wire: int, n: int) -> np.ndarray:
    mats = [I2] * n
    mats[wire] = op
    return _kron_all(mats)


GROUP_ELEMENTS = [(p, q) for p in range(4) for q in range(2)]  # (p, q) <-> a^p y^q


GROUP_INDEX = {g: i for i, g in enumerate(GROUP_ELEMENTS)}  # index in the 8-dim group-element basis


def group_multiply(g1: Tuple[int, int], g2: Tuple[int, int]) -> Tuple[int, int]:
    """(a^p1 y^q1)(a^p2 y^q2) = a^{p1 + p2*(-1)^q1} y^{q1+q2}, from yay=a^{-1}."""
    p1, q1 = g1
    p2, q2 = g2
    sign = 1 if q1 == 0 else -1
    p = (p1 + sign * p2) % 4
    q = (q1 + q2) % 2
    return (p, q)


def D2(g: Tuple[int, int]) -> np.ndarray:
    """The 2-dimensional (j=2) irrep matrix for group element g=(p,q): D^2(a)^p @ D^2(y)^q."""
    Da = -1j * Y  # D^2(a) = -i*sigma_y
    Dy = Z  # D^2(y) = sigma_z
    p, q = g
    out = np.eye(2, dtype=complex)
    for _ in range(p):
        out = out @ Da
    for _ in range(q):
        out = out @ Dy
    return out


def _u_mn_diagonal(m: int, n: int) -> np.ndarray:
    """U_mn as an 8x8 matrix, diagonal in the group-element basis: U_mn|g> = D^2_mn(g)|g> (paper's Eq. 14)."""
    diag = np.array([D2(g)[m, n] for g in GROUP_ELEMENTS], dtype=complex)
    return np.diag(diag)


def _link_qubit_index(g: Tuple[int, int]) -> int:
    """Map group element (p,q) to a computational-basis index on the link's 3 qubits: 1 qubit for q
    (2 levels) + 2 qubits for p (4 levels), matching the paper's Fig. 2 split. Basis order: |p>|q>."""
    p, q = g
    return p * 2 + q  # matches GROUP_ELEMENTS / GROUP_INDEX ordering exactly (p outer, q inner)


def _change_of_basis_to_irreps() -> np.ndarray:
    """8x8 unitary W with W @ |g>-basis-vector giving components in the |j,m,n> basis, ordered:
    [j=0, j=0bar, j=1, j=1bar] (1 state each) then [j=2: (m,n) = (0,0),(0,1),(1,0),(1,1)] (4 states).
    |jmn> = sum_g sqrt(dim(j)/8) * conj(D^j_mn(g)) |g>  (inverting the paper's Eq. 9 Peter-Weyl relation)."""

    def D_triv(g, sign_a, sign_y):
        p, q = g
        return (sign_a**p) * (sign_y**q)

    trivial_chars = [
        (1, 1),  # j=0
        (1, -1),  # j=0bar
        (-1, 1),  # j=1
        (-1, -1),  # j=1bar
    ]
    W = np.zeros((8, 8), dtype=complex)
    col = 0
    for sign_a, sign_y in trivial_chars:
        for gi, g in enumerate(GROUP_ELEMENTS):
            W[gi, col] = np.sqrt(1.0 / 8.0) * np.conj(D_triv(g, sign_a, sign_y))
        col += 1
    for m in range(2):
        for n in range(2):
            for gi, g in enumerate(GROUP_ELEMENTS):
                W[gi, col] = np.sqrt(2.0 / 8.0) * np.conj(D2(g)[m, n])
            col += 1
    return W


def electric_projector() -> np.ndarray:
    """Pi_{j=2}: projector onto the 4-dimensional j=2 sector of one link's 8-dim space."""
    W = _change_of_basis_to_irreps()
    W_j2 = W[:, 4:8]
    return W_j2 @ W_j2.conj().T


def _jw_annihilation(k: int, n_modes: int) -> np.ndarray:
    """psi_k on n_modes fermionic modes, Jordan-Wigner-mapped: (prod_{j<k} Z_j) sigma-_k."""
    mats = []
    for j in range(n_modes):
        if j < k:
            mats.append(Z)
        elif j == k:
            mats.append(SIGMA_MINUS)
        else:
            mats.append(I2)
    return _kron_all(mats)


def d8_lgt_hamiltonian(M: float, lambda_E: float, epsilon: float) -> np.ndarray:
    """N=2 (1 link) D8 LGT Hamiltonian, NUM_QUBITS=7 (4 matter + 3 link). Dense Hermitian matrix.
    Task parameters: staggered mass M, electric coupling lambda_E, gauge-matter hopping epsilon."""
    n_matter = N_MATTER_QUBITS
    n_link = N_LINK_QUBITS
    dim_matter = 2**n_matter
    dim_link = 2**n_link

    psi = [_jw_annihilation(k, n_matter) for k in range(n_matter)]  # modes 0,1 = site0 (m=0,1); 2,3 = site1

    # H_M = M * sum_x (-1)^x * sum_m n_{x,m}
    H_M_matter = np.zeros((dim_matter, dim_matter), dtype=complex)
    for site, sign in [(0, 1.0), (1, -1.0)]:
        for m in range(2):
            k = site * 2 + m
            n_k = psi[k].conj().T @ psi[k]
            H_M_matter += M * sign * n_k
    H_M = np.kron(H_M_matter, np.eye(dim_link, dtype=complex))

    # H_E = lambda_E * Pi_{j=2}(link), embedded on the link register's 3 qubits via the group-element
    # basis <-> qubit-index map, then lifted to the full (matter x link) space.
    Pi_g_basis = electric_projector()  # 8x8, indexed by GROUP_ELEMENTS order
    Pi_link = np.zeros((dim_link, dim_link), dtype=complex)
    for gi, g in enumerate(GROUP_ELEMENTS):
        for gj, g2 in enumerate(GROUP_ELEMENTS):
            Pi_link[_link_qubit_index(g), _link_qubit_index(g2)] = Pi_g_basis[gi, gj]
    H_E = lambda_E * np.kron(np.eye(dim_matter, dtype=complex), Pi_link)

    # H_GM = epsilon * sum_{m,n} psi^dag_m(0) U_mn(link) psi_n(1) + h.c.
    H_GM_full = np.zeros((dim_matter * dim_link, dim_matter * dim_link), dtype=complex)
    for m in range(2):
        for n in range(2):
            U_mn_g_basis = _u_mn_diagonal(m, n)  # 8x8 diagonal, indexed by GROUP_ELEMENTS order
            U_mn_link = np.zeros((dim_link, dim_link), dtype=complex)
            for gi, g in enumerate(GROUP_ELEMENTS):
                U_mn_link[_link_qubit_index(g), _link_qubit_index(g)] = U_mn_g_basis[gi, gi]
            term = np.kron(psi[m].conj().T @ psi[2 + n], U_mn_link)
            H_GM_full += epsilon * term
    H_GM = H_GM_full + H_GM_full.conj().T

    return H_M + H_E + H_GM


def exact_ground_energy(H: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(H)))


def sample_task_space(
    n_tasks: int, seed: int, M_range=(0.2, 2.0), lambdaE_range=(0.2, 2.0), eps_range=(0.2, 2.0)
) -> np.ndarray:
    """Task = (M, lambda_E, epsilon): staggered mass, electric coupling, gauge-matter hopping."""
    rng = np.random.default_rng(seed)
    M = rng.uniform(*M_range, size=n_tasks)
    lam = rng.uniform(*lambdaE_range, size=n_tasks)
    eps = rng.uniform(*eps_range, size=n_tasks)
    return np.stack([M, lam, eps], axis=1)


def param_shape(depth: int, num_qubits: int = NUM_QUBITS) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)


def verify_construction(atol: float = 1e-8) -> bool:
    """(1) D^2(g) is unitary for every g. (2) D^2 is a genuine representation:
    D^2(g1) @ D^2(g2) == D^2(g1*g2) for all 64 pairs -- a strong, independent check on both the irrep
    matrices and the group multiplication table together. (3) The irrep change-of-basis W is unitary.
    (4) Pi_{j=2} is a valid projector (Hermitian, idempotent, rank 4). (5) Jordan-Wigner fermionic
    operators satisfy the canonical anticommutation relations. (6) H is Hermitian at a sample parameter
    point."""
    ok = True

    for g in GROUP_ELEMENTS:
        Dg = D2(g)
        err = np.max(np.abs(Dg @ Dg.conj().T - np.eye(2)))
        ok = ok and err < atol
    print(f"  D^2(g) unitary for all 8 elements: {'OK' if ok else 'FAIL'}")

    rep_ok = True
    for g1 in GROUP_ELEMENTS:
        for g2 in GROUP_ELEMENTS:
            lhs = D2(g1) @ D2(g2)
            rhs = D2(group_multiply(g1, g2))
            if np.max(np.abs(lhs - rhs)) > atol:
                rep_ok = False
    print(f"  D^2(g1)D^2(g2) = D^2(g1*g2) for all 64 pairs: {'OK' if rep_ok else 'FAIL'}")
    ok = ok and rep_ok

    W = _change_of_basis_to_irreps()
    unit_err = np.max(np.abs(W @ W.conj().T - np.eye(8)))
    print(f"  Irrep change-of-basis W unitary: max err={unit_err:.2e}  {'OK' if unit_err < atol else 'FAIL'}")
    ok = ok and unit_err < atol

    Pi = electric_projector()
    herm_err = np.max(np.abs(Pi - Pi.conj().T))
    idem_err = np.max(np.abs(Pi @ Pi - Pi))
    rank = int(round(np.real(np.trace(Pi))))
    proj_ok = herm_err < atol and idem_err < atol and rank == 4
    print(
        f"  Pi_(j=2) projector: Hermitian err={herm_err:.2e}, idempotent err={idem_err:.2e}, "
        f"rank={rank} (want 4)  {'OK' if proj_ok else 'FAIL'}"
    )
    ok = ok and proj_ok

    n_modes = N_MATTER_QUBITS
    psi = [_jw_annihilation(k, n_modes) for k in range(n_modes)]
    jw_ok = True
    for i in range(n_modes):
        for j in range(n_modes):
            anticomm = psi[i] @ psi[j].conj().T + psi[j].conj().T @ psi[i]
            target = np.eye(2**n_modes) if i == j else np.zeros((2**n_modes, 2**n_modes))
            if np.max(np.abs(anticomm - target)) > atol:
                jw_ok = False
    print(f"  Jordan-Wigner CAR {{psi_i, psi_j^dag}}=delta_ij: {'OK' if jw_ok else 'FAIL'}")
    ok = ok and jw_ok

    H = d8_lgt_hamiltonian(M=1.0, lambda_E=1.0, epsilon=1.0)
    herm_err_H = np.max(np.abs(H - H.conj().T))
    print(f"  H Hermiticity: max|H-H^dag|={herm_err_H:.2e}  {'OK' if herm_err_H < atol else 'FAIL'}")
    ok = ok and herm_err_H < atol

    return ok


# ---------------------------------------------------------------------------------------------------------
# Variational circuits (PennyLane reference implementations; the benchmark runs the exact torch
# simulators in qmaml_hep.simulators, cross-checked against these on every run)
# ---------------------------------------------------------------------------------------------------------


class D8LGTPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_D8LGT(M,lambda_E,epsilon)|psi(theta)>, measured via qml.Hermitian."""

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
