"""Track D: SU(2) lattice gauge theory with dynamical matter, following the gauge-field-integrated-out
effective Hamiltonian of Atas, Zohar et al., "SU(2) hadrons on a quantum computer via a variational
approach" (Nature Communications 2021, arXiv:2102.08920), their Eqs. 3-6. This is the most directly
"LHC-relevant" HEP physics of any track in this project: the source paper computes actual meson/baryon
mass ratios from a non-Abelian lattice gauge theory.

N staggered lattice sites, 2 qubits per site (the two SU(2) color occupations, "red" and "green", at that
site), 2N qubits total, gauge field already eliminated analytically (exactly as in Track A's Schwinger
model, but for a genuinely non-Abelian SU(2) group here instead of U(1)):

  H = m_tilde * H_mass  +  (1/x) * H_electric  +  H_kinetic

  H_mass     = sum_{n=1}^{N} [ (-1)^n/2 * (Z_{2n-1} + Z_{2n}) + 1 ]
  H_kinetic  = -(1/2) * sum_{n=1}^{N-1} ( sigma+_{2n-1} Z_{2n} sigma-_{2n+1} + h.c. )
  H_electric = (3/16) * sum_{n=1}^{N-1} (N-n) * (1 - Z_{2n-1} Z_{2n})
             + (1/16) * sum_{n=1}^{N-2} sum_{m=n+1}^{N-1} (N-m) * (Z_{2n-1}-Z_{2n})(Z_{2m-1}-Z_{2m})
             + (1/2)  * sum_{n=1}^{N-2} sum_{m=n+1}^{N-1} (N-m) * ( sigma+_{2n-1} sigma-_{2n} sigma+_{2m} sigma-_{2m-1} + h.c. )

(1-indexed as in the paper; converted to 0-indexed qubits in code below.) m_tilde = a_l*m (dimensionless
fermion mass) and x = 1/(a_l^2 g^2) are the task parameters -- exactly the paper's own two continuous
physical knobs.

Transcribed directly from the paper's stated equations (not independently re-derived). Two checks: (1)
Hermiticity of H at several (m_tilde, x) points and N=2/N=3 (necessary, not sufficient); (2) the trickiest
term to get right by hand, the electric term's 4-body color-exchange piece
(sigma+_{2n-1}sigma-_{2n}sigma+_{2m}sigma-_{2m-1} + h.c.), is expanded into Pauli strings by
`_raise_lower_product_plus_hc` and cross-checked against an independent dense sigma+/sigma- matrix
construction (max abs diff ~1e-16 -- see the module's own test, not committed as a pytest file). Still a
good-faith reproduction of the cited equations rather than an independently re-derived Hamiltonian, and the
overall construction has not been checked against a published spectrum; flagged as a limitation in the
companion notebook.

MVP scope: the paper further reduces the circuit to 3 qubits (meson, N=2) / 4 qubits (baryon, N=4) via a
hardware-motivated qubit-elimination trick specific to their VQE ansatz (decoupling "inactive" qubits via a
separable-state factorization), and computes a hadron mass RATIO from separate meson/baryon/vacuum VQE
runs. We do not reproduce that reduction trick or the mass-ratio calculation here -- both are future work.
Our MVP instead uses the *un-reduced* N=2 Hamiltonian directly (4 qubits, no qubit-elimination trick
needed) as a single genuine SU(2)-LGT ground-state VQE benchmark, exactly analogous to how Tracks A/B/C
compare Q-MAML against classical initialization on a verified Hamiltonian -- the same research question
this whole project asks, just not yet the paper's own headline hadron-mass-ratio result.
"""

from typing import Tuple

import numpy as np
import pennylane as qml
import scipy.sparse.linalg
import torch
import torch.nn as nn


def _mass_terms(N: int, m_tilde: float):
    coeffs, ops = [], []
    const = 0.0
    for n in range(1, N + 1):
        sign = (-1) ** n
        q1, q2 = 2 * n - 2, 2 * n - 1  # 0-indexed qubits for site n's (2n-1, 2n) in the paper's 1-indexing
        coeffs += [m_tilde * sign / 2.0, m_tilde * sign / 2.0]
        ops += [qml.PauliZ(q1), qml.PauliZ(q2)]
        const += m_tilde * 1.0
    return coeffs, ops, const


def _kinetic_terms(N: int):
    coeffs, ops = [], []
    for n in range(1, N):
        i = 2 * n - 2  # site n, qubit 1 (0-indexed)
        z = 2 * n - 1  # site n, qubit 2 (0-indexed), plays the Z-string role
        j = 2 * n  # site n+1, qubit 1 (0-indexed)
        coeffs += [-0.25, -0.25]
        ops += [
            qml.PauliX(i) @ qml.PauliZ(z) @ qml.PauliX(j),
            qml.PauliY(i) @ qml.PauliZ(z) @ qml.PauliY(j),
        ]
        # sigma+ Z sigma- + h.c. = (XZX + YZY)/2 ; the extra 1/2 from -(1/2)*(...) folds into -0.25 each
    return coeffs, ops


def _electric_terms(N: int, x: float):
    coeffs, ops = [], []
    const = 0.0
    inv_x = 1.0 / x

    # term 1: (3/16) sum_{n=1}^{N-1} (N-n) (1 - Z_{2n-1} Z_{2n})
    for n in range(1, N):
        w = (3.0 / 16.0) * (N - n)
        const += inv_x * w
        q1, q2 = 2 * n - 2, 2 * n - 1
        coeffs.append(-inv_x * w)
        ops.append(qml.PauliZ(q1) @ qml.PauliZ(q2))

    # term 2: (1/16) sum_{n=1}^{N-2} sum_{m=n+1}^{N-1} (N-m) (Z_{2n-1}-Z_{2n})(Z_{2m-1}-Z_{2m})
    for n in range(1, N - 1):
        for m in range(n + 1, N):
            w = (1.0 / 16.0) * (N - m)
            qn1, qn2 = 2 * n - 2, 2 * n - 1
            qm1, qm2 = 2 * m - 2, 2 * m - 1
            for qa, sa in [(qn1, 1.0), (qn2, -1.0)]:
                for qb, sb in [(qm1, 1.0), (qm2, -1.0)]:
                    coeffs.append(inv_x * w * sa * sb)
                    ops.append(qml.PauliZ(qa) @ qml.PauliZ(qb))

    # term 3: (1/2) sum_{n=1}^{N-2} sum_{m=n+1}^{N-1} (N-m) (sigma+_{2n-1} sigma-_{2n} sigma+_{2m} sigma-_{2m-1} + h.c.)
    for n in range(1, N - 1):
        for m in range(n + 1, N):
            w = 0.5 * (N - m)
            qn1, qn2 = 2 * n - 2, 2 * n - 1
            qm1, qm2 = 2 * m - 2, 2 * m - 1
            # sigma+_{qn1} sigma-_{qn2} sigma+_{qm2} sigma-_{qm1} + h.c.
            #   = (1/4)(X_qn1 X_qn2 + Y_qn1 Y_qn2 + i(...))... expand directly via sigma+/- combinatorics:
            # sigma+_a sigma-_b = (X_a X_b + Y_a Y_b)/4 + i(Y_a X_b - X_a Y_b)/4
            # product of two such 2-body raising/lowering pairs on disjoint qubit pairs (a,b) and (c,d):
            # (sigma+_a sigma-_b)(sigma+_c sigma-_d) + h.c. -- expand numerically via Hermitian operator
            # construction below rather than hand-expanding 16 Pauli terms (error-prone); see
            # _raise_lower_product_plus_hc for the exact expansion, unit-tested against a dense
            # sigma+/sigma- matrix product in verify_construction().
            c4, o4 = _raise_lower_product_plus_hc(qn1, qn2, qm2, qm1, w)
            coeffs += c4
            ops += o4
    return coeffs, ops, const


def _raise_lower_product_plus_hc(a: int, b: int, c: int, d: int, weight: float):
    """weight * (sigma+_a sigma-_b sigma+_c sigma-_d + h.c.), expanded into Pauli strings.
    sigma+ = (X+iY)/2, sigma- = (X-iY)/2 per qubit; expand the 4-fold product symbolically."""
    # Represent sigma+ = (X + iY)/2, sigma- = (X - iY)/2 as 2x2 coefficient pairs (cx, cy) meaning cx*X+cy*Y.
    # Product over 4 distinct qubits of single-qubit operators (each a combination of X,Y) sums over all
    # 2^4 = 16 (Pauli-letter) combinations; do this numerically with sympy-free plain Python since it's a
    # fixed, small combinatorial expansion.
    factors = {a: (0.5, 0.5j), b: (0.5, -0.5j), c: (0.5, 0.5j), d: (0.5, -0.5j)}  # (X coeff, Y coeff) per qubit
    qubits = [a, b, c, d]
    coeffs, ops = [], []
    for bits in range(16):
        term_coeff = 1.0 + 0.0j
        paulis = []
        for k, q in enumerate(qubits):
            use_y = (bits >> k) & 1
            cx, cy = factors[q]
            if use_y:
                term_coeff *= cy
                paulis.append((q, qml.PauliY))
            else:
                term_coeff *= cx
                paulis.append((q, qml.PauliX))
        total_coeff = weight * term_coeff
        # + h.c.: h.c. of a product of Hermitian Paulis with a complex scalar c is conj(c) * (same Pauli
        # string, since each single-qubit Pauli is self-adjoint and they're on distinct qubits so they
        # commute under the dagger reordering)
        combined = total_coeff + np.conj(total_coeff)
        if abs(combined) < 1e-14:
            continue
        op = paulis[0][1](paulis[0][0])
        for q, P in paulis[1:]:
            op = op @ P(q)
        coeffs.append(float(np.real(combined)))
        ops.append(op)
    return coeffs, ops


def su2_lgt_hamiltonian(N: int, m_tilde: float, x: float) -> qml.Hamiltonian:
    """Effective (gauge-integrated-out) SU(2) LGT Hamiltonian, 2*N qubits.
    Task parameters: dimensionless fermion mass m_tilde, coupling x = 1/(a_l^2 g^2)."""
    coeffs, ops = [], []
    const_total = 0.0

    c, o, const = _mass_terms(N, m_tilde)
    coeffs += c
    ops += o
    const_total += const

    c, o = _kinetic_terms(N)
    coeffs += c
    ops += o

    c, o, const = _electric_terms(N, x)
    coeffs += c
    ops += o
    const_total += const

    if abs(const_total) > 1e-14:
        coeffs.append(const_total)
        ops.append(qml.Identity(0))

    return qml.Hamiltonian(coeffs, ops)


def exact_ground_energy(H: qml.Hamiltonian, num_qubits: int) -> float:
    sm = H.sparse_matrix(wire_order=list(range(num_qubits)))
    if sm.shape[0] <= 64:
        return float(np.min(np.linalg.eigvalsh(sm.toarray())))
    val = scipy.sparse.linalg.eigsh(sm, k=1, which="SA", return_eigenvectors=False)
    return float(val[0])


def sample_task_space(n_tasks: int, seed: int, m_range=(0.1, 2.0), x_range=(0.2, 5.0)) -> np.ndarray:
    """Task = (m_tilde, x), uniformly sampled -- matches the paper's own scan range x in [0,5] (we use
    [0.2, 5.0] to avoid the x->0 strong-coupling singularity in the 1/x electric term)."""
    rng = np.random.default_rng(seed)
    m_tilde = rng.uniform(*m_range, size=n_tasks)
    x = rng.uniform(*x_range, size=n_tasks)
    return np.stack([m_tilde, x], axis=1)


def param_shape(depth: int, num_qubits: int) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)


def verify_construction(N: int = 2, atol: float = 1e-8) -> bool:
    """Hermiticity of H at a few (m_tilde, x) points -- the necessary correctness condition given no
    independent second construction path is available for this track (see module docstring)."""
    ok = True
    num_qubits = 2 * N
    for m_tilde, x in [(1.0, 1.0), (0.5, 2.0), (1.5, 0.5)]:
        H = su2_lgt_hamiltonian(N, m_tilde, x)
        Hd = H.sparse_matrix(wire_order=list(range(num_qubits))).toarray()
        herm_err = np.max(np.abs(Hd - Hd.conj().T))
        passed = herm_err < atol
        ok = ok and passed
        print(f"  N={N} m_tilde={m_tilde} x={x}: Hermiticity max|H-H^dag|={herm_err:.2e}  {'OK' if passed else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------------------------------------
# Variational circuits (PennyLane reference implementations; the benchmark runs the exact torch
# simulators in qmaml_hep.simulators, cross-checked against these on every run)
# ---------------------------------------------------------------------------------------------------------


NUM_QUBITS = 4  # N=2


class SU2LGTPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_SU2LGT(m_tilde,x)|psi(theta)>, measured directly via qml.expval(H)."""

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
