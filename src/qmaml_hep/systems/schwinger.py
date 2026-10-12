"""Lattice Schwinger model Hamiltonian (1+1D QED), single-flavor staggered-fermion
formulation, gauge field eliminated via Gauss's law (Kogut-Susskind / Jordan-Wigner
mapping to qubits). Standard form (e.g. Kokail et al. 2019, Nature 569, 355;
Martinez et al. 2016; see also arXiv:2504.20824 for the 2-flavor variant this
single-flavor version simplifies):

  H = 1/(2a) * sum_{j=0}^{n-2} (sigma+_j sigma-_{j+1} + sigma+_{j+1} sigma-_j)
    + m0/2  * sum_{j=0}^{n-1} (-1)^j sigma^z_j
    + (a g^2)/8 * sum_{j=0}^{n-2} ( sum_{k=0}^{j} (sigma^z_k + (-1)^k) )^2

n qubits = 2L staggered sites (L physical fermion sites; even j = fermion,
odd j = antifermion). Task parameters: bare mass m0, coupling g (lattice
spacing a fixed to 1, the standard convention in this literature).

The Coulomb/electric term (third line) is built by expanding the square
programmatically (not hand-derived) to avoid algebraic slips: for each j,
L_j = sum_{k<=j} Z_k + b_j  (b_j = sum_{k<=j} (-1)^k, a classical scalar), so
L_j^2 = (j+1) + 2*b_j*sum_{k<=j} Z_k + 2*sum_{k<l<=j} Z_k Z_l  (since Z_k^2=I).
Verified against a brute-force dense-matrix construction for n<=8 -- see
tests/test_systems.py.
"""

from typing import Tuple

import numpy as np
import pennylane as qml
import scipy.sparse.linalg
import torch
import torch.nn as nn


def schwinger_hamiltonian(num_qubits: int, m0: float, g: float, a: float = 1.0):
    """num_qubits must be even (n = 2L staggered sites)."""
    n = num_qubits
    coeffs, ops = [], []

    # Hopping term: (1/2a) * sum_j (XX + YY)/2  [since sigma+ sigma- + h.c. = (XX+YY)/2]
    hop = 1.0 / (2.0 * a)
    for j in range(n - 1):
        coeffs += [hop / 2.0, hop / 2.0]
        ops += [
            qml.PauliX(j) @ qml.PauliX(j + 1),
            qml.PauliY(j) @ qml.PauliY(j + 1),
        ]

    # Mass term: (m0/2) * sum_j (-1)^j Z_j
    for j in range(n):
        coeffs.append((m0 / 2.0) * ((-1) ** j))
        ops.append(qml.PauliZ(j))

    # Electric/Coulomb term: (a g^2 / 8) * sum_j L_j^2, expanded programmatically
    coul = a * g**2 / 8.0
    # accumulate into dicts to merge repeated terms across different j before
    # handing to qml.Hamiltonian (keeps the term count small and correct)
    const_total = 0.0
    z_coeffs = {k: 0.0 for k in range(n)}
    zz_coeffs = {}
    for j in range(n - 1):
        b_j = sum((-1) ** k for k in range(j + 1))
        const_total += coul * (j + 1 + b_j**2)
        for k in range(j + 1):
            z_coeffs[k] += coul * 2 * b_j
        for k in range(j + 1):
            for l in range(k + 1, j + 1):
                zz_coeffs[(k, l)] = zz_coeffs.get((k, l), 0.0) + coul * 2

    if const_total != 0.0:
        coeffs.append(const_total)
        ops.append(qml.Identity(0))
    for k, c in z_coeffs.items():
        if c != 0.0:
            coeffs.append(c)
            ops.append(qml.PauliZ(k))
    for (k, l), c in zz_coeffs.items():
        if c != 0.0:
            coeffs.append(c)
            ops.append(qml.PauliZ(k) @ qml.PauliZ(l))

    return qml.Hamiltonian(coeffs, ops)


def exact_ground_energy(H, num_qubits: int) -> float:
    sm = H.sparse_matrix(wire_order=list(range(num_qubits)))
    if sm.shape[0] <= 64:
        return float(np.min(np.linalg.eigvalsh(sm.toarray())))
    val = scipy.sparse.linalg.eigsh(sm, k=1, which="SA", return_eigenvectors=False)
    return float(val[0])


def sample_task_space(n_tasks: int, seed: int, m0_range=(0.0, 1.5), g_range=(0.3, 2.0)) -> np.ndarray:
    """Task = (m0, g), uniformly sampled over the given physical parameter ranges."""
    rng = np.random.default_rng(seed)
    m0 = rng.uniform(m0_range[0], m0_range[1], size=n_tasks)
    g = rng.uniform(g_range[0], g_range[1], size=n_tasks)
    return np.stack([m0, g], axis=1)


def param_shape(num_qubits: int, depth: int) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)


# ---------------------------------------------------------------------------------------------------------
# Variational circuits (PennyLane reference implementations; the benchmark runs the exact torch
# simulators in qmaml_hep.simulators, cross-checked against these on every run)
# ---------------------------------------------------------------------------------------------------------


class SchwingerPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_Schwinger(m0,g)|psi(theta)>, measured directly via qml.expval(H)."""

    def __init__(self, num_qubits: int, depth: int, use_lightning: bool = True):
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
