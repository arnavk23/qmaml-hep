"""Molecule Hamiltonian VQE task space, matching the Q-MAML paper's second experimental
setting (Lee, Cho & Kim 2025, arXiv:2501.05906, "Molecule Hamiltonian" section) -- a second,
independent calibration of our system beyond the Heisenberg XYZ setting in
systems/heisenberg.py.

Task space: H2 molecule at different bond lengths (paper: "12 distinct bond lengths"), using
PennyLane's built-in Hartree-Fock via qml.qchem (no external quantum-chemistry package needed).
Task descriptor: the Hamiltonian's Pauli-term coefficient vector C (via `H.terms()`), per the
paper's Task Space Definition -- for a fixed molecule/active space the Pauli-string basis is
identical across bond lengths, only coefficients vary, so no zero-padding is needed here (unlike
the paper's general case across differently-sized molecules).

Ansatz: paper specifies 7 layers of StronglyEntanglingLayers (Schuld et al. 2020) -- we can use
this exactly, since it's a standard PennyLane template (unlike the Heisenberg XYZ ansatz, whose
exact appendix circuit we had to reconstruct).
"""

from typing import Tuple

import numpy as np
import pennylane as qml
import scipy.sparse.linalg
import torch
import torch.nn as nn


def h2_hamiltonian(bond_length: float):
    symbols = ["H", "H"]
    coords = np.array([0.0, 0.0, -bond_length / 2, 0.0, 0.0, bond_length / 2])
    H, num_qubits = qml.qchem.molecular_hamiltonian(symbols, coords)
    return H, num_qubits


def hamiltonian_coeffs(H) -> torch.Tensor:
    coeffs, _ops = H.terms()
    return torch.tensor(np.array(coeffs), dtype=torch.float64)


def exact_ground_energy(H, num_qubits: int) -> float:
    sm = H.sparse_matrix(wire_order=list(range(num_qubits)))
    if sm.shape[0] <= 64:
        return float(np.min(np.linalg.eigvalsh(sm.toarray())))
    val = scipy.sparse.linalg.eigsh(sm, k=1, which="SA", return_eigenvectors=False)
    return float(val[0])


def sample_bond_lengths(n_tasks: int, seed: int, lo: float = 0.5, hi: float = 2.5) -> np.ndarray:
    """Paper samples 12 distinct bond lengths; we default to the same range (Angstrom)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=n_tasks)


class MoleculePQC(nn.Module):
    """StronglyEntanglingLayers PQC whose cost is <psi(theta)|H_mol|psi(theta)>, measured directly
    via qml.expval(H) -- PennyLane handles the weighted Pauli-string sum internally."""

    def __init__(self, num_qubits: int, depth: int, use_lightning: bool = True):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
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
        @qml.qnode(self.dev, interface="torch", diff_method=self._diff_method)
        def circuit(weights):
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.expval(H)

        return circuit

    def cost(self, weights: torch.Tensor, H) -> torch.Tensor:
        circuit = self.make_circuit(H)
        return circuit(weights.to(torch.float64))


def param_shape(num_qubits: int, depth: int) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)
