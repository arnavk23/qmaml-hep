"""Heisenberg XYZ VQE task space, matching the setting from the Q-MAML
paper (Lee, Cho & Kim 2025, arXiv:2501.05906) Eq. 4 / "Heisenberg XYZ Hamiltonian"
section. Used to calibrate our paper-faithful Q-MAML system
 against a known VQE setting before extending
the algorithm to HEP classification .

H_XYZ = -sum_{n=1}^{N-1} (Jx sx_n sx_{n+1} + Jy sy_n sy_{n+1} + Jz sz_n sz_{n+1} + h sz_n), h=0

Ansatz: the paper states blocks of IsingXX/IsingYY/IsingZZ gates but does not give
the exact circuit in the main text (appendix, not available to us) -- we use a
standard nearest-neighbor brick-style block (one IsingXX+IsingYY+IsingZZ triple per
edge per layer), which is consistent with the paper's description. This is a
documented limitation, not a silent guess.
"""

from typing import Tuple

import numpy as np
import pennylane as qml
import scipy.sparse.linalg
import torch
import torch.nn as nn


def heisenberg_xyz_hamiltonian(num_qubits: int, Jx: float, Jy: float, Jz: float, h: float = 0.0):
    coeffs, obs = [], []
    for n in range(num_qubits - 1):
        coeffs += [-Jx, -Jy, -Jz]
        obs += [
            qml.PauliX(n) @ qml.PauliX(n + 1),
            qml.PauliY(n) @ qml.PauliY(n + 1),
            qml.PauliZ(n) @ qml.PauliZ(n + 1),
        ]
    if h != 0.0:
        for n in range(num_qubits):
            coeffs.append(-h)
            obs.append(qml.PauliZ(n))
    return qml.Hamiltonian(coeffs, obs)


def exact_ground_energy(H, num_qubits: int) -> float:
    sm = H.sparse_matrix(wire_order=list(range(num_qubits)))
    if sm.shape[0] <= 64:
        return float(np.min(np.linalg.eigvalsh(sm.toarray())))
    val = scipy.sparse.linalg.eigsh(sm, k=1, which="SA", return_eigenvectors=False)
    return float(val[0])


def sample_task_space(n_tasks: int, seed: int, jrange: float = 3.0, step: float = 0.1) -> np.ndarray:
    """J^i = [Jx, Jy, Jz], each sampled from {-jrange, -jrange+step, ..., jrange} (paper's Task Space Def.)."""
    rng = np.random.default_rng(seed)
    grid = np.arange(-jrange, jrange + step / 2, step)
    return rng.choice(grid, size=(n_tasks, 3))


class HeisenbergPQC(nn.Module):
    """IsingXX/YY/ZZ brick-ansatz PQC whose cost is <psi(theta)|H_XYZ(J)|psi(theta)>."""

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

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def circuit(weights, coeffs, obs_wires_list, obs_kind_list):
            for l in range(depth):
                for n in range(num_qubits - 1):
                    qml.IsingXX(weights[l, n, 0], wires=[n, n + 1])
                    qml.IsingYY(weights[l, n, 1], wires=[n, n + 1])
                    qml.IsingZZ(weights[l, n, 2], wires=[n, n + 1])
            # Expectation value of H, computed as a weighted sum of Pauli-string expvals
            terms = []
            for kind, wires in zip(obs_kind_list, obs_wires_list):
                if kind == "XX":
                    terms.append(qml.expval(qml.PauliX(wires[0]) @ qml.PauliX(wires[1])))
                elif kind == "YY":
                    terms.append(qml.expval(qml.PauliY(wires[0]) @ qml.PauliY(wires[1])))
                elif kind == "ZZ":
                    terms.append(qml.expval(qml.PauliZ(wires[0]) @ qml.PauliZ(wires[1])))
                elif kind == "Z":
                    terms.append(qml.expval(qml.PauliZ(wires[0])))
            return terms

        self._circuit = circuit

    def cost(self, weights: torch.Tensor, Jx: float, Jy: float, Jz: float, h: float = 0.0) -> torch.Tensor:
        obs_kind, obs_wires, coeffs = [], [], []
        for n in range(self.num_qubits - 1):
            obs_kind += ["XX", "YY", "ZZ"]
            obs_wires += [(n, n + 1)] * 3
            coeffs += [-Jx, -Jy, -Jz]
        if h != 0.0:
            for n in range(self.num_qubits):
                obs_kind.append("Z")
                obs_wires.append((n,))
                coeffs.append(-h)
        vals = self._circuit(weights.to(torch.float64), coeffs, obs_wires, obs_kind)
        return sum(c * v for c, v in zip(coeffs, vals))


def param_shape(num_qubits: int, depth: int) -> Tuple[int, int, int]:
    return (depth, num_qubits - 1, 3)
