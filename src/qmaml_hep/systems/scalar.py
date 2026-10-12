"""Track C: phi^4 lattice scalar field theory (Li, Macridin, Mrenna & Spentzouris,
"Simulating scalar field theories on quantum computers with limited resources",
arXiv:2210.07985, FERMILAB-PUB-22-757-QIS). Bosonic, no gauge field, no fermion
encoding at all -- structurally the most different of this project's three HEP
Hamiltonian tracks from the Schwinger model (Track A, systems/schwinger.py)
and the Z2 lattice gauge theory (Track B, systems/z2lgt.py). Its physical
task parameters (bare mass^2, coupling) sweep the theory across a genuine
symmetric/broken-symmetry phase transition (paper Section II, Fig. 1), making
"does a Learner trained on one side of the transition generalize to the other"
a well-posed, new question -- flagged as future work in the companion notebook,
not attempted in this MVP pass.

Dimensionless lattice Hamiltonian (paper Eq. 7, external field f0 = 0 here --
the symmetric-vs-broken-symmetry study does not require it, paper Section II):
  H = sum_j [ (1/2) Pi_j^2 + (1/2) m0^2 Phi_j^2 + (lambda0/4!) Phi_j^4 ]
      + (1/2) sum_j (Phi_{j+1} - Phi_j)^2                      [open boundary, 1+1D]
Task parameters: bare mass^2 (m0_sq) and quartic coupling (lambda0) -- the
paper's own free parameters (its Fig. 1 sweeps m0^2/lambda0^(2/3) at fixed
lambda0=1; very negative values approach the broken-symmetry regime).

Qubit encoding (paper Section III.B, "discretized field amplitude basis"): each
site gets n_q qubits, giving N_phi = 2^n_q basis states for the field amplitude
Phi_j at that site (paper Eq. 22-24). Phi_j is diagonal in this basis (Eq. 24);
Pi_j is diagonal in the discrete-Fourier-conjugate basis (Eq. 10, 13, 16-18):
  Pi_j = mu_disc * F_j @ Phi_j @ F_j^dagger
where F_j is the paper's finite (off-centered) discrete Fourier transform
(Eq. 13) and mu_disc is a fixed *representation* parameter (paper calls it "the
boson mass mu" but it only tunes discretization fineness/accuracy -- distinct
from the physical bare mass^2 m0_sq that indexes the task distribution; the
paper makes this same distinction, Section III.A). mu_disc=1.0 is fixed here,
not swept.

This module builds H, Phi_j, Pi_j directly as dense matrices (not Pauli-string
decompositions) since we only need static energy expectation values for VQE, not
the paper's Trotterized *time evolution* circuits (its Section III.C); the PQC's
cost is <psi(theta)| H |psi(theta)> via qml.Hermitian(H, wires=...), exactly
analogous to Tracks A/B's qml.Hamiltonian-based qml.expval(H).

MVP scale: N_SITES=2, n_q=3 (N_phi=8 levels/site) -> 6 qubits total, matching
Track A's minimal L=2 qubit count. This is one of the paper's own worked toy
sizes (Fig. 1's "2 sites" curve), though the paper does not fix n_q for that
figure -- our n_q=3 truncation is our own MVP choice, stated explicitly rather
than implied.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

N_SITES = 2


N_Q_PER_SITE = 3


N_PHI = 2**N_Q_PER_SITE  # 8 field-amplitude levels per site


NUM_QUBITS = N_SITES * N_Q_PER_SITE  # 6


MU_DISC = 1.0  # fixed representation parameter (not a task param)


def _site_field_operator(n_phi: int, mu_disc: float) -> np.ndarray:
    """Phi (Eq. 9, 11): diagonal, eigenvalues Delta_phi * (alpha - (n_phi-1)/2)."""
    delta_phi = np.sqrt(2.0 * np.pi / (n_phi * mu_disc))
    alphas = np.arange(n_phi)
    eigvals = delta_phi * (alphas - (n_phi - 1) / 2.0)
    return np.diag(eigvals).astype(complex)


def _dft_matrix(n_phi: int) -> np.ndarray:
    """Off-centered finite Fourier transform (Eq. 13)."""
    idx = np.arange(n_phi) - (n_phi - 1) / 2.0
    F = np.exp(1j * 2.0 * np.pi / n_phi * np.outer(idx, idx)) / np.sqrt(n_phi)
    return F


def _site_conjugate_operator(Phi: np.ndarray, mu_disc: float) -> np.ndarray:
    """Pi = mu_disc * F @ Phi @ F^dagger (Eq. 10)."""
    n_phi = Phi.shape[0]
    F = _dft_matrix(n_phi)
    return mu_disc * (F @ Phi @ F.conj().T)


def _embed(op: np.ndarray, site: int, n_sites: int, n_phi: int) -> np.ndarray:
    """Kron site `op` into the full N_SITES-register Hilbert space."""
    mats = [np.eye(n_phi, dtype=complex)] * n_sites
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def scalar_field_hamiltonian(
    m0_sq: float, lambda0: float, n_sites: int = N_SITES, n_q_per_site: int = N_Q_PER_SITE, mu_disc: float = MU_DISC
) -> np.ndarray:
    """Dense Hermitian matrix, dimension (2**n_q_per_site)**n_sites.
    Task parameters: bare mass^2 m0_sq, quartic coupling lambda0."""
    n_phi = 2**n_q_per_site
    dim = n_phi**n_sites
    Phi = _site_field_operator(n_phi, mu_disc)
    Pi = _site_conjugate_operator(Phi, mu_disc)
    Phi2 = Phi @ Phi
    Phi4 = Phi2 @ Phi2
    Pi2 = Pi @ Pi

    H = np.zeros((dim, dim), dtype=complex)
    Phi_embedded = []
    for j in range(n_sites):
        H += _embed(0.5 * Pi2, j, n_sites, n_phi)
        H += _embed(0.5 * m0_sq * Phi2, j, n_sites, n_phi)
        H += _embed((lambda0 / 24.0) * Phi4, j, n_sites, n_phi)
        Phi_embedded.append(_embed(Phi, j, n_sites, n_phi))
    for j in range(n_sites - 1):
        diff = Phi_embedded[j + 1] - Phi_embedded[j]
        H += 0.5 * (diff @ diff)
    return H


def exact_ground_energy(H: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(H)))


def sample_task_space(n_tasks: int, seed: int, m0_sq_range=(-2.5, 1.0), lambda0_range=(0.5, 1.5)) -> np.ndarray:
    """Task = (m0_sq, lambda0). m0_sq sweeps from the broken-symmetry-tending
    regime (very negative) to the symmetric regime (positive), per paper Fig. 1."""
    rng = np.random.default_rng(seed)
    m0_sq = rng.uniform(*m0_sq_range, size=n_tasks)
    lambda0 = rng.uniform(*lambda0_range, size=n_tasks)
    return np.stack([m0_sq, lambda0], axis=1)


def param_shape(depth: int, num_qubits: int = NUM_QUBITS) -> Tuple[int, int, int]:
    return (depth, num_qubits, 3)


def verify_construction(m0_sq=-1.0, lambda0=1.0, atol=1e-8) -> bool:
    """(1) Phi, Pi, H Hermitian to machine precision -- exact requirement.
    (2) Pi's eigenvalues match the paper's analytic kappa_beta formula (Eq. 16-17)
    exactly -- exact requirement, confirms the DFT-conjugation construction.
    (3) Reconstructing H_ho = Pi^2/2 + mu^2 Phi^2/2 recovers the harmonic-
    oscillator spectrum n+1/2 accurately for the lowest ~N_phi/2 levels, degrading
    near the truncation edge -- this is what the paper's Eq. 19 O(eps) claim
    actually means (accurate on the *low-energy* subspace, not literally every
    field-amplitude basis state), so a naive [Phi,Pi]=i check on "middle" basis
    indices is the wrong diagnostic and was dropped after being caught giving a
    large, non-decreasing-with-N_phi residual here -- the QHO-spectrum check
    below is the correct one and passes cleanly."""
    Phi = _site_field_operator(N_PHI, MU_DISC)
    Pi = _site_conjugate_operator(Phi, MU_DISC)
    ok = True

    herm_err_phi = np.max(np.abs(Phi - Phi.conj().T))
    herm_err_pi = np.max(np.abs(Pi - Pi.conj().T))
    print(f"  Hermiticity: max|Phi-Phi^dag|={herm_err_phi:.2e}  max|Pi-Pi^dag|={herm_err_pi:.2e}")
    ok = ok and herm_err_phi < atol and herm_err_pi < atol

    eig_pi = np.sort(np.real(np.linalg.eigvalsh(Pi)))
    delta_kappa = np.sqrt(2.0 * np.pi * MU_DISC / N_PHI)
    expected = delta_kappa * (np.arange(N_PHI) - (N_PHI - 1) / 2.0)
    eig_err = np.max(np.abs(eig_pi - expected))
    print(f"  Pi eigenvalues vs. analytic kappa_beta: max err={eig_err:.2e}")
    ok = ok and eig_err < atol

    H_ho = 0.5 * Pi @ Pi + 0.5 * (MU_DISC**2) * (Phi @ Phi)
    spectrum = np.sort(np.real(np.linalg.eigvalsh(H_ho)))
    n_check = N_PHI // 2
    analytic = MU_DISC * (np.arange(n_check) + 0.5)
    rel_err = np.max(np.abs(spectrum[:n_check] - analytic) / analytic)
    print(f"  QHO spectrum, lowest {n_check} levels vs. analytic n+1/2: max rel err={rel_err:.2%}")
    ok = ok and rel_err < 0.05

    H = scalar_field_hamiltonian(m0_sq, lambda0)
    herm_err_H = np.max(np.abs(H - H.conj().T))
    print(f"  H({m0_sq},{lambda0}) Hermiticity: max|H-H^dag|={herm_err_H:.2e}")
    ok = ok and herm_err_H < atol
    return ok


# ---------------------------------------------------------------------------------------------------------
# Variational circuits (PennyLane reference implementations; the benchmark runs the exact torch
# simulators in qmaml_hep.simulators, cross-checked against these on every run)
# ---------------------------------------------------------------------------------------------------------


class ScalarFieldPQC(nn.Module):
    """Hardware-efficient (StronglyEntanglingLayers) ansatz PQC whose cost is
    <psi(theta)|H_phi4(m0_sq,lambda0)|psi(theta)>, measured via qml.Hermitian
    since H is a dense field-amplitude-basis matrix, not a Pauli sum."""

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
