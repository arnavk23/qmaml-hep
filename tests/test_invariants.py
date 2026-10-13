"""Symmetries the structured ansätze are built to respect, checked on the exact simulators for random angles."""

import numpy as np
import pytest
import torch

from qmaml_hep.tracks import TRACKS, make_fast


def _pauli_rot_state(sim, w):
    psi = sim.psi0.clone()
    for layer in range(w.shape[0]):
        for k, P in enumerate(sim.gens):
            psi = torch.cos(w[layer, k] / 2) * psi - 1j * torch.sin(w[layer, k] / 2) * (P @ psi)
    return psi


def _uccsd_state(sim, w):
    w = w.reshape(sim.depth, sim.n_exc)
    psi = sim.psi0
    for layer in range(sim.depth):
        for k, c, G, G2 in sim.mats:
            psi = psi + torch.sin(c * w[layer, k]) * (G @ psi) + (1 - torch.cos(c * w[layer, k])) * (G2 @ psi)
    return psi


def test_z2_hva_preserves_gauss_law():
    """The Hamiltonian-variational ansatz is built from gauge-invariant generators, so every site's Gauss operator
    keeps the eigenvalue it has on the vacuum, for any angles."""
    from qmaml_hep.systems.z2lgt import MATTER, NUM_QUBITS, gauss_operator

    tr = TRACKS["B_z2_hva"]()
    make_fast(tr)
    sim = tr["pqc"]("all")
    w = torch.rand(tr["shape"]("all"), dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 2 * np.pi
    psi0, psi = sim.psi0, _pauli_rot_state(sim, w)
    for site in MATTER:
        G = torch.tensor(gauss_operator(site).matrix(wire_order=range(NUM_QUBITS)), dtype=torch.complex128)
        g0 = torch.real(torch.vdot(psi0, G @ psi0))
        assert abs(abs(float(g0)) - 1.0) < 1e-12, "the vacuum is a Gauss-law eigenstate"
        assert float(torch.linalg.vector_norm(G @ psi - g0 * psi)) < 1e-10


@pytest.mark.parametrize("nucleus", ["deuteron", "triton", "helium3"])
def test_uccsd_conserves_proton_and_neutron_number(nucleus):
    from qmaml_hep.simulators import FastUCCSD
    from qmaml_hep.systems.nuclear import N_PROTON_ORBITALS, NUCLEI, NUM_QUBITS, _excitations_for

    s_wires, d_wires, init_state = _excitations_for(nucleus)
    sim = FastUCCSD(NUM_QUBITS, s_wires, d_wires, init_state, depth=2)
    w = torch.rand(2 * sim.n_exc, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 2 * np.pi
    prob = (_uccsd_state(sim, w).abs() ** 2).numpy()
    n_p, n_n = NUCLEI[nucleus]
    for b in np.nonzero(prob > 1e-14)[0]:
        bits = [(int(b) >> (NUM_QUBITS - 1 - q)) & 1 for q in range(NUM_QUBITS)]
        assert sum(bits[:N_PROTON_ORBITALS]) == n_p and sum(bits[N_PROTON_ORBITALS:]) == n_n
    assert abs(prob.sum() - 1.0) < 1e-12


def test_susy_qm_ground_energy_converges_to_zero_with_the_cutoff():
    """Unbroken supersymmetry (harmonic superpotential) has E0 = 0. Truncating the boson to 2**n_b levels breaks
    it slightly; the residual must shrink with the cutoff and is far below the 1e-3 threshold at the n_b = 3 used."""
    from qmaml_hep.systems.susyqm import N_B_DEFAULT, exact_ground_energy, susyqm_hamiltonian

    e0 = [abs(exact_ground_energy(susyqm_hamiltonian("HO", n_b=nb, m=1.3, g=0.0, mu=0.0))) for nb in (2, 3, 4, 5)]
    assert e0[0] > e0[1] > e0[2] and e0[3] < 1e-12
    assert e0[N_B_DEFAULT - 2] < 1e-6


def test_lmg_spectrum_is_symmetric_under_epsilon_sign_at_v_zero():
    from qmaml_hep.systems.lmg import lmg_hamiltonian

    a = np.linalg.eigvalsh(lmg_hamiltonian(1.1, 0.0))
    b = np.linalg.eigvalsh(lmg_hamiltonian(-1.1, 0.0))
    assert np.allclose(a, -b[::-1])
