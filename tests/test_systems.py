"""Physics checks for every Hamiltonian: the construction checks each system module ships with, plus
Hermiticity and exact-diagonalization consistency for the systems built as PennyLane operators."""

import importlib

import numpy as np
import pytest
import torch

VERIFY = {
    "lmg": "verify_construction",
    "yukawa": "verify_construction",
    "d8lgt": "verify_construction",
    "neutrino": "verify_construction",
    "nuclear": "verify_construction",
    "scalar": "verify_construction",
    "susyqm": "verify_construction",
    "su2lgt": "verify_construction",
    "z2lgt": "verify_gauge_invariance",
}


@pytest.mark.parametrize("system", sorted(VERIFY))
def test_construction_check(system):
    mod = importlib.import_module(f"qmaml_hep.systems.{system}")
    assert bool(getattr(mod, VERIFY[system])())


def test_susyqm_double_well_against_grid_solution():
    from qmaml_hep.systems.susyqm import verify_double_well

    assert bool(verify_double_well())


@pytest.mark.parametrize("n", [4, 6])
def test_schwinger_hermitian_and_ground_energy(n):
    from qmaml_hep.systems.schwinger import exact_ground_energy, schwinger_hamiltonian

    H = schwinger_hamiltonian(n, 0.7, 1.3).sparse_matrix(wire_order=range(n)).toarray()
    assert np.abs(H - H.conj().T).max() < 1e-12
    assert abs(exact_ground_energy(schwinger_hamiltonian(n, 0.7, 1.3), n) - np.linalg.eigvalsh(H)[0]) < 1e-9


def test_task_sampling_is_seeded():
    from qmaml_hep.tracks import TRACKS

    for key in ("A_schwinger", "F_susyqm", "J_lmg"):
        tr = TRACKS[key]()
        assert tr["sample"](5, 3) == tr["sample"](5, 3)
        assert tr["sample"](5, 3) != tr["sample"](5, 4)


def test_heisenberg_fast_ansatz_matches_pennylane():
    """The closed-form IsingXX/YY/ZZ brick simulator against the PennyLane circuit, energy and gradient."""
    from qmaml_hep.simulators import heisenberg_energy, ising_brick_state
    from qmaml_hep.systems.heisenberg import HeisenbergPQC

    n, depth, J = 4, 3, (0.7, -1.1, 0.4)
    w1 = torch.rand(depth, n - 1, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 6.28
    w1.requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    e_ref = HeisenbergPQC(n, depth, use_lightning=False).cost(w1, *J)
    e_fast = heisenberg_energy(ising_brick_state(w2, n), n, J)
    e_ref.backward(), e_fast.backward()
    assert abs(float(e_ref) - float(e_fast)) < 1e-10
    assert float((w1.grad - w2.grad).abs().max()) < 1e-10
