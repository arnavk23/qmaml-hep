"""Property-based tests of the simulators: identities that must hold for every parameter value, not just the
handful of points the PennyLane cross-checks sample."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qmaml_hep.simulators import FastHEA, sel_expvals, sel_state

ANGLES = st.floats(-4 * np.pi, 4 * np.pi, allow_nan=False, allow_infinity=False)


def _weights(n, depth):
    return arrays(np.float64, (depth, n, 3), elements=ANGLES).map(torch.from_numpy)


def _random_hermitian(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    return (A + A.conj().T) / 2


@settings(max_examples=60, deadline=None)
@given(w=_weights(3, 2))
def test_state_is_normalized(w):
    psi = sel_state(w, 3)
    assert abs(float(torch.linalg.vector_norm(psi)) - 1.0) < 1e-12


@settings(max_examples=60, deadline=None)
@given(w=_weights(3, 2), seed=st.integers(0, 10_000))
def test_energy_lies_within_the_spectrum(w, seed):
    H = _random_hermitian(3, seed)
    lam = np.linalg.eigvalsh(H)
    E = float(FastHEA(3).cost(w, H))
    assert lam[0] - 1e-10 <= E <= lam[-1] + 1e-10


@settings(max_examples=40, deadline=None)
@given(w=_weights(2, 2), k=st.integers(0, 11))
def test_energy_is_2pi_periodic_in_every_angle(w, k):
    H = _random_hermitian(2, 0)
    shifted = w.clone().reshape(-1)
    shifted[k] += 2 * np.pi
    sim = FastHEA(2)
    assert abs(float(sim.cost(w, H)) - float(sim.cost(shifted.reshape(w.shape), H))) < 1e-10


@settings(max_examples=30, deadline=None)
@given(w=_weights(3, 2), seed=st.integers(0, 10_000))
def test_parameter_shift_rule_equals_autograd(w, seed):
    """Every Rot angle is a Pauli rotation exp(-i a P / 2), so dE/da = [E(a + pi/2) - E(a - pi/2)] / 2 exactly."""
    H = _random_hermitian(3, seed)
    sim = FastHEA(3)
    w = w.clone().requires_grad_(True)
    sim.cost(w, H).backward()
    flat = w.detach().reshape(-1)
    for k in range(0, flat.numel(), 5):
        plus, minus = flat.clone(), flat.clone()
        plus[k] += np.pi / 2
        minus[k] -= np.pi / 2
        shift = (float(sim.cost(plus.reshape(w.shape), H)) - float(sim.cost(minus.reshape(w.shape), H))) / 2
        assert abs(shift - float(w.grad.reshape(-1)[k])) < 1e-9


@settings(max_examples=40, deadline=None)
@given(
    x=arrays(np.float64, (5, 4), elements=st.floats(-50, 50, allow_nan=False)).map(torch.from_numpy),
    w=_weights(4, 2),
)
def test_classifier_expectations_are_bounded(x, w):
    z = sel_expvals(x, w)
    assert z.shape == (5, 4)
    assert float(z.abs().max()) <= 1 + 1e-12


def test_identity_circuit_leaves_zero_state():
    # Rot(0, 0, 0) is the identity; the CNOT ring maps |0...0> to itself
    psi = sel_state(torch.zeros(3, 4, 3, dtype=torch.float64), 4)
    expected = torch.zeros(16, dtype=torch.complex128)
    expected[0] = 1
    assert torch.allclose(psi, expected)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_single_qubit_and_edge_sizes(n):
    w = torch.rand(2, n, 3, dtype=torch.float64)
    assert abs(float(torch.linalg.vector_norm(sel_state(w, n))) - 1.0) < 1e-12
