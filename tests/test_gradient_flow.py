"""Data types and gradient flow on every path from a classical network into a circuit.

These are the questions "what is the dtype of each tensor?" and "does the gradient actually reach the Learner?"
answered by assertion rather than by reading the code.
"""

import numpy as np
import pytest
import torch

from qmaml_hep.benchmark import W0_SCALE
from qmaml_hep.learner import Learner
from qmaml_hep.simulators import FastHEA, FastPQCModel, sel_expvals, sel_state
from qmaml_hep.systems.lmg import NUM_QUBITS as LMG_QUBITS
from qmaml_hep.systems.lmg import lmg_hamiltonian
from qmaml_hep.systems.schwinger import schwinger_hamiltonian


def test_learner_matches_paper_architecture():
    net = Learner(in_dim=2, out_shape=(3, 4, 3), hidden=256, n_hidden=2)
    sizes = [(m.in_features, m.out_features) for m in net.net if isinstance(m, torch.nn.Linear)]
    assert sizes == [(2, 256), (256, 256), (256, 36)]
    assert net(torch.zeros(2)).shape == (3, 4, 3)


def test_vqe_path_is_float64_end_to_end():
    """Algorithm 1 for the Hamiltonian tracks: descriptor -> Learner -> theta0 -> statevector -> energy."""
    torch.manual_seed(0)
    n, depth = 4, 2
    H = schwinger_hamiltonian(n, 0.5, 1.0)
    learner = Learner(2, (depth, n, 3)).double()
    x = torch.tensor([0.5, 1.0], dtype=torch.float64)

    theta = W0_SCALE * learner(x)
    psi = sel_state(theta, n)
    energy = FastHEA(n).cost(theta, H)

    assert all(p.dtype == torch.float64 for p in learner.parameters())
    assert theta.dtype == torch.float64
    assert psi.dtype == torch.complex128
    assert energy.dtype == torch.float64 and energy.dim() == 0
    assert abs(float(torch.linalg.vector_norm(psi.detach())) - 1.0) < 1e-12

    energy.backward()
    for name, p in learner.named_parameters():
        assert p.grad is not None, name
        assert p.grad.dtype == torch.float64, name
        assert torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().max()) > 0.0, f"no gradient reaches {name}"


def test_vqe_learner_gradient_matches_finite_differences():
    """dE/dW through the circuit equals a central finite difference, for one weight of each Learner layer."""
    torch.manual_seed(1)
    n, depth = 2, 2
    H = lmg_hamiltonian(1.2, 0.4)
    learner = Learner(2, (depth, LMG_QUBITS, 3), hidden=8).double()
    x = torch.tensor([1.2, 0.4], dtype=torch.float64)
    sim = FastHEA(n)

    def energy():
        return sim.cost(W0_SCALE * learner(x), H)

    energy().backward()
    eps = 1e-6
    for p in learner.parameters():
        idx = tuple(0 for _ in p.shape)
        with torch.no_grad():
            p[idx] += eps
            e_plus = float(energy())
            p[idx] -= 2 * eps
            e_minus = float(energy())
            p[idx] += eps
        fd = (e_plus - e_minus) / (2 * eps)
        assert abs(float(p.grad[idx]) - fd) < 1e-7 * max(1.0, abs(fd))


def test_circuit_gradients_pass_gradcheck():
    torch.manual_seed(2)
    n = 3
    H = np.diag(np.arange(2**n, dtype=float))
    w = torch.rand(2, n, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda w: FastHEA(n).cost(w, H), (w,))
    x = torch.randn(4, n, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, w: sel_expvals(x, w), (x, w))


def test_classification_path_dtypes_and_gradients():
    """Classification: float32 extractor -> float64 circuit -> float32 head, with the Learner in float64.

    The casts at the circuit boundary are differentiable, so the loss gradient must reach (a) the Learner
    through theta0, (b) the trainable angle projection through the encoding angles, and (c) the head; the
    frozen backbone must receive none, and the task embedding must be detached.
    """
    from qmaml_hep.classification.protocol import EMB, Arm
    from qmaml_hep.classification.protocol import W0_SCALE as CLS_W0

    torch.manual_seed(3)
    d_in, nq = 8, 4
    norm = (torch.zeros(d_in), torch.ones(d_in))
    arm = Arm("qmaml", d_in, nq, norm, seed=0)
    xs, xq = torch.randn(8, d_in), torch.randn(8, d_in)
    yq = torch.tensor([0, 1] * 4)

    emb = arm.ext.embed(xs)
    assert emb.dtype == torch.float32 and not emb.requires_grad and emb.shape == (8, EMB)
    theta0 = arm.theta0(xs)
    assert theta0.dtype == torch.float64 and theta0.requires_grad
    angles = arm.ext(xq)
    assert angles.dtype == torch.float32
    expvals = sel_expvals(angles, theta0)
    assert expvals.dtype == torch.float64
    logits = arm.logits(xq, theta0)
    assert logits.dtype == torch.float32

    loss = torch.nn.functional.cross_entropy(logits, yq)
    loss.backward()
    grads = {
        "learner": [p.grad for p in arm.learner.parameters()],
        "to_angles": [p.grad for p in arm.ext.to_angles.parameters()],
        "head": [p.grad for p in arm.pqc.fc.parameters()],
    }
    for part, gs in grads.items():
        assert all(g is not None and float(g.abs().max()) > 0 for g in gs), f"no gradient reaches the {part}"
    assert all(p.grad is None for p in arm.ext.backbone.parameters()), "the frozen backbone received a gradient"
    assert arm.pqc.weights.grad is None, "Q-MAML arm must not train the shared angles"
    assert CLS_W0 > 0


@pytest.mark.parametrize("init", ["zero", "pi", "uniform", "gaussian"])
def test_classifier_weights_are_float64(init):
    m = FastPQCModel(4, 2, init_type=init)
    assert m.weights.dtype == torch.float64
    assert m(torch.randn(3, 4)).dtype == torch.float32
