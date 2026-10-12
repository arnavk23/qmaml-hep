"""
Exact batched statevector simulation of the classification PQC:
    RY(x_i) angle encoding on every wire (optionally tanh-bounded to (-pi, pi)),
    qml.StronglyEntanglingLayers(weights)  (Rot = RZ(omega) RY(theta) RZ(phi); CNOT ring with range
    r_l = (l mod (n-1)) + 1, PennyLane's default),
    <Z_i> on every wire, then a linear head -> 2 logits.
Pure torch (complex128), so a whole batch is one tensor contraction and autograd gives exact gradients.
Checked against the PennyLane QNode by verify() below; used only to make the round-2 classification
protocol (hyperparameter tuning, 10 seeds, 60 test tasks) computationally feasible.
"""

import numpy as np
import torch
import torch.nn as nn


def _apply_1q(state, U, wire, n):
    """state: (B, 2,...,2); U: (B,2,2) or (2,2) complex."""
    state = torch.movedim(state, wire + 1, -1)
    if U.dim() == 2:
        state = state @ U.transpose(-1, -2)
    else:
        shp = state.shape
        state = torch.einsum("bij,b...j->b...i", U, state.reshape(shp[0], -1, 2)).reshape(shp)
    return torch.movedim(state, -1, wire + 1)


def _ry(a):
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex128)


def _rot(phi, theta, omega):
    # RZ(omega) RY(theta) RZ(phi)
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    e = lambda x: torch.exp(1j * x.to(torch.complex128))
    return torch.stack(
        [
            torch.stack([e(-(phi + omega) / 2) * c, -e((phi - omega) / 2) * s], -1),
            torch.stack([e(-(phi - omega) / 2) * s, e((phi + omega) / 2) * c], -1),
        ],
        -2,
    )


def _cnot(state, ctrl, tgt):
    state = state.clone()
    idx1 = [slice(None)] * state.dim()
    idx1[ctrl + 1] = 1
    sub = state[tuple(idx1)]
    t = tgt + 1 if tgt < ctrl else tgt  # target axis inside the sub-tensor (control axis removed)
    state[tuple(idx1)] = torch.flip(sub, dims=[t])
    return state


_PERM_CACHE, _ZSIGN_CACHE = {}, {}


def _ring_perm(n, r):
    """Basis permutation of the CNOT ring CNOT(i, (i+r) % n), i = 0..n-1, applied in order (wire 0 = MSB)."""
    key = (n, r)
    if key not in _PERM_CACHE:
        idx = np.arange(2**n)
        for i in range(n):
            c, t = n - 1 - i, n - 1 - ((i + r) % n)  # bit positions (LSB = 0)
            idx = np.where((idx >> c) & 1, idx ^ (1 << t), idx)
        # idx[b] = image of basis state b; new_state[idx[b]] = old_state[b]
        inv = np.empty_like(idx)
        inv[idx] = np.arange(2**n)
        _PERM_CACHE[key] = torch.from_numpy(inv)
    return _PERM_CACHE[key]


def _zsign(n):
    if n not in _ZSIGN_CACHE:
        b = np.arange(2**n)
        _ZSIGN_CACHE[n] = torch.tensor(
            np.stack([1 - 2 * ((b >> (n - 1 - i)) & 1) for i in range(n)], 1), dtype=torch.float64
        )
    return _ZSIGN_CACHE[n]


def _kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def sel_expvals(x, weights, bound_angles=True):
    """Dense-unitary evaluation: U_var built once per call (it does not depend on the sample), the encoding
    is a product state, so each sample costs one matrix-vector product. Same result as sel_expvals_gatewise."""
    x = x.to(torch.float64)
    w = weights.to(torch.float64)
    B, n = x.shape
    if bound_angles:
        x = torch.tanh(x) * np.pi
    # encoding: RY(x_i)|0> = (cos x/2, sin x/2) per wire, product over wires (wire 0 most significant)
    c, s = torch.cos(x / 2), torch.sin(x / 2)
    psi = torch.stack([c[:, 0], s[:, 0]], 1)
    for i in range(1, n):
        psi = (psi[:, :, None] * torch.stack([c[:, i], s[:, i]], 1)[:, None, :]).reshape(B, -1)
    psi = psi.to(torch.complex128)
    for l in range(w.shape[0]):
        R = _rot(w[l, :, 0], w[l, :, 1], w[l, :, 2])  # (n, 2, 2)
        for i in range(n):
            psi = torch.einsum("ij,bajc->baic", R[i], psi.reshape(B, 2**i, 2, 2 ** (n - i - 1))).reshape(B, -1)
        if n > 1:
            psi = psi[:, _ring_perm(n, (l % (n - 1)) + 1)]
    return (psi.abs() ** 2) @ _zsign(n)


def sel_expvals_gatewise(x, weights, bound_angles=True):
    """x: (B, n) real; weights: (L, n, 3). Returns (B, n) float64 <Z_i>."""
    x = x.to(torch.float64)
    w = weights.to(torch.float64)
    B, n = x.shape
    if bound_angles:
        x = torch.tanh(x) * np.pi
    state = torch.zeros((B,) + (2,) * n, dtype=torch.complex128)
    state[(slice(None),) + (0,) * n] = 1.0
    for i in range(n):
        state = _apply_1q(state, _ry(x[:, i]), i, n)
    L = w.shape[0]
    for l in range(L):
        for i in range(n):
            state = _apply_1q(state, _rot(w[l, i, 0], w[l, i, 1], w[l, i, 2]), i, n)
        if n > 1:
            r = (l % (n - 1)) + 1
            for i in range(n):
                state = _cnot(state, i, (i + r) % n)
    p = state.abs() ** 2
    out = []
    for i in range(n):
        pi = torch.movedim(p, i + 1, 1).reshape(B, 2, -1).sum(-1)
        out.append(pi[:, 0] - pi[:, 1])
    return torch.stack(out, 1)


class FastPQCModel(nn.Module):
    """Angle-encoding classifier: RY(tanh(x) pi) encoding, StronglyEntanglingLayers, <Z_i>, linear head.

    The circuit runs in float64/complex128; its expectation values are cast to float32 for the head."""

    def __init__(self, num_qubits, depth, init_type="zero", bound_angles=True):
        super().__init__()
        self.num_qubits, self.depth, self.bound_angles = num_qubits, depth, bound_angles
        shape = (depth, num_qubits, 3)
        if init_type == "zero":
            w = torch.zeros(shape, dtype=torch.float64)
        elif init_type == "pi":
            w = torch.full(shape, np.pi, dtype=torch.float64)
        elif init_type == "uniform":
            w = torch.rand(shape, dtype=torch.float64) * (0.05 * np.pi)
        elif init_type == "gaussian":
            gamma = 1.0 / (4 * num_qubits * (depth + 2))
            w = torch.normal(0.0, gamma, size=shape, dtype=torch.float64)
        else:
            w = torch.randn(shape, dtype=torch.float64)
        self.weights = nn.Parameter(w)
        self.fc = nn.Linear(num_qubits, 2)

    def forward(self, x, weights_override=None):
        w = weights_override if weights_override is not None else self.weights
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.fc(sel_expvals(x, w, self.bound_angles).to(torch.float32))


def verify(n=4, depth=2, B=5, seed=0, atol=1e-10):
    import pennylane as qml

    torch.manual_seed(seed)
    x = torch.randn(B, n, dtype=torch.float64)
    w = torch.randn(depth, n, 3, dtype=torch.float64, requires_grad=True)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(xi, w):
        s = torch.tanh(xi) * np.pi
        for i in range(n):
            qml.RY(s[i], wires=i)
        qml.StronglyEntanglingLayers(w, wires=range(n))
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    ref = torch.stack([torch.stack(circ(x[b], w)) for b in range(B)])
    fast = sel_expvals(x, w, True)
    g_ref = torch.autograd.grad(ref.sum(), w)[0]
    g_fast = torch.autograd.grad(fast.sum(), w)[0]
    return float((ref - fast).abs().max()), float((g_ref - g_fast).abs().max())


if __name__ == "__main__":
    for n, d in [(2, 3), (4, 2), (6, 2), (8, 2), (5, 4)]:
        e, g = verify(n, d)
        print(f"n={n} depth={d}: max |expval diff|={e:.2e}  max |grad diff|={g:.2e}")


# ---------------------------------------------------------------------------------------------------------
# Single-state simulators for the paper-scale replication (qmaml_hep.experiments.paper_replication)
# ---------------------------------------------------------------------------------------------------------
_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1.0, -1.0]),
}
_XX, _YY, _ZZ = (torch.tensor(np.kron(_PAULI[p], _PAULI[p]), dtype=torch.complex128) for p in "XYZ")
_BELL = torch.tensor(
    np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]]).T / np.sqrt(2), dtype=torch.complex128
)  # columns: Phi+, Phi-, Psi+, Psi-


def _apply_2q_adjacent(psi, U, i, n):
    return torch.einsum("ij,ajb->aib", U, psi.reshape(2**i, 4, 2 ** (n - i - 2))).reshape(-1)


def ising_brick_state(w, n):
    """|psi(w)> for the Heisenberg ansatz: per layer l and edge (i, i+1), IsingXX(w[l,i,0]) IsingYY(w[l,i,1])
    IsingZZ(w[l,i,2]) -- the three generators commute, so the triple is exp(-i/2 (a XX + b YY + c ZZ))."""
    psi = torch.zeros(2**n, dtype=torch.complex128)
    psi[0] = 1.0
    w = w.to(torch.float64)
    # XX, YY, ZZ are simultaneously diagonal in the Bell basis (Phi+, Phi-, Psi+, Psi-) with eigenvalues
    # (+1,-1,+1), (-1,+1,+1), (+1,+1,-1), (-1,-1,-1), so U = B diag(exp(-i lam/2)) B^dagger in closed form.
    a, b, c = w[..., 0], w[..., 1], w[..., 2]
    lam = torch.stack([a - b + c, -a + b + c, a + b - c, -a - b - c], -1)  # (L, n-1, 4)
    U = torch.einsum("ik,...k,jk->...ij", _BELL, torch.exp(-0.5j * lam.to(torch.complex128)), _BELL.conj())
    for l in range(w.shape[0]):
        for i in range(n - 1):
            psi = _apply_2q_adjacent(psi, U[l, i], i, n)
    return psi


def heisenberg_energy(psi, n, J):
    """<psi| -sum_i (Jx XX + Jy YY + Jz ZZ)_{i,i+1} |psi>."""
    e = 0.0
    for i in range(n - 1):
        for Jc, P in zip(J, (_XX, _YY, _ZZ)):
            e = e - Jc * torch.real(torch.vdot(psi, _apply_2q_adjacent(psi, P, i, n)))
    return e


def sel_state(w, n, psi0=None):
    """|psi(w)> = StronglyEntanglingLayers(w)|psi0> (default CNOT ranges); psi0 defaults to |0...0>."""
    if psi0 is None:
        psi = torch.zeros(1, 2**n, dtype=torch.complex128)
        psi[0, 0] = 1.0
    else:
        psi = psi0.reshape(1, -1).to(torch.complex128)
    w = w.to(torch.float64)
    for l in range(w.shape[0]):
        R = _rot(w[l, :, 0], w[l, :, 1], w[l, :, 2])
        for i in range(n):
            psi = torch.einsum("ij,bajc->baic", R[i], psi.reshape(1, 2**i, 2, 2 ** (n - i - 1))).reshape(1, -1)
        if n > 1:
            psi = psi[:, _ring_perm(n, (l % (n - 1)) + 1)]
    return psi[0]


# ---------------------------------------------------------------------------------------------------------
# Fast exact energies for the unified benchmark (drop-in for the per-track PennyLane PQC classes)
# ---------------------------------------------------------------------------------------------------------
class _DenseCache:
    def __init__(self, n):
        self.n, self.store = n, {}

    def __call__(self, H):
        k = id(H)
        if k not in self.store:
            if isinstance(H, np.ndarray):
                M = H
            elif torch.is_tensor(H):
                M = H.numpy()
            else:
                M = H.sparse_matrix(wire_order=range(self.n)).toarray()
            self.store[k] = (H, torch.tensor(np.asarray(M), dtype=torch.complex128))  # keep H alive: id stays unique
        return self.store[k][1]


def _energy(psi, Hd):
    return torch.real(torch.vdot(psi, Hd @ psi))


class FastHEA:
    """StronglyEntanglingLayers ansatz, exact energy <psi|H|psi> with H as a dense matrix. `preps` maps a cost
    keyword (e.g. 'vacuum_prep', 'fermion_occupied') to the initial state that keyword selects."""

    def __init__(self, n, preps=None):
        self.n, self.dense, self.preps = n, _DenseCache(n), preps or {}

    def cost(self, weights, H, **kw):
        psi0 = None
        for k, v in kw.items():
            if v and k in self.preps:
                psi0 = self.preps[k]
        return _energy(sel_state(weights, self.n, psi0), self.dense(H))


class FastPauliRotAnsatz:
    """Layers of exp(-i theta_k/2 P_k) = cos(theta_k/2) I - i sin(theta_k/2) P_k from a fixed initial state;
    `gens` is the per-layer list of dense Pauli matrices P_k, in circuit order."""

    def __init__(self, n, gens, psi0, shape):
        self.n, self.gens, self.psi0, self.shape = n, gens, psi0, shape
        self.dense = _DenseCache(n)

    def cost(self, weights, H, **_):
        psi = self.psi0.clone()
        w = weights.to(torch.float64)
        for l in range(w.shape[0]):
            for k, P in enumerate(self.gens):
                c, s_ = torch.cos(w[l, k] / 2), torch.sin(w[l, k] / 2)
                psi = c * psi - 1j * s_ * (P @ psi)
        return _energy(psi, self.dense(H))


class FastUCCSD:
    """qml.UCCSD (BasisState(init_state), then per repeat: FermionicDoubleExcitation for every d_wires entry,
    then FermionicSingleExcitation for every s_wires entry) as exact dense rotations. Each excitation is
    U(theta) = exp(theta G) with G^3 = -G, so U = I + sin(c theta) G + (1 - cos(c theta)) G^2 with G = A/c; A is read off
    PennyLane's own gate matrices by a symmetric difference, and the
    closed form is checked against PennyLane at a random angle for every excitation."""

    def __init__(self, n, s_wires, d_wires, init_state, depth):
        import pennylane as qml

        self.n, self.depth = n, depth
        self.n_exc = len(s_wires) + len(d_wires)
        I = np.eye(2**n)
        mats = []  # (weight index within a layer, G, G^2) in circuit order
        ops = [
            (len(s_wires) + i, lambda t, w=w: qml.FermionicDoubleExcitation(t, wires1=w[0], wires2=w[1]))
            for i, w in enumerate(d_wires)
        ]
        ops += [(j, lambda t, w=w: qml.FermionicSingleExcitation(t, wires=w)) for j, w in enumerate(s_wires)]
        rng = np.random.default_rng(0)
        for k, mk in ops:
            # U(t) = exp(t A), A anti-Hermitian with A^3 = -c^2 A (a rotation in one 2-D subspace), so
            # U(t) = I + sin(c t)/c A + (1 - cos(c t))/c^2 A^2 (PennyLane uses c = 1/2).
            eps = 1e-5
            A = (qml.matrix(mk(eps), wire_order=range(n)) - qml.matrix(mk(-eps), wire_order=range(n))) / (2 * eps)
            c = np.sqrt(max(np.real(-np.vdot(A @ A @ A, A) / np.vdot(A, A)), 1e-30))
            c = round(c * 4) / 4 if abs(c * 4 - round(c * 4)) < 1e-4 else c
            A = A / c
            A[np.abs(A) < 1e-9] = 0.0
            A = np.round(A, 9)
            A2 = A @ A
            t = rng.uniform(-np.pi, np.pi)
            err = np.abs(
                qml.matrix(mk(t), wire_order=range(n)) - (I + np.sin(c * t) * A + (1 - np.cos(c * t)) * A2)
            ).max()
            assert err < 1e-10, f"UCCSD closed form off by {err:.1e} (c={c})"
            mats.append((k, c, torch.tensor(A, dtype=torch.complex128), torch.tensor(A2, dtype=torch.complex128)))
        self.mats = mats
        idx = int("".join(str(int(b)) for b in init_state), 2)
        self.psi0 = torch.zeros(2**n, dtype=torch.complex128)
        self.psi0[idx] = 1.0
        self.dense = _DenseCache(n)

    def cost(self, weights, H, **_):
        w = weights.reshape(self.depth, self.n_exc).to(torch.float64)
        psi = self.psi0
        for l in range(self.depth):
            for k, c, G, G2 in self.mats:
                psi = psi + torch.sin(c * w[l, k]) * (G @ psi) + (1 - torch.cos(c * w[l, k])) * (G2 @ psi)
        return _energy(psi, self.dense(H))
