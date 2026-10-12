"""
Paper-scale replication of Lee, Cho & Kim (AAAI 2025, arXiv:2501.05906), Figure 4(a) and 4(c), using every
setting the paper states:
  Learner       input -> 256 -> 256 -> P (ReLU), 30 epochs, Adam lr 1e-3, no output scaling
  Adaptation    2000 iterations, Adam lr 1e-3
  Baselines     Zero, pi, Uniform U[-0.05pi, 0.05pi], Gaussian N(0, gamma^2) with the paper's gamma^2
  Heisenberg    12 qubits, J = (Jx, Jy, Jz) drawn from {-3.0, -2.9, ..., 3.0}, h = 0, 16 test tasks;
                ansatz = blocks of IsingXX/IsingYY/IsingZZ gates (the exact layout is in the paper's appendix,
                which is not public; we use one XX/YY/ZZ triple per nearest-neighbour edge per layer), L = 20
                layers (the paper's gamma^2 = 1/176 corresponds to S = 2, L = 20); gamma^2 = 1/176
  Molecule      10 qubits (5 active orbitals), 7 StronglyEntanglingLayers, task descriptor = Pauli coefficient
                vector C, 12 training bond lengths, 6 test bond lengths; gamma^2 = 1/360. The paper does not name
                its molecules; we use LiH (2 active electrons, 5 active orbitals, STO-3G).
Not stated in the paper, chosen here: number of Heisenberg training tasks (32), LiH bond-length range
(1.0-3.0 Angstrom), random seed. Energies are stored raw (no log regularizer); the gap is reported with an
explicit log base in the analysis.

Usage:  python -m qmaml_hep.experiments.paper_replication heisenberg   |   python -m qmaml_hep.experiments.paper_replication molecule
"""

import json
import os
import sys
import time

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from qmaml_hep.learner import Learner
from qmaml_hep.paths import RESULTS

OUT = os.path.join(RESULTS, "paper_replication")
EPOCHS, LR_PRE, ITERS, LR_AD = 30, 1e-3, 2000, 1e-3


class ExpvalPQC:
    """Exact statevector energy. `fast(w, H)` is a pure-torch simulator of the same circuit (qmaml_hep.simulators);
    it is checked against the PennyLane QNode (`template`) on a random point before any run."""

    def __init__(self, n, template, fast):
        self.dev = qml.device("lightning.qubit", wires=n)
        self.template, self.fast = template, fast

    def cost_pennylane(self, w, H):
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def c(w):
            self.template(w)
            return qml.expval(H)

        return c(w.to(torch.float64))

    def cost(self, w, H):
        return self.fast(w, H)

    def check(self, shape, H, seed=123):
        g = torch.Generator().manual_seed(seed)
        w1 = (torch.rand(shape, generator=g, dtype=torch.float64) * 2 * np.pi).requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)
        e1 = self.cost_pennylane(w1, H)
        e1.backward()
        e2 = self.cost(w2, H)
        e2.backward()
        de, dg = abs(float(e1) - float(e2)), float((w1.grad - w2.grad).abs().max())
        print(f"  fast-vs-PennyLane check: |dE|={de:.2e}  max|dgrad|={dg:.2e}", flush=True)
        assert de < 1e-6 and dg < 1e-6


def ground(H, n):
    import scipy.sparse.linalg

    return float(
        scipy.sparse.linalg.eigsh(H.sparse_matrix(wire_order=range(n)), k=1, which="SA", return_eigenvectors=False)[0]
    )


def run(name, n, shape, template, fast, gamma2, train, test, desc, ham, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pqc = ExpvalPQC(n, template, fast)
    Htr = [ham(t) for t in train]
    Hte = [ham(t) for t in test]
    pqc.check(shape, Htr[0])
    E0 = [ground(H, n) for H in Hte]
    learner = Learner(len(desc(train[0])), shape, 256, n_hidden=2).double()
    opt = torch.optim.Adam(learner.parameters(), lr=LR_PRE)
    pre = []
    t0 = time.time()
    for ep in range(EPOCHS):
        tot = 0.0
        for t, H in zip(train, Htr):
            opt.zero_grad()
            c = pqc.cost(learner(desc(t)), H)
            c.backward()
            opt.step()
            tot += c.item()
        pre.append(tot / len(train))
        print(f"[{name}] pretrain epoch {ep + 1}: mean E = {pre[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    inits = {
        "qmaml": lambda t: learner(desc(t)).detach(),
        "zero": lambda t: torch.zeros(shape, dtype=torch.float64),
        "pi": lambda t: torch.full(shape, np.pi, dtype=torch.float64),
        "uniform": lambda t: (torch.rand(shape, dtype=torch.float64) * 2 - 1) * 0.05 * np.pi,
        "gaussian": lambda t: torch.normal(0.0, gamma2**0.5, size=shape, dtype=torch.float64),
    }
    # optional sharding over held-out tasks (env SHARD/NSHARD); every shard pretrains the identical Learner
    # (same seed, deterministic CPU), and qmaml_hep.experiments.merge_shards reassembles the task order
    sh, nsh = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    path = os.path.join(OUT, f"{name}.json" if nsh == 1 else f"{name}_shard{sh}of{nsh}.json")
    res = {
        "E0": E0,
        "tasks": [list(map(float, np.atleast_1d(t))) for t in test],
        "pretrain_loss": pre,
        "train_E0_mean": float(np.mean([ground(H, n) for H in Htr])),
        "curves": {k: [] for k in inits},
        "task_index": [],
    }
    for i, (t, H) in enumerate(zip(test, Hte)):
        if i % nsh != sh:
            continue
        res["task_index"].append(i)
        for k, f in inits.items():
            th = nn.Parameter(f(t).clone())
            o = torch.optim.Adam([th], lr=LR_AD)
            en = []
            for _ in range(ITERS):
                o.zero_grad()
                c = pqc.cost(th, H)
                c.backward()
                o.step()
                en.append(c.item())
            res["curves"][k].append(en)
            print(f"[{name}] task {i} {k}: E_final - E0 = {en[-1] - E0[i]:.3e}", flush=True)
        json.dump(res, open(path, "w"))


def heisenberg():
    n, L = 12, 20
    shape = (L, n - 1, 3)
    grid = np.round(np.arange(-3.0, 3.0 + 1e-9, 0.1), 1)
    rng = np.random.default_rng(0)
    train, test = rng.choice(grid, (32, 3)), rng.choice(grid, (16, 3))

    def ham(J):
        c, o = [], []
        for i in range(n - 1):
            c += [-J[0], -J[1], -J[2]]
            o += [
                qml.PauliX(i) @ qml.PauliX(i + 1),
                qml.PauliY(i) @ qml.PauliY(i + 1),
                qml.PauliZ(i) @ qml.PauliZ(i + 1),
            ]
        return qml.Hamiltonian(c, o)

    def template(w):
        for l in range(L):
            for i in range(n - 1):
                qml.IsingXX(w[l, i, 0], wires=[i, i + 1])
                qml.IsingYY(w[l, i, 1], wires=[i, i + 1])
                qml.IsingZZ(w[l, i, 2], wires=[i, i + 1])

    from qmaml_hep.simulators import heisenberg_energy, ising_brick_state

    def ham_j(J):
        H = ham(J)
        H.J = [float(x) for x in J]
        return H

    fast = lambda w, H: heisenberg_energy(ising_brick_state(w, n), n, H.J)
    run(
        "heisenberg_q12_L20",
        n,
        shape,
        template,
        fast,
        1 / 176,
        train,
        test,
        lambda J: torch.tensor(J, dtype=torch.float64),
        ham_j,
    )


def molecule():
    n, L = 10, 7
    shape = (L, n, 3)
    rng = np.random.default_rng(0)
    train, test = [float(x) for x in np.linspace(1.0, 3.0, 12)], [float(x) for x in np.sort(rng.uniform(1.0, 3.0, 6))]
    cache = {}

    def ham(b):
        if b not in cache:
            H, q = qml.qchem.molecular_hamiltonian(
                ["Li", "H"], np.array([0, 0, 0, 0, 0, b / 0.529177]), active_electrons=2, active_orbitals=5
            )
            assert q == n
            cache[b] = H
        return cache[b]

    keys = [str(o) for o in ham(float(train[0])).terms()[1]]

    def desc(b):
        # Pauli-coefficient vector C aligned by Pauli-word label (zero where a term is absent)
        c, o = ham(b).terms()
        d = {str(op): float(np.real(v)) for v, op in zip(c, o)}
        return torch.tensor([d.get(k, 0.0) for k in keys], dtype=torch.float64)

    from qmaml_hep.simulators import sel_state

    dense = {}

    def fast(w, H):
        k = id(H)
        if k not in dense:
            dense[k] = torch.tensor(H.sparse_matrix(wire_order=range(n)).toarray(), dtype=torch.complex128)
        psi = sel_state(w, n)
        return torch.real(torch.vdot(psi, dense[k] @ psi))

    run(
        "molecule_LiH_q10_L7",
        n,
        shape,
        lambda w: qml.StronglyEntanglingLayers(w, wires=range(n)),
        fast,
        1 / 360,
        train,
        test,
        desc,
        ham,
    )


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(1)
    {"heisenberg": heisenberg, "molecule": molecule}[sys.argv[1]]()
