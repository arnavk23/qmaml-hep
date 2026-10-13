"""Track registry: one adapter per physical system, plus the switch to exact torch simulators.

Each ``track_*`` function returns a dict with
  sample(n, seed) -> list of tasks        desc(task) -> float64 descriptor (the Learner's input)
  ham(task) -> H                          e0(task, H) -> exact ground energy
  group(task) -> parameter-shape group    pqc(group) -> object with .cost(theta, H, **kw)
  shape(group) -> parameter shape         classical(shape, kind) -> initial angles
  extra: {name: (init kind, cost-kwargs fn)} for physics-motivated state preparations
  cfg: adaptation and pretraining budgets, train/test counts
"""

import numpy as np
import torch

from qmaml_hep.initializers import classical_init


# ---------------------------------------------------------------------------------------------------------
# Track adapters. Each returns a dict with:
#   sample(n, seed) -> list of tasks;  desc(task) -> 1-D float64 tensor (Learner input, raw units)
#   ham(task) -> H;  e0(task, H) -> exact ground energy;  group(task) -> key (parameter shape can differ by group)
#   pqc(group) -> object with .cost(theta, H, **kw);  shape(group) -> parameter shape
#   classical(shape, kind) -> tensor;  schemes_extra: {name: (init_kind, cost_kwargs_fn(task))}
#   cfg: depth, adapt_iters, adapt_lr, pretrain_epochs, pretrain_lr, n_train, n_test, qubits, err_scale(task)
# ---------------------------------------------------------------------------------------------------------
def _single(x):
    return "all"


def track_schwinger(num_qubits=4, depth=3, **_):
    from qmaml_hep.systems.schwinger import (
        SchwingerPQC,
        exact_ground_energy,
        param_shape,
        sample_task_space,
        schwinger_hamiltonian,
    )

    pqc = SchwingerPQC(num_qubits, depth)
    iters = {4: 300, 6: 300, 8: 300}[num_qubits]
    return dict(
        name=f"schwinger_q{num_qubits}_d{depth}",
        qubits=num_qubits,
        depth=depth,
        sample=lambda n, s: [tuple(map(float, t)) for t in sample_task_space(n, seed=s)],
        desc=lambda t: torch.tensor(t, dtype=torch.float64),
        ham=lambda t: schwinger_hamiltonian(num_qubits, t[0], t[1]),
        e0=lambda t, H: exact_ground_energy(H, num_qubits),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: param_shape(num_qubits, depth),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        cfg=dict(adapt_iters=iters, adapt_lr=0.02, pretrain_epochs=40, pretrain_lr=0.005, n_train=16, n_test=6),
    )


def track_z2(variant="hea", depth=3, **_):
    from qmaml_hep.systems.z2lgt import (
        NUM_QUBITS,
        Z2LGTHVAPQC,
        Z2LGTPQC,
        exact_ground_energy,
        sample_task_space,
        z2_lgt_hamiltonian,
    )

    if variant == "hea":
        pqc = Z2LGTPQC(NUM_QUBITS, depth)
        shape = (depth, NUM_QUBITS, 3)
        extra = {"gauss_vacuum": ("gauss_vacuum", lambda t: {"vacuum_prep": True})}
    else:
        pqc = Z2LGTHVAPQC(NUM_QUBITS, depth)
        shape = pqc.shape
        extra = {}
    return dict(
        name=f"z2_{variant}",
        qubits=NUM_QUBITS,
        depth=depth,
        sample=lambda n, s: [tuple(map(float, t)) for t in sample_task_space(n, seed=s)],
        desc=lambda t: torch.tensor(t, dtype=torch.float64),
        ham=lambda t: z2_lgt_hamiltonian(t[0], t[1], t[2], V=10.0),
        e0=lambda t, H: exact_ground_energy(H),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: shape,
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra=extra,
        cfg=dict(adapt_iters=300, adapt_lr=0.02, pretrain_epochs=40, pretrain_lr=0.005, n_train=16, n_test=6),
    )


def track_scalar(depth=3, **_):
    from qmaml_hep.systems.scalar import (
        NUM_QUBITS,
        ScalarFieldPQC,
        exact_ground_energy,
        sample_task_space,
        scalar_field_hamiltonian,
    )

    pqc = ScalarFieldPQC(NUM_QUBITS, depth)
    return dict(
        name="scalar",
        qubits=NUM_QUBITS,
        depth=depth,
        sample=lambda n, s: [tuple(map(float, t)) for t in sample_task_space(n, seed=s)],
        desc=lambda t: torch.tensor(t, dtype=torch.float64),
        ham=lambda t: scalar_field_hamiltonian(t[0], t[1]),
        e0=lambda t, H: exact_ground_energy(H),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: (depth, NUM_QUBITS, 3),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        cfg=dict(adapt_iters=300, adapt_lr=0.02, pretrain_epochs=40, pretrain_lr=0.005, n_train=16, n_test=6),
    )


def track_su2(num_qubits=4, depth=None, **_):
    from qmaml_hep.systems.su2lgt import SU2LGTPQC, exact_ground_energy, sample_task_space, su2_lgt_hamiltonian

    depth = depth or {4: 4, 6: 6, 8: 6}[num_qubits]
    N = num_qubits // 2
    pqc = SU2LGTPQC(num_qubits, depth)
    return dict(
        name=f"su2_q{num_qubits}",
        qubits=num_qubits,
        depth=depth,
        sample=lambda n, s: [tuple(map(float, t)) for t in sample_task_space(n, seed=s)],
        desc=lambda t: torch.tensor(t, dtype=torch.float64),
        ham=lambda t: su2_lgt_hamiltonian(N, t[0], t[1]),
        e0=lambda t, H: exact_ground_energy(H, num_qubits),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: (depth, num_qubits, 3),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        cfg=dict(
            adapt_iters=250 if num_qubits < 8 else 300,
            adapt_lr=0.05,
            pretrain_epochs=40,
            pretrain_lr=0.01,
            n_train=16,
            n_test=6,
        ),
    )


def track_neutrino(num_qubits=4, depth=3, **_):
    from qmaml_hep.systems.neutrino import NeutrinoPQC, exact_ground_energy, neutrino_hamiltonian, sample_task_space

    pqc = NeutrinoPQC(num_qubits, depth)
    return dict(
        name=f"neutrino_q{num_qubits}",
        qubits=num_qubits,
        depth=depth,
        sample=lambda n, s: [tuple(map(float, t)) for t in sample_task_space(n, seed=s)],
        desc=lambda t: torch.tensor(t, dtype=torch.float64),
        ham=lambda t: neutrino_hamiltonian(num_qubits, t[0], t[1]),
        e0=lambda t, H: exact_ground_energy(H, num_qubits),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: (depth, num_qubits, 3),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        cfg=dict(adapt_iters=200, adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01, n_train=16, n_test=6),
    )


def track_susyqm(depth=3, **_):
    from qmaml_hep.systems.susyqm import (
        NUM_QUBITS,
        SusyQMPQC,
        encode_task,
        exact_ground_energy,
        sample_task_space,
        susyqm_hamiltonian,
    )

    pqc = SusyQMPQC(NUM_QUBITS, depth)
    return dict(
        name="susyqm",
        qubits=NUM_QUBITS,
        depth=depth,
        sample=lambda n, s: list(sample_task_space(n, seed=s)),
        desc=lambda t: encode_task(*t),
        ham=lambda t: susyqm_hamiltonian(t[0], m=t[1], g=t[2], mu=t[3]),
        e0=lambda t, H: exact_ground_energy(H),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: (depth, NUM_QUBITS, 3),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={"basis_paper": ("basis_paper", lambda t: {"fermion_occupied": t[0] in ("HO", "AHO")})},
        # E0 = 0 exactly for unbroken SUSY, so the error is normalized by the mass scale m (the natural
        # energy unit; the first excitation is O(m)) instead of |E0|.
        err_scale=lambda t, e0: float(t[1]),
        cfg=dict(adapt_iters=200, adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01, n_train=18, n_test=9),
    )


def track_d8(depth=3, **_):
    from qmaml_hep.systems.d8lgt import D8LGTPQC, NUM_QUBITS, d8_lgt_hamiltonian, exact_ground_energy, sample_task_space

    pqc = D8LGTPQC(NUM_QUBITS, depth)
    return dict(
        name="d8",
        qubits=NUM_QUBITS,
        depth=depth,
        sample=lambda n, s: [tuple(map(float, t)) for t in sample_task_space(n, seed=s)],
        desc=lambda t: torch.tensor(t, dtype=torch.float64),
        ham=lambda t: d8_lgt_hamiltonian(*t),
        e0=lambda t, H: exact_ground_energy(H),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: (depth, NUM_QUBITS, 3),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        cfg=dict(adapt_iters=200, adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01, n_train=16, n_test=6),
    )


def track_nuclear(depth=2, **_):
    from qmaml_hep.systems.nuclear import (
        NUCLEUS_NAMES,
        NUM_QUBITS,
        NuclearUCCSDPQC,
        encode_task,
        exact_ground_energy,
        nuclear_hamiltonian,
        sample_task_space,
    )

    pqcs = {n: NuclearUCCSDPQC(n, depth=depth) for n in NUCLEUS_NAMES}
    return dict(
        name="nuclear",
        qubits=NUM_QUBITS,
        depth=depth,
        sample=lambda n, s: list(sample_task_space(n, seed=s)),
        desc=lambda t: encode_task(*t),
        ham=lambda t: nuclear_hamiltonian(*t),
        e0=lambda t, H: exact_ground_energy(H, t[0]),
        group=lambda t: t[0],
        pqc=lambda g: pqcs[g],
        shape=lambda g: (depth, pqcs[g].n_excitations),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        # theta = pi is equivalent to theta = 0 for every fermionic excitation (period-2pi generator, the
        # state only changes sign), so 'pi' is a duplicate of 'zero' here and is dropped.
        drop=("pi",),
        cfg=dict(adapt_iters=150, adapt_lr=0.1, pretrain_epochs=30, pretrain_lr=0.05, n_train=18, n_test=9),
    )


def track_yukawa(depth=3, **_):
    from qmaml_hep.systems.yukawa import (
        NUM_QUBITS,
        YukawaPQC,
        exact_ground_energy,
        sample_task_space,
        yukawa_hamiltonian,
    )

    pqc = YukawaPQC(NUM_QUBITS, depth)
    return dict(
        name="yukawa",
        qubits=NUM_QUBITS,
        depth=depth,
        sample=lambda n, s: [tuple(map(float, t)) for t in sample_task_space(n, seed=s)],
        desc=lambda t: torch.tensor(t, dtype=torch.float64),
        ham=lambda t: yukawa_hamiltonian(*t),
        e0=lambda t, H: exact_ground_energy(H),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: (depth, NUM_QUBITS, 3),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        cfg=dict(adapt_iters=150, adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01, n_train=16, n_test=6),
    )


def track_lmg(depth=3, **_):
    from qmaml_hep.systems.lmg import LMGPQC, NUM_QUBITS, exact_ground_energy, lmg_hamiltonian, sample_task_space

    pqc = LMGPQC(NUM_QUBITS, depth)
    return dict(
        name="lmg",
        qubits=NUM_QUBITS,
        depth=depth,
        sample=lambda n, s: [tuple(map(float, t)) for t in sample_task_space(n, seed=s)],
        desc=lambda t: torch.tensor(t, dtype=torch.float64),
        ham=lambda t: lmg_hamiltonian(*t),
        e0=lambda t, H: exact_ground_energy(H),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: (depth, NUM_QUBITS, 3),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        cfg=dict(adapt_iters=150, adapt_lr=0.05, pretrain_epochs=40, pretrain_lr=0.01, n_train=16, n_test=6),
    )


def track_heisenberg(num_qubits=6, depth=8, **_):
    """Calibration: the original paper's Heisenberg XYZ task space at a size where VQE converges."""
    import scipy.sparse.linalg

    from qmaml_hep.initializers import classical_init
    from qmaml_hep.simulators import heisenberg_energy, ising_brick_state
    from qmaml_hep.systems.heisenberg import heisenberg_xyz_hamiltonian, sample_task_space

    class P:
        def cost(self, w, J):
            return heisenberg_energy(ising_brick_state(w, num_qubits), num_qubits, J)

    pqc = P()

    def e0(t, J):
        H = heisenberg_xyz_hamiltonian(num_qubits, *J).sparse_matrix(wire_order=range(num_qubits))
        return (
            float(scipy.sparse.linalg.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])
            if H.shape[0] > 64
            else float(np.linalg.eigvalsh(H.toarray())[0])
        )

    return dict(
        name=f"heisenberg_q{num_qubits}_d{depth}",
        qubits=num_qubits,
        depth=depth,
        sample=lambda n, s: [tuple(map(float, t)) for t in sample_task_space(n, seed=s)],
        desc=lambda t: torch.tensor(t, dtype=torch.float64),
        ham=lambda t: t,
        e0=e0,
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: (depth, num_qubits - 1, 3),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        cfg=dict(adapt_iters=1000, adapt_lr=0.01, pretrain_epochs=30, pretrain_lr=0.001, n_train=16, n_test=8),
    )


def track_h2(depth=7, **_):
    """Calibration: H2 bond-length task space, 7 StronglyEntanglingLayers, descriptor = Pauli coefficients."""
    from qmaml_hep.systems.molecule import (
        MoleculePQC,
        exact_ground_energy,
        h2_hamiltonian,
        hamiltonian_coeffs,
        sample_bond_lengths,
    )

    pqc = MoleculePQC(4, depth)
    cache = {}

    def ham(t):
        if t[0] not in cache:
            cache[t[0]] = h2_hamiltonian(t[0])[0]
        return cache[t[0]]

    return dict(
        name="h2",
        qubits=4,
        depth=depth,
        sample=lambda n, s: [(float(b),) for b in sample_bond_lengths(n, seed=s)],
        desc=lambda t: hamiltonian_coeffs(ham(t)),
        ham=ham,
        e0=lambda t, H: exact_ground_energy(H, 4),
        group=_single,
        pqc=lambda g: pqc,
        shape=lambda g: (depth, 4, 3),
        classical=lambda sh, k: classical_init(sh, k, depth),
        extra={},
        cfg=dict(adapt_iters=2000, adapt_lr=0.001, pretrain_epochs=30, pretrain_lr=0.001, n_train=12, n_test=6),
    )


TRACKS = {
    "A_schwinger": track_schwinger,
    "B_z2": lambda **k: track_z2("hea", **k),
    "B_z2_hva": lambda **k: track_z2("hva", **k),
    "C_scalar": track_scalar,
    "D_su2": track_su2,
    "E_neutrino": track_neutrino,
    "F_susyqm": track_susyqm,
    "G_d8": track_d8,
    "H_nuclear": track_nuclear,
    "I_yukawa": track_yukawa,
    "J_lmg": track_lmg,
    "V_heisenberg": track_heisenberg,
    "V_h2": track_h2,
}

FAST_HEA_CLASSES = {
    "SchwingerPQC",
    "Z2LGTPQC",
    "ScalarFieldPQC",
    "SU2LGTPQC",
    "NeutrinoPQC",
    "SusyQMPQC",
    "D8LGTPQC",
    "YukawaPQC",
    "LMGPQC",
    "MoleculePQC",
}


def make_fast(tr):
    """Swap the track's PennyLane PQC for an exact torch statevector simulator of the same circuit (returns the
    original so the caller can cross-check). Nuclear (UCCSD) keeps PennyLane."""
    import pennylane as qml

    from qmaml_hep.simulators import FastHEA, FastPauliRotAnsatz, FastUCCSD

    if tr["group"] is not _single:
        groups = sorted({tr["group"](t) for t in tr["sample"](200, 0)})
        origs = {g: tr["pqc"](g) for g in groups}
        if not all(type(o).__name__ == "NuclearUCCSDPQC" for o in origs.values()):
            return None
        fasts = {g: FastUCCSD(o.num_qubits, o.s_wires, o.d_wires, o.init_state, o.depth) for g, o in origs.items()}
        tr["pqc"] = lambda g: fasts[g]
        return origs
    orig = tr["pqc"]("all")
    cls = type(orig).__name__
    n = tr["qubits"]
    if cls in FAST_HEA_CLASSES:
        preps = {}
        if cls == "Z2LGTPQC":
            from qmaml_hep.systems.z2lgt import vacuum_prep

            dev = qml.device("default.qubit", wires=n)
            preps["vacuum_prep"] = torch.tensor(
                qml.QNode(lambda: (vacuum_prep(), qml.state())[1], dev)(), dtype=torch.complex128
            )
        if cls == "SusyQMPQC":
            v = torch.zeros(2**n, dtype=torch.complex128)
            v[1] = 1.0  # PauliX on the last (fermion) wire
            preps["fermion_occupied"] = v
        fast = FastHEA(n, preps)
    elif cls == "Z2LGTHVAPQC":
        from qmaml_hep.systems.z2lgt import LINKS, MATTER, vacuum_prep

        dev = qml.device("default.qubit", wires=n)
        psi0 = torch.tensor(qml.QNode(lambda: (vacuum_prep(), qml.state())[1], dev)(), dtype=torch.complex128)
        words = (
            [("XZX", list(b)) for b in orig._HOPPING_BONDS]
            + [("YZY", list(b)) for b in orig._HOPPING_BONDS]
            + [("Z", [m]) for m in MATTER]
            + [("X", [l]) for l in LINKS]
            + [("ZZZZ", list(LINKS))]
        )
        gens = [
            torch.tensor(
                qml.matrix(qml.prod(*[getattr(qml, "Pauli" + c)(q) for c, q in zip(w, ws)]), wire_order=range(n)),
                dtype=torch.complex128,
            )
            for w, ws in words
        ]
        fast = FastPauliRotAnsatz(n, gens, psi0, orig.shape)
    else:
        return None
    tr["pqc"] = lambda g: fast
    return orig


def check_fast(fast, orig, H, shape, kws=({},), seed=7):
    g = torch.Generator().manual_seed(seed)
    for kw in kws:
        w1 = (torch.rand(shape, generator=g, dtype=torch.float64) * 2 * np.pi).requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)
        e1 = orig.cost(w1, H, **kw)
        e1.backward()
        e2 = fast.cost(w2, H, **kw)
        e2.backward()
        # relative to the energy / gradient scale: lightning.qubit evaluates Hamiltonian expectations in single
        # precision (~1e-7 relative), the fast simulator and default.qubit in double
        e1, e2 = float(e1.detach()), float(e2.detach())
        de = abs(e1 - e2) / max(1.0, abs(e2))
        dg = float((w1.grad - w2.grad).abs().max()) / max(1.0, float(w2.grad.abs().max()))
        print(f"  fast-simulator check {kw}: rel|dE|={de:.1e} rel max|dgrad|={dg:.1e}", flush=True)
        assert de < 1e-6 and dg < 1e-6, "fast simulator disagrees with the PennyLane circuit"
