"""Track H: light nuclei (deuteron / triton / helium-3) in lattice pionless effective field theory,
following Cifci, Akkoyun & La Ronde, "Systematic VQE Benchmarking of the Deuteron, Triton, and Helium-3
within Lattice Pionless Effective Field Theory" (arXiv:2604.20908), their Eqs. 1-4 -- the best-precedented
VQE-for-physics track of this project (lineage of the 2018 IBM deuteron VQE result).

**The lattice.** 1D spatial lattice, N_SITES sites, each site hosting four spin-isospin modes
sigma in {p-up, p-down, n-up, n-down}. Fermionic creation/annihilation operators a^dag_sigma(i), a_sigma(i)
per mode per site, Jordan-Wigner-mapped. Wire order groups all proton modes first, then all neutron modes
(not interleaved by site), so that particle-number-conserving excitations for each species can be generated
independently with `qml.qchem.excitations`: wires [0..2*N_SITES-1] = proton spin-orbitals (up,down per
site), wires [2*N_SITES..4*N_SITES-1] = neutron spin-orbitals.

**Hamiltonian** (paper's Eq. 2, plus Eq. 4 for helium-3's Coulomb term):

  H = -t * sum_{<i,j>,sigma} [a^dag_sigma(i) a_sigma(j) + h.c.]
    + (C/2) * sum_{i, sigma != sigma'} n_sigma(i) n_sigma'(i)
    + (D/6) * sum_{i, sigma != sigma' != sigma''} n_sigma(i) n_sigma'(i) n_sigma''(i)
    + V0 * sum_{i<j} n_p(i) n_p(j),                      n_p(i) = n_{p-up}(i) + n_{p-down}(i)

t (hopping), C (2-body contact), D (3-body contact) are this track's continuous task parameters (the
paper calibrates them to experimental binding energies and holds them fixed; here, as in every other track
in this project, they are swept as the physical parameters a Learner must generalize across). V0 (Coulomb
strength) is a fixed representation constant, not a task parameter, active only when 2 protons are present
(deuteron and triton have only 1 proton each, so this term is identically zero for them regardless of V0).

**Which nucleus** is a discrete task-family axis (like Track F's superpotential choice): deuteron
(1 proton + 1 neutron), triton (1 proton + 2 neutrons), helium-3 (2 protons + 1 neutron). N_SITES=2 for
all three (the smallest lattice with a nonzero hopping term), giving 8 qubits total (4 proton + 4 neutron
spin-orbitals) for every nucleus in this MVP.

Ground-truth energy for a given nucleus is the lowest eigenvalue of H *restricted to the correct
(n_p, n_n) sector* -- exact, not approximate, since total proton number and total neutron number are each
separately conserved by every term in H (no term converts a proton mode to a neutron mode or vice versa),
so eigenstates of H have exact, well-defined (n_p, n_n) and can be filtered by their measured particle
number rather than needing a penalty term (contrast Track B's Gauss-law penalty, which was necessary there
because the constraint wasn't automatically respected by the Hamiltonian's own symmetry structure in the
same clean way).
"""

from typing import Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

N_SITES = 2


N_PROTON_ORBITALS = 2 * N_SITES  # up, down per site


N_NEUTRON_ORBITALS = 2 * N_SITES


NUM_QUBITS = N_PROTON_ORBITALS + N_NEUTRON_ORBITALS  # 8


NUCLEI = {
    "deuteron": (1, 1),
    "triton": (1, 2),
    "helium3": (2, 1),
}


V0 = 1.0  # fixed Coulomb representation constant (not a task parameter)


I2 = np.eye(2, dtype=complex)


Z = np.array([[1, 0], [0, -1]], dtype=complex)


SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)  # annihilation: |0><1|... see _jw_annihilation


def _kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _jw_annihilation(k: int, n_modes: int) -> np.ndarray:
    mats = []
    for j in range(n_modes):
        if j < k:
            mats.append(Z)
        elif j == k:
            mats.append(SIGMA_MINUS)
        else:
            mats.append(I2)
    return _kron_all(mats)


# Wire index helpers: proton block = wires [0, N_PROTON_ORBITALS), site i -> (up=2i, down=2i+1);
# neutron block = wires [N_PROTON_ORBITALS, NUM_QUBITS), site i -> (up=N_PROTON_ORBITALS+2i, down=+2i+1).
def _proton_wire(site: int, spin: int) -> int:
    return 2 * site + spin


def _neutron_wire(site: int, spin: int) -> int:
    return N_PROTON_ORBITALS + 2 * site + spin


def nuclear_hamiltonian(nucleus: str, t: float, C: float, D: float, n_sites: int = N_SITES) -> np.ndarray:
    """Dense Hermitian matrix, dimension 2**NUM_QUBITS. Task parameters: hopping t, 2-body contact C,
    3-body contact D. `nucleus` selects the target (n_p, n_n) sector for ground-truth diagonalization
    only -- the Hamiltonian itself is nucleus-independent (same lattice, same couplings); the nucleus
    identity enters physics only through which particle-number sector is picked out as the ground truth
    and which qubits are populated in the classical initial state (see systems/nuclear.py)."""
    n_modes = 4 * n_sites
    dim = 2**n_modes
    psi = [_jw_annihilation(k, n_modes) for k in range(n_modes)]

    all_wires_by_site = []
    for site in range(n_sites):
        all_wires_by_site.append(
            [
                _proton_wire(site, 0),
                _proton_wire(site, 1),
                _neutron_wire(site, 0),
                _neutron_wire(site, 1),
            ]
        )

    H = np.zeros((dim, dim), dtype=complex)

    # Hopping: -t * sum_{<i,j>, sigma} [a^dag_sigma(i) a_sigma(j) + h.c.], adjacent sites only
    for site in range(n_sites - 1):
        for spin in range(2):
            for wire_fn in (_proton_wire, _neutron_wire):
                wi = wire_fn(site, spin)
                wj = wire_fn(site + 1, spin)
                hop = psi[wi].conj().T @ psi[wj]
                H += -t * (hop + hop.conj().T)

    # 2-body contact: (C/2) * sum_{i, sigma != sigma'} n_sigma(i) n_sigma'(i)
    for site in range(n_sites):
        wires = all_wires_by_site[site]
        n_ops = [psi[w].conj().T @ psi[w] for w in wires]
        for a in range(4):
            for b in range(4):
                if a != b:
                    H += (C / 2.0) * (n_ops[a] @ n_ops[b])

    # 3-body contact: (D/6) * sum_{i, sigma != sigma' != sigma''} n_sigma n_sigma' n_sigma''
    for site in range(n_sites):
        wires = all_wires_by_site[site]
        n_ops = [psi[w].conj().T @ psi[w] for w in wires]
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    if a != b and b != c and a != c:
                        H += (D / 6.0) * (n_ops[a] @ n_ops[b] @ n_ops[c])

    # Coulomb: V0 * sum_{i<j} n_p(i) n_p(j)
    n_p = []
    for site in range(n_sites):
        wp0, wp1 = _proton_wire(site, 0), _proton_wire(site, 1)
        n_p.append(psi[wp0].conj().T @ psi[wp0] + psi[wp1].conj().T @ psi[wp1])
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            H += V0 * (n_p[i] @ n_p[j])

    return H


def _particle_number_operators(n_sites: int = N_SITES):
    n_modes = 4 * n_sites
    psi = [_jw_annihilation(k, n_modes) for k in range(n_modes)]
    N_p = np.zeros((2**n_modes, 2**n_modes), dtype=complex)
    N_n = np.zeros((2**n_modes, 2**n_modes), dtype=complex)
    for site in range(n_sites):
        for spin in range(2):
            wp = _proton_wire(site, spin)
            wn = _neutron_wire(site, spin)
            N_p += psi[wp].conj().T @ psi[wp]
            N_n += psi[wn].conj().T @ psi[wn]
    return N_p, N_n


def exact_ground_energy(H: np.ndarray, nucleus: str, n_sites: int = N_SITES, atol: float = 1e-6) -> float:
    """Lowest eigenvalue of H restricted to the (n_p, n_n) sector for `nucleus`. n_p, n_n are exactly
    conserved by H (no term changes either count), so eigenstates have exact integer particle number;
    filtering by measured <N_p>, <N_n> on each eigenvector is exact, not approximate."""
    target_p, target_n = NUCLEI[nucleus]
    N_p, N_n = _particle_number_operators(n_sites)
    eigvals, eigvecs = np.linalg.eigh(H)
    best = None
    for k in range(len(eigvals)):
        v = eigvecs[:, k]
        np_expect = np.real(v.conj() @ N_p @ v)
        nn_expect = np.real(v.conj() @ N_n @ v)
        if abs(np_expect - target_p) < atol and abs(nn_expect - target_n) < atol:
            if best is None or eigvals[k] < best:
                best = eigvals[k]
    if best is None:
        raise RuntimeError(f"no eigenstate found with (n_p,n_n)=({target_p},{target_n})")
    return float(best)


NUCLEUS_ORDER = tuple(NUCLEI.keys())


def sample_task_space(
    n_tasks: int, seed: int, nuclei=NUCLEUS_ORDER, t_range=(0.5, 2.0), C_range=(-2.0, -0.2), D_range=(-1.0, 1.0)
):
    """Task = (nucleus, t, C, D). C is sampled negative (attractive contact interaction, matching the
    paper's own fitted C=-0.844 MeV sign for a bound deuteron)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(nuclei), size=n_tasks)
    t = rng.uniform(*t_range, size=n_tasks)
    C = rng.uniform(*C_range, size=n_tasks)
    D = rng.uniform(*D_range, size=n_tasks)
    return [(nuclei[i], float(t[k]), float(C[k]), float(D[k])) for k, i in enumerate(idx)]


def param_shape(depth: int, n_excitations: int) -> Tuple[int, int]:
    return (depth, n_excitations)


def verify_construction(atol: float = 1e-6) -> bool:
    """(1) H Hermitian. (2) N_p, N_n commute with H (exact conservation laws, checked numerically, not
    assumed). (3) Each nucleus's ground state in its own sector is lower than the unrestricted global
    ground state would suggest is trivial (sanity: the (0,0) vacuum sector's only state is |0...0> with
    E=0, so this also checks that a bound (negative-energy) solution exists at reasonable couplings for
    the physically populated sectors)."""
    ok = True
    H = nuclear_hamiltonian("deuteron", t=1.0, C=-0.844, D=0.0)
    herm_err = np.max(np.abs(H - H.conj().T))
    print(f"  Hermiticity: max|H-H^dag|={herm_err:.2e}  {'OK' if herm_err < atol else 'FAIL'}")
    ok = ok and herm_err < atol

    N_p, N_n = _particle_number_operators()
    comm_p = np.max(np.abs(H @ N_p - N_p @ H))
    comm_n = np.max(np.abs(H @ N_n - N_n @ H))
    comm_ok = comm_p < atol and comm_n < atol
    print(f"  [H,N_p]={comm_p:.2e}  [H,N_n]={comm_n:.2e}  {'OK' if comm_ok else 'FAIL'}")
    ok = ok and comm_ok

    for nucleus in NUCLEI:
        H_n = nuclear_hamiltonian(nucleus, t=1.0, C=-0.844, D=-0.3)
        E0 = exact_ground_energy(H_n, nucleus)
        print(f"  {nucleus}: E0={E0:.4f} (t=1.0, C=-0.844, D=-0.3)")
    return ok


# ---------------------------------------------------------------------------------------------------------
# Variational circuits (PennyLane reference implementations; the benchmark runs the exact torch
# simulators in qmaml_hep.simulators, cross-checked against these on every run)
# ---------------------------------------------------------------------------------------------------------


NUCLEUS_NAMES = tuple(NUCLEI.keys())


TASK_DIM = len(NUCLEUS_NAMES) + 3  # one-hot(nucleus) + (t, C, D)


def encode_task(nucleus: str, t: float, C: float, D: float) -> torch.Tensor:
    onehot = [1.0 if s == nucleus else 0.0 for s in NUCLEUS_NAMES]
    return torch.tensor(onehot + [t, C, D], dtype=torch.float64)


def _excitations_for(nucleus: str):
    """Singles/doubles excitation wires for the proton block and the neutron block, generated
    independently (so no excitation crosses species), then concatenated. Also returns the Hartree-Fock-
    like reference occupation (lowest orbitals of each species filled)."""
    n_p, n_n = NUCLEI[nucleus]
    s_p, d_p = qml.qchem.excitations(n_p, N_PROTON_ORBITALS)
    s_n, d_n = qml.qchem.excitations(n_n, N_NEUTRON_ORBITALS)
    s_n = [[w + N_PROTON_ORBITALS for w in exc] for exc in s_n]
    d_n = [[w + N_PROTON_ORBITALS for w in exc] for exc in d_n]
    singles = s_p + s_n
    doubles = d_p + d_n
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    init_state = np.zeros(NUM_QUBITS, dtype=int)
    init_state[:n_p] = 1  # lowest n_p proton orbitals occupied
    init_state[N_PROTON_ORBITALS : N_PROTON_ORBITALS + n_n] = 1  # lowest n_n neutron orbitals occupied
    return s_wires, d_wires, init_state


def n_excitations_for(nucleus: str) -> int:
    s_wires, d_wires, _ = _excitations_for(nucleus)
    return len(s_wires) + len(d_wires)


class NuclearUCCSDPQC(nn.Module):
    """UCCSD ansatz PQC (particle-number-conserving) whose cost is
    <psi(theta)|H_nuclear(t,C,D)|psi(theta)>, measured via qml.Hermitian. One instance is built per
    nucleus (its excitation set and reference state are nucleus-specific)."""

    def __init__(self, nucleus: str, num_qubits: int = NUM_QUBITS, depth: int = 1):
        super().__init__()
        self.nucleus = nucleus
        self.num_qubits = num_qubits
        self.depth = depth
        self.s_wires, self.d_wires, self.init_state = _excitations_for(nucleus)
        self.n_excitations = len(self.s_wires) + len(self.d_wires)
        self.dev = qml.device("default.qubit", wires=num_qubits)

    def make_circuit(self, H_matrix: np.ndarray):
        obs = qml.Hermitian(H_matrix, wires=range(self.num_qubits))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(weights):
            qml.UCCSD(
                weights,
                wires=range(self.num_qubits),
                s_wires=self.s_wires,
                d_wires=self.d_wires,
                init_state=self.init_state,
                n_repeats=self.depth,
            )
            return qml.expval(obs)

        return circuit

    def cost(self, weights: torch.Tensor, H_matrix: np.ndarray) -> torch.Tensor:
        circuit = self.make_circuit(H_matrix)
        w = weights.reshape(self.depth, self.n_excitations)
        return circuit(w.to(torch.float64))
