"""Symmetry-sector energy bound for the stated-settings Heisenberg replication (results/paper_replication/heisenberg_q12_L20.json).
The XX/YY/ZZ circuit from |0..0> conserves prod Z and prod X; the state keeps weight 1/2 in each prod X sector, so
E >= (e_{Z+,X+} + e_{Z+,X-})/2. Prints the bound and each initializer's final error per test task."""
import json, sys, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
import os; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from qmaml_hep.systems.heisenberg import heisenberg_xyz_hamiltonian
import pennylane as qml
d = json.load(open(os.path.join(ROOT, 'results/paper_replication/heisenberg_q12_L20.json')))
n = 12; N = 2**n
I = sp.identity(2, format='csr'); X = sp.csr_matrix([[0,1],[1,0]]); Z = sp.csr_matrix([[1,0],[0,-1]])
def prod(op):
    M = sp.csr_matrix([[1]])
    for _ in range(n): M = sp.kron(M, op, format='csr')
    return M
PZ, PX = prod(Z), prod(X)
# basis for Z+ sector split by X parity: states (|b> +- |~b>)/sqrt2 with even popcount
even = [b for b in range(N) if bin(b).count('1') % 2 == 0]
reps = sorted({min(b, b ^ (N-1)) for b in even})
def basis(sign):
    rows, cols, vals = [], [], []
    for j, b in enumerate(reps):
        rows += [b, b ^ (N-1)]; cols += [j, j]; vals += [1/np.sqrt(2), sign/np.sqrt(2)]
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, len(reps)))
Bp, Bm = basis(1), basis(-1)
out = []
for t, E0, curves in zip(d['tasks'], d['E0'], zip(*d['curves'].values())):
    H = heisenberg_xyz_hamiltonian(n, *t)
    Hm = qml.matrix(H, wire_order=range(n)) if not sp.issparse(H) else H
    Hm = sp.csr_matrix(Hm)
    ep = np.linalg.eigvalsh((Bp.T @ Hm @ Bp).toarray())[0]
    em = np.linalg.eigvalsh((Bm.T @ Hm @ Bm).toarray())[0]
    floor = 0.5 * (ep + em)
    fe = abs(floor - E0) / abs(E0)
    fins = {k: abs(c[-1] - E0) / abs(E0) for k, c in zip(d['curves'], curves)}
    out.append((t, E0, fe, fins))
    print(t, 'E0 %.4f floor_err %.2e' % (E0, fe), ' '.join('%s %.1e' % (k, v) for k, v in fins.items()))
fe = np.array([o[2] for o in out])
print('floor<1e-3:', (fe < 1e-3).sum(), '/16; median floor err %.3g max %.3g' % (np.median(fe), fe.max()))
for k in d['curves']:
    f = np.array([o[3][k] for o in out]); print(k, 'within 1e-3 rel of floor:', (np.abs(f - fe) < 1e-3).sum(), 'reach:', (f < 1e-3).sum(), 'reach where floor ok:', ((f<1e-3)&(fe<1e-3)).sum())
