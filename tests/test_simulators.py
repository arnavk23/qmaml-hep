"""The exact torch simulators against the PennyLane reference circuits, for every track and every
state-preparation option. The benchmark repeats this check at the start of each run."""

import pytest
import torch

from qmaml_hep.simulators import verify
from qmaml_hep.tracks import TRACKS, check_fast, make_fast

FAST = [k for k in TRACKS if k not in ("H_nuclear", "V_heisenberg")]


@pytest.mark.parametrize("key", FAST)
def test_track_simulator_matches_pennylane(key):
    tr = TRACKS[key]()
    orig = make_fast(tr)
    assert orig is not None, f"{key} has no fast simulator"
    for task in tr["sample"](2, 0):
        kws = [{}] + [kwf(task) for _, kwf in tr["extra"].values()]
        if key == "F_susyqm":
            kws += [{"fermion_occupied": True}, {"fermion_occupied": False}]
        check_fast(tr["pqc"]("all"), orig, tr["ham"](task), tr["shape"]("all"), kws)


@pytest.mark.slow
def test_nuclear_uccsd_simulator_matches_pennylane():
    tr = TRACKS["H_nuclear"]()
    origs = make_fast(tr)
    for task in tr["sample"](6, 0):
        g = tr["group"](task)
        check_fast(tr["pqc"](g), origs[g], tr["ham"](task), tr["shape"](g))


@pytest.mark.parametrize("n,depth", [(2, 3), (4, 2), (5, 4)])
def test_classifier_circuit_matches_pennylane(n, depth):
    err_e, err_g = verify(n, depth)
    assert err_e < 1e-12 and err_g < 1e-12


def test_simulators_are_deterministic():
    tr = TRACKS["A_schwinger"]()
    make_fast(tr)
    H = tr["ham"](tr["sample"](1, 0)[0])
    w = torch.rand(tr["shape"]("all"), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert float(tr["pqc"]("all").cost(w, H)) == float(tr["pqc"]("all").cost(w.clone(), H))
