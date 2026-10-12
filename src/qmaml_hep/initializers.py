"""Task-independent ("classical") initializers, shared by every track.

All return float64 tensors of the requested shape. ``uniform`` and ``gaussian`` draw from torch's global
generator, so the benchmark's ``torch.manual_seed`` calls fix them.
"""

from __future__ import annotations

import numpy as np
import torch

# Initializers that are zero angles on top of a physics-motivated *state* (the state preparation is selected
# by a keyword passed to the circuit's cost, see qmaml_hep.tracks):
#   gauss_vacuum  Z2 gauge theory: start in the Gauss-law-satisfying vacuum (arXiv:2507.19203)
#   basis_paper   SUSY QM: the source paper's hand-picked basis state (fermion occupied for HO/AHO)
STATE_PREP_KINDS = ("gauss_vacuum", "basis_paper")


def classical_init(shape, kind: str, depth: int) -> torch.Tensor:
    """``zero``, ``pi``, ``uniform`` (U[-0.05 pi, 0.05 pi]) or ``gaussian`` (Zhang et al. 2022, S = 2)."""
    if kind == "zero" or kind in STATE_PREP_KINDS:
        return torch.zeros(shape, dtype=torch.float64)
    if kind == "pi":
        return torch.full(shape, float(np.pi), dtype=torch.float64)
    if kind == "uniform":
        alpha = 0.05
        return (torch.rand(shape, dtype=torch.float64) * 2 - 1) * alpha * np.pi
    if kind == "gaussian":
        S = 2
        gamma2 = 1.0 / (4 * S * (depth + 2))
        return torch.normal(0.0, gamma2**0.5, size=shape, dtype=torch.float64)
    raise ValueError(f"unknown initializer {kind!r}")
