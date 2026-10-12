"""The Q-MAML Learner: a classical network that maps a task descriptor to initial circuit angles.

Gradient flow (Algorithm 1, no inner loop)::

    descriptor x  --Learner (float64 MLP)-->  theta0 = s * f_W(x)  --circuit (complex128 statevector)-->  E(theta0)
         ^                                                                                                  |
         +---------------------------- dE/dW by autograd through the circuit ------------------------------+

Every tensor on this path is float64 (real) or complex128 (amplitudes); there is no float32 cast anywhere
between the Learner's weights and the energy. ``tests/test_gradient_flow.py`` checks this end to end.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class Learner(nn.Module):
    """Fully connected ReLU network ``in_dim -> hidden x n_hidden -> prod(out_shape)``.

    ``n_hidden=2, hidden=256`` is the architecture of Lee, Cho & Kim (AAAI 2025). The output is reshaped to
    the circuit's parameter shape; callers scale it (``theta0 = s * learner(x)``) and cast the module with
    ``.double()`` so that the whole Algorithm-1 graph is float64.
    """

    def __init__(self, in_dim: int, out_shape: Sequence[int], hidden: int = 256, n_hidden: int = 2):
        super().__init__()
        if n_hidden < 1:
            raise ValueError("n_hidden must be >= 1")
        self.out_shape = tuple(int(s) for s in out_shape)
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, math.prod(self.out_shape)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(*self.out_shape)
