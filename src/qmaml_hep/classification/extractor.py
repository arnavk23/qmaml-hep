import torch
import torch.nn as nn


class TabularFeatureExtractor(nn.Module):
    """Frozen random MLP backbone (input -> hidden -> output_dim) plus a trainable linear projection to one
    encoding angle per qubit. ``embed`` returns the backbone features without gradient (the Learner's task
    embedding is their mean over the support set); ``forward`` returns the angles. float32 throughout; the
    circuit casts its inputs to float64."""

    def __init__(self, input_dim: int, output_dim: int, num_qubits: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.ReLU(),
        )
        self.to_angles = nn.Linear(output_dim, num_qubits)
        self.register_buffer("_mean", torch.zeros(input_dim))
        self.register_buffer("_std", torch.ones(input_dim))

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._mean.copy_(mean)
        self._std.copy_(std.clamp_min(1e-6))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self._mean) / self._std

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(self._norm(x))
        return self.to_angles(feat)


__all__ = ["TabularFeatureExtractor"]
