"""Q-network definition for value-based RL agents."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """A small MLP for estimating action values."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each discrete action."""
        return self.network(inputs)
