"""Standard Graph Convolutional Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """2-layer Graph Convolutional Network."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_dim]
            A: Normalized adjacency matrix [N, N]

        Returns:
            Log probabilities [N, out_dim]
        """
        # Layer 1: aggregate -> transform -> activate
        x = A @ x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 2: aggregate -> transform
        x = A @ x
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
