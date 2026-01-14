"""Magnetic Graph Neural Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MagNet(nn.Module):
    """
    Magnetic Graph Neural Network.

    Uses complex adjacency to encode edge direction.
    Processes real and imaginary channels separately then combines.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # Separate transforms for real and imaginary parts
        self.fc1_re = nn.Linear(in_dim, hidden_dim)
        self.fc1_im = nn.Linear(in_dim, hidden_dim)
        self.fc2_re = nn.Linear(hidden_dim, out_dim)
        self.fc2_im = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, A_mag: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_dim] (real)
            A_mag: Magnetic adjacency matrix [N, N] (complex)

        Returns:
            Log probabilities [N, out_dim]
        """
        # Layer 1: complex aggregation
        x_c = x.to(torch.complex64)
        h = A_mag @ x_c

        # Process real and imaginary separately, then combine
        h_re = F.relu(self.fc1_re(h.real))
        h_im = F.relu(self.fc1_im(h.imag))
        h = h_re + h_im
        h = F.dropout(h, p=0.5, training=self.training)

        # Layer 2: complex aggregation
        h_c = h.to(torch.complex64)
        h = A_mag @ h_c

        out = self.fc2_re(h.real) + self.fc2_im(h.imag)

        return F.log_softmax(out, dim=1)
