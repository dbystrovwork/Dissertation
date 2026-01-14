"""Graph generators."""

import torch
import numpy as np


def generate_sbm(
    n: int,
    k: int,
    p_in: float,
    p_out: float,
    balanced: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate directed stochastic block model.

    Args:
        n: Number of nodes
        k: Number of classes/blocks
        p_in: Edge probability within blocks
        p_out: Edge probability between blocks
        balanced: If True, consistent edge directions (structurally balanced).
                  If False, random directions (creates odd cycles, unbalanced).

    Returns:
        edge_index: [2, E] tensor of directed edges
        labels: [N] tensor of node labels
    """
    labels = torch.tensor([i % k for i in range(n)])
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if np.random.rand() < p:
                if balanced:
                    edges.append([i, j])  # Consistent: low -> high
                else:
                    # Random direction creates odd cycles
                    if np.random.rand() < 0.5:
                        edges.append([i, j])
                    else:
                        edges.append([j, i])

    if len(edges) == 0:
        edges = [[0, 1]]

    return torch.tensor(edges, dtype=torch.long).t(), labels
