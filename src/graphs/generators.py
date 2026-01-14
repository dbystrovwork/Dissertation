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


def generate_hierarchical_dag(
    n: int,
    num_levels: int,
    p_forward: float = 0.3,
    p_skip: float = 0.1,
    p_lateral: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate hierarchical DAG where direction = information flow.

    Node labels = hierarchy level (0 = input, num_levels-1 = output).
    Edges flow forward through levels, encoding class-relevant structure.

    Args:
        n: Number of nodes
        num_levels: Number of hierarchy levels (also number of classes)
        p_forward: Probability of edge to next level
        p_skip: Probability of edge skipping levels
        p_lateral: Probability of edge within same level

    Returns:
        edge_index: [2, E] tensor of directed edges (always i -> j where level[i] <= level[j])
        labels: [N] tensor of hierarchy levels
    """
    # Assign nodes to levels
    nodes_per_level = n // num_levels
    labels = []
    for level in range(num_levels):
        count = nodes_per_level + (1 if level < n % num_levels else 0)
        labels.extend([level] * count)
    labels = torch.tensor(labels)

    edges = []

    for i in range(n):
        level_i = labels[i].item()

        for j in range(i + 1, n):
            level_j = labels[j].item()

            # Only allow forward edges (i -> j where level_i <= level_j)
            if level_j < level_i:
                continue

            # Determine edge probability based on level difference
            level_diff = level_j - level_i

            if level_diff == 0:
                # Lateral edge within same level
                p = p_lateral
            elif level_diff == 1:
                # Forward edge to next level
                p = p_forward
            else:
                # Skip connection
                p = p_skip / level_diff  # Decay with distance

            if np.random.rand() < p:
                edges.append([i, j])

    if len(edges) == 0:
        # Ensure connectivity
        for level in range(num_levels - 1):
            i = np.random.choice([idx for idx, lbl in enumerate(labels) if lbl == level])
            j = np.random.choice([idx for idx, lbl in enumerate(labels) if lbl == level + 1])
            edges.append([i, j])

    return torch.tensor(edges, dtype=torch.long).t(), labels
