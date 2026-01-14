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


def generate_small_graph(name: str) -> tuple[torch.Tensor, int, str]:
    """
    Generate pedagogical graphs for spectral analysis.

    Args:
        name: Graph name (see options below)

    Returns:
        edge_index: [2, E] tensor of directed edges
        num_nodes: Number of nodes in the graph
        description: String explaining the graph's pedagogical purpose

    Available graphs:
        - triangle_cycle: 3-node directed cycle
        - path_4: 4-node directed path
        - diamond: 4-node diamond (two parallel paths)
        - cycle_4: 4-node directed cycle
        - star_out: 5-node star (hub broadcasts)
        - star_in: 5-node star (hub aggregates)
        - hexagon: 6-node directed cycle
        - two_triangles: 6-node two-community graph
        - hierarchical_3level: 9-node hierarchical structure
        - complete_4: 4-node complete graph (all bidirectional)
    """
    if name == "triangle_cycle":
        # 0→1→2→0 (directed cycle)
        edges = [[0, 1], [1, 2], [2, 0]]
        num_nodes = 3
        description = "Simplest non-trivial cycle showing direction encoding via phase"

    elif name == "path_4":
        # 0→1→2→3 (directed path, no cycles)
        edges = [[0, 1], [1, 2], [2, 3]]
        num_nodes = 4
        description = "Hierarchical flow without cycles - eigenvectors show gradient"

    elif name == "diamond":
        # 0→1, 0→2, 1→3, 2→3 (two parallel paths)
        edges = [[0, 1], [0, 2], [1, 3], [2, 3]]
        num_nodes = 4
        description = "Multiple pathways demonstrating eigenvector localization"

    elif name == "cycle_4":
        # 0→1→2→3→0 (simple cycle)
        edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        num_nodes = 4
        description = "Simple cycle for parameter sweeps"

    elif name == "star_out":
        # 0→1, 0→2, 0→3, 0→4 (broadcast pattern)
        edges = [[0, 1], [0, 2], [0, 3], [0, 4]]
        num_nodes = 5
        description = "Hub centrality (broadcast) - eigenvector localization on hub"

    elif name == "star_in":
        # 1→0, 2→0, 3→0, 4→0 (aggregation pattern)
        edges = [[1, 0], [2, 0], [3, 0], [4, 0]]
        num_nodes = 5
        description = "Hub centrality (aggregation) - contrast with star_out"

    elif name == "hexagon":
        # 0→1→2→3→4→5→0 (6-node cycle)
        edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
        num_nodes = 6
        description = "Larger cycle for structural balance exploration"

    elif name == "two_triangles":
        # Two triangles (0,1,2) and (3,4,5) with bidirectional bridge 2↔3
        edges = [
            # Triangle 1: 0→1→2→0
            [0, 1], [1, 2], [2, 0],
            # Triangle 2: 3→4→5→3
            [3, 4], [4, 5], [5, 3],
            # Bridge (bidirectional)
            [2, 3], [3, 2]
        ]
        num_nodes = 6
        description = "Community structure detection via Fiedler vector"

    elif name == "hierarchical_3level":
        # 3 levels × 3 nodes, edges flow forward
        # Level 0: [0,1,2], Level 1: [3,4,5], Level 2: [6,7,8]
        edges = [
            # Level 0 → Level 1
            [0, 3], [0, 4],
            [1, 4], [1, 5],
            [2, 5],
            # Level 1 → Level 2
            [3, 6],
            [4, 7],
            [5, 8],
        ]
        num_nodes = 9
        description = "Miniature hierarchical DAG - eigenvectors align with levels"

    elif name == "complete_4":
        # All edges bidirectional (fully connected)
        edges = [
            [0, 1], [1, 0],
            [0, 2], [2, 0],
            [0, 3], [3, 0],
            [1, 2], [2, 1],
            [1, 3], [3, 1],
            [2, 3], [3, 2],
        ]
        num_nodes = 4
        description = "Baseline - no directional bias, all edges reciprocal"

    else:
        raise ValueError(f"Unknown graph name: {name}. See docstring for available options.")

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index, num_nodes, description
