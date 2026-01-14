"""Graph generators."""

from __future__ import annotations

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


def generate_directed_cycle(n: int, labels=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a pure directed cycle: 0→1→2→...→(n-1)→0.

    Args:
        n: Number of nodes
        labels: Optional labels for nodes. If None, uses sequential positions (0, 1, 2, ...)

    Returns:
        edge_index: [2, E] tensor of directed edges
        labels: Tensor of node labels
    """
    edges = [[i, (i + 1) % n] for i in range(n)]
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    if labels is None:
        labels = torch.arange(n, dtype=torch.long)
    else:
        labels = torch.tensor(labels, dtype=torch.long)

    return edge_index, labels


def generate_cycle_with_bidir(n: int, bidir_frac: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a cycle with some edges bidirectional.

    Args:
        n: Number of nodes
        bidir_frac: Fraction of edges to make bidirectional (0.0 to 1.0)
            - 0.0: pure directed cycle
            - 0.5: half edges bidirectional
            - 1.0: fully undirected cycle

    Returns:
        edge_index: [2, E] tensor of directed edges
        labels: Tensor of node labels (position in cycle)
    """
    edges = []

    # Start with directed cycle
    for i in range(n):
        j = (i + 1) % n
        edges.append([i, j])

        # Add reverse edge with probability bidir_frac
        if np.random.rand() < bidir_frac:
            edges.append([j, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    labels = torch.arange(n, dtype=torch.long)  # Position in cycle

    return edge_index, labels


def generate_tree(n: int, branching: int = 2, directed: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a tree structure.

    Args:
        n: Number of nodes (will be truncated to fit perfect tree if needed)
        branching: Branching factor (2 = binary tree, 3 = ternary, etc.)
        directed: If True, edges flow root→leaves; if False, random orientation

    Returns:
        edge_index: [2, E] tensor of directed edges
        labels: Tensor of node labels (depth in tree)
    """
    # Calculate tree depth
    depth = 0
    nodes_at_depth = 1
    total_nodes = 1

    while total_nodes + nodes_at_depth * branching <= n:
        depth += 1
        nodes_at_depth *= branching
        total_nodes += nodes_at_depth

    # Adjust n to perfect tree size
    n = total_nodes

    edges = []
    labels = torch.zeros(n, dtype=torch.long)

    # Build tree level by level
    node_idx = 0
    for d in range(depth):
        nodes_in_level = branching ** d
        for i in range(nodes_in_level):
            current_node = node_idx
            labels[current_node] = d

            for b in range(branching):
                child_idx = node_idx + nodes_in_level * branching + b
                if child_idx < n:
                    if directed:
                        # Root → leaves
                        edges.append([current_node, child_idx])
                    else:
                        # Random orientation
                        if np.random.rand() < 0.5:
                            edges.append([current_node, child_idx])
                        else:
                            edges.append([child_idx, current_node])

            node_idx += 1

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    return edge_index, labels


def generate_small_world(n: int, k: int = 4, p: float = 0.1, directed: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a small-world graph using Watts-Strogatz model (directed variant).

    Args:
        n: Number of nodes
        k: Each node connected to k nearest neighbors in ring (must be even)
        p: Probability of rewiring each edge
        directed: If True, preserve direction after rewiring

    Returns:
        edge_index: [2, E] tensor of directed edges
        labels: Tensor of node labels (original position in ring)
    """
    if k % 2 != 0:
        k = k + 1  # Make even

    edges = []

    # Start with directed ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            target = (i + j) % n
            edges.append([i, target])

    # Rewiring
    rewired_edges = []
    for u, v in edges:
        if np.random.rand() < p:
            # Rewire: keep u, choose new target
            new_v = np.random.randint(0, n)
            while new_v == u or [u, new_v] in rewired_edges:
                new_v = np.random.randint(0, n)
            rewired_edges.append([u, new_v])
        else:
            rewired_edges.append([u, v])

    edge_index = torch.tensor(rewired_edges, dtype=torch.long).t()
    labels = torch.arange(n, dtype=torch.long)  # Original position in ring

    return edge_index, labels


def generate_csbm(
    n_communities: int = 2,
    community_size: int = 60,
    p_in: float = 0.6,
    p_out: float = 0.1,
    eta: float = 0.0,
    sub_communities_per_comm: list[int] | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generate a Complex Stochastic Block Model (CSBM) with phase structure.

    This implements the CSBM as described in spectral clustering papers,
    with three key configurations:
    1. Balanced Phase-Structured CSBM (eta=0): Clean phase offsets between sub-communities
    2. Phase-Mixed CSBM (eta>0): Random phase reassignment with probability eta
    3. Use generate_csbm_unbalanced() for strictly unbalanced graphs

    Args:
        n_communities: Number of level-1 communities (k)
        community_size: Size of each level-1 community (n_p)
        p_in: Probability of edge within a community
        p_out: Probability of edge between communities
        eta: Mixing probability to randomize phases within communities (0 = balanced, 1 = fully mixed)
        sub_communities_per_comm: List of sub-community counts [l1, l2, ..., lk] for each community.
                                  If None, defaults to [2, 2, ..., 2]
        seed: Random seed for reproducibility

    Returns:
        edge_index: [2, E] edge indices
        labels: Node labels (level-1 community assignments)
        metadata: Dict with:
            - 'sub_community_labels': Level-2 community assignments
            - 'edge_phases': Phase for each edge in edge_index (as q values where phase = 2πq)
            - 'n_communities': Number of level-1 communities
            - 'sub_communities': List of sub-community counts
            - 'eta': Mixing probability used

    Example:
        # Balanced CSBM with 2 communities, each with 3 sub-communities
        edge_index, labels, meta = generate_csbm(
            n_communities=2, community_size=60, p_in=0.6, p_out=0.1,
            eta=0.0, sub_communities_per_comm=[3, 3]
        )
        # Use meta['edge_phases'] to build complex-weighted matrices with varying q
    """
    if seed is not None:
        np.random.seed(seed)

    if sub_communities_per_comm is None:
        sub_communities_per_comm = [2] * n_communities

    assert len(sub_communities_per_comm) == n_communities, \
        "sub_communities_per_comm must have length n_communities"

    n_total = n_communities * community_size

    # Assign nodes to level-1 communities
    level1_labels = np.repeat(np.arange(n_communities), community_size)

    # Assign nodes to level-2 sub-communities
    level2_labels = np.zeros(n_total, dtype=int)
    for comm_idx in range(n_communities):
        start_node = comm_idx * community_size
        n_sub = sub_communities_per_comm[comm_idx]

        # Divide community into sub-communities (approximately equal)
        sub_sizes = [community_size // n_sub] * n_sub
        for i in range(community_size % n_sub):
            sub_sizes[i] += 1

        node_idx = start_node
        for sub_idx, sub_size in enumerate(sub_sizes):
            level2_labels[node_idx:node_idx + sub_size] = sub_idx
            node_idx += sub_size

    # Generate edges based on SBM
    edges = []
    edge_phases = []

    for i in range(n_total):
        for j in range(i + 1, n_total):
            # Determine edge probability
            if level1_labels[i] == level1_labels[j]:
                p_edge = p_in
            else:
                p_edge = p_out

            # Add edge with probability p_edge (undirected)
            if np.random.rand() < p_edge:
                comm_i = level1_labels[i]

                # Determine phase based on sub-communities
                if level1_labels[i] == level1_labels[j]:
                    # Same level-1 community: use sub-community phase structure
                    sub_i = level2_labels[i]
                    sub_j = level2_labels[j]
                    n_sub = sub_communities_per_comm[comm_i]

                    # Initial balanced configuration: phase = ((sub_j - sub_i) * 2π / n_sub) mod 2π
                    # Store as q where phase = 2πq
                    phase_diff = (sub_j - sub_i) % n_sub
                    q = phase_diff / n_sub

                    # Apply mixing with probability eta
                    if np.random.rand() < eta:
                        # Randomize phase to another value in the same community
                        q = np.random.choice(np.arange(n_sub)) / n_sub

                else:
                    # Different level-1 communities: phase = 0 (or could be random)
                    q = 0.0

                # Add both directions for undirected edge
                edges.append([i, j])
                edge_phases.append(q)
                edges.append([j, i])
                edge_phases.append(-q % 1.0)  # Conjugate phase for reverse direction

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    labels = torch.tensor(level1_labels, dtype=torch.long)
    edge_phases_tensor = np.array(edge_phases)

    metadata = {
        'sub_community_labels': torch.tensor(level2_labels, dtype=torch.long),
        'edge_phases': edge_phases_tensor,
        'n_communities': n_communities,
        'sub_communities': sub_communities_per_comm,
        'eta': eta,
    }

    return edge_index, labels, metadata


def generate_csbm_unbalanced(
    n_communities: int = 2,
    community_size: int = 60,
    p_in: float = 0.6,
    p_out: float = 0.1,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generate a strictly unbalanced CSBM with completely random phases.

    This creates frustrated cycles where phase accumulation is non-zero,
    testing MagNet's ability to exploit directional information in
    unbalanced graphs.

    Args:
        n_communities: Number of level-1 communities
        community_size: Size of each community
        p_in: Probability of edge within community
        p_out: Probability of edge between communities
        seed: Random seed

    Returns:
        edge_index: [2, E] edge indices
        labels: Node labels (community assignments)
        metadata: Dict with 'edge_phases' and other info
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = n_communities * community_size
    level1_labels = np.repeat(np.arange(n_communities), community_size)

    edges = []
    edge_phases = []

    for i in range(n_total):
        for j in range(i + 1, n_total):
            # Determine edge probability
            if level1_labels[i] == level1_labels[j]:
                p_edge = p_in
            else:
                p_edge = p_out

            if np.random.rand() < p_edge:
                # Completely random phase (creates strict unbalance)
                q = np.random.rand()

                edges.append([i, j])
                edge_phases.append(q)
                edges.append([j, i])
                edge_phases.append(-q % 1.0)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    labels = torch.tensor(level1_labels, dtype=torch.long)
    edge_phases_tensor = np.array(edge_phases)

    metadata = {
        'sub_community_labels': labels.clone(),  # No sub-communities in unbalanced version
        'edge_phases': edge_phases_tensor,
        'n_communities': n_communities,
        'sub_communities': [1] * n_communities,
        'eta': None,  # Not applicable
    }

    return edge_index, labels, metadata
