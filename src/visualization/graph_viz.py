"""Graph visualization utilities."""

from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Patch
import networkx as nx
from pathlib import Path


def visualize_graph(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    edge_phases: np.ndarray = None,
    title: str = "Graph",
    layout: str = "spring",
    node_size: int = 300,
    save_path: Path = None,
    figsize: tuple = (12, 8),
    show_legend: bool = True,
) -> None:
    """
    Visualize a graph with node labels and optional edge phases.

    Args:
        edge_index: [2, E] tensor of directed edges
        labels: [N] tensor of node labels (determines node color)
        edge_phases: [E] optional array of phase values for edge coloring
        title: Plot title
        layout: Layout algorithm ("spring", "circular", "kamada_kawai", "shell")
        node_size: Size of nodes
        save_path: Path to save figure (if None, displays interactively)
        figsize: Figure size (width, height)
        show_legend: Whether to show legend

    Features:
    - Colors nodes by label (discrete colormap)
    - Colors edges by phase (if provided, continuous colormap)
    - Arrow direction shows edge orientation
    - Legend for node labels
    - Colorbar for edge phases (if provided)
    """
    # Convert to NetworkX
    G = nx.DiGraph()
    n_nodes = len(labels)
    G.add_nodes_from(range(n_nodes))

    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # Compute layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=1.0/np.sqrt(n_nodes))
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize)

    # Node colors based on labels
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    unique_labels = np.unique(labels_np)
    n_labels = len(unique_labels)

    # Use a discrete colormap
    cmap_nodes = cm.get_cmap('tab10' if n_labels <= 10 else 'tab20')
    node_colors = [cmap_nodes(i / max(n_labels - 1, 1)) for i in labels_np]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        ax=ax,
        alpha=0.9
    )

    # Draw node labels (node IDs)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', ax=ax)

    # Draw edges
    if edge_phases is not None:
        # Color edges by phase
        # Normalize phases to [0, 1] for colormap
        phase_norm = Normalize(vmin=0, vmax=2*np.pi)
        cmap_edges = cm.get_cmap('twilight')  # Cyclic colormap for phases

        # Convert phases (q values) to actual phases (2πq mod 2π)
        actual_phases = (2 * np.pi * edge_phases) % (2 * np.pi)

        for i, (u, v) in enumerate(edges):
            edge_color = cmap_edges(phase_norm(actual_phases[i]))
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                edge_color=[edge_color],
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                width=2,
                ax=ax,
                alpha=0.6
            )

        # Add colorbar for edge phases
        sm = cm.ScalarMappable(cmap=cmap_edges, norm=phase_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Edge Phase (radians)', rotation=270, labelpad=20)
        cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    else:
        # Simple edge drawing without phase coloring
        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            width=1.5,
            edge_color='gray',
            ax=ax,
            alpha=0.5
        )

    # Legend for node labels
    if show_legend and n_labels > 1:
        legend_elements = [
            Patch(facecolor=cmap_nodes(i / max(n_labels - 1, 1)),
                  label=f'Label {int(label)}')
            for i, label in enumerate(unique_labels)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_csbm_structure(
    edge_index: torch.Tensor,
    metadata: dict,
    title: str = "CSBM Structure",
    save_path: Path = None,
    figsize: tuple = (14, 10),
) -> None:
    """
    Specialized visualization for CSBM showing hierarchical community structure.

    Args:
        edge_index: [2, E] tensor of directed edges
        metadata: Dict from generate_csbm containing:
            - 'sub_community_labels': Level-2 sub-community assignments
            - 'edge_phases': Phase for each edge
            - 'n_communities': Number of level-1 communities
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Features:
    - Level-1 communities: Spatial grouping
    - Level-2 sub-communities: Node colors within each level-1 community
    - Edge phases: Edge colors
    - Layout: Community-optimized (spring layout with community structure)
    """
    # Extract metadata
    level1_labels = metadata.get('level1_labels', None)
    if level1_labels is None:
        # Infer from sub_community labels and n_communities
        sub_labels = metadata['sub_community_labels'].numpy()
        n_communities = metadata['n_communities']
        n_nodes = len(sub_labels)
        community_size = n_nodes // n_communities
        level1_labels = np.repeat(np.arange(n_communities), community_size)
    else:
        level1_labels = level1_labels.numpy() if isinstance(level1_labels, torch.Tensor) else level1_labels

    sub_labels = metadata['sub_community_labels'].numpy()
    edge_phases = metadata['edge_phases']
    n_communities = metadata['n_communities']

    # Convert to NetworkX
    G = nx.DiGraph()
    n_nodes = len(sub_labels)
    G.add_nodes_from(range(n_nodes))

    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # Create community-aware layout
    # Use spring layout with community-based initial positions
    pos = {}
    community_centers = []

    # Position communities in a circle
    for comm_idx in range(n_communities):
        angle = 2 * np.pi * comm_idx / n_communities
        center_x = 2 * np.cos(angle)
        center_y = 2 * np.sin(angle)
        community_centers.append((center_x, center_y))

        # Nodes in this community
        nodes_in_comm = [i for i, lbl in enumerate(level1_labels) if lbl == comm_idx]

        # Sub-layout within community
        sub_G = G.subgraph(nodes_in_comm)
        if len(nodes_in_comm) > 0:
            sub_pos = nx.spring_layout(sub_G, seed=42, scale=0.8)
            for node, (x, y) in sub_pos.items():
                pos[node] = (center_x + x, center_y + y)

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize)

    # Node colors based on sub-community labels within each level-1 community
    n_subs = metadata['sub_communities'][0]  # Assume same for all communities
    cmap_subs = cm.get_cmap('Set3')

    node_colors = []
    for i in range(n_nodes):
        comm = level1_labels[i]
        sub = sub_labels[i]
        # Create unique color for each (community, sub-community) pair
        color_idx = (comm * n_subs + sub) / (n_communities * n_subs)
        node_colors.append(cmap_subs(color_idx))

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=400,
        ax=ax,
        alpha=0.9,
        edgecolors='black',
        linewidths=1.5
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_color='black', ax=ax)

    # Draw edges colored by phase
    phase_norm = Normalize(vmin=0, vmax=2*np.pi)
    cmap_edges = cm.get_cmap('twilight')
    actual_phases = (2 * np.pi * edge_phases) % (2 * np.pi)

    for i, (u, v) in enumerate(edges):
        edge_color = cmap_edges(phase_norm(actual_phases[i]))
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=[edge_color],
            arrows=True,
            arrowsize=12,
            arrowstyle='->',
            width=1.5,
            ax=ax,
            alpha=0.4
        )

    # Draw community boundaries (circles)
    for comm_idx, (cx, cy) in enumerate(community_centers):
        circle = plt.Circle((cx, cy), 1.2, color='black', fill=False,
                           linestyle='--', linewidth=2, alpha=0.3)
        ax.add_patch(circle)
        ax.text(cx, cy + 1.5, f'Community {comm_idx}',
               ha='center', fontsize=12, fontweight='bold')

    # Colorbar for edge phases
    sm = cm.ScalarMappable(cmap=cmap_edges, norm=phase_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Edge Phase (radians)', rotation=270, labelpad=20)
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    # Legend for sub-communities
    legend_elements = []
    for sub_idx in range(n_subs):
        color_idx = sub_idx / n_subs
        legend_elements.append(
            Patch(facecolor=cmap_subs(color_idx),
                  label=f'Sub-comm {sub_idx}', edgecolor='black')
        )
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             title='Sub-communities')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved CSBM visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_networks(
    graphs: list[tuple],
    save_path: Path = None,
    figsize: tuple = (20, 12),
) -> None:
    """
    Create multi-panel comparison of different networks.

    Args:
        graphs: List of tuples, each containing:
            (edge_index, labels, edge_phases, title)
            where edge_phases can be None
        save_path: Path to save figure
        figsize: Figure size

    Creates grid layout showing multiple graphs side-by-side with consistent styling.
    """
    n_graphs = len(graphs)

    # Determine grid layout
    if n_graphs <= 2:
        nrows, ncols = 1, n_graphs
    elif n_graphs <= 4:
        nrows, ncols = 2, 2
    elif n_graphs <= 6:
        nrows, ncols = 2, 3
    else:
        nrows = (n_graphs + 2) // 3
        ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_graphs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for idx, (edge_index, labels, edge_phases, title) in enumerate(graphs):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Convert to NetworkX
        G = nx.DiGraph()
        n_nodes = len(labels)
        G.add_nodes_from(range(n_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        # Layout
        if n_nodes < 50:
            pos = nx.spring_layout(G, seed=42, k=1.0/np.sqrt(n_nodes))
        else:
            pos = nx.kamada_kawai_layout(G)

        # Node colors
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        unique_labels = np.unique(labels_np)
        n_labels = len(unique_labels)
        cmap_nodes = cm.get_cmap('tab10' if n_labels <= 10 else 'tab20')
        node_colors = [cmap_nodes(i / max(n_labels - 1, 1)) for i in labels_np]

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=200 if n_nodes < 50 else 100,
            ax=ax,
            alpha=0.9
        )

        # Draw edges
        if edge_phases is not None and len(edge_phases) > 0:
            phase_norm = Normalize(vmin=0, vmax=2*np.pi)
            cmap_edges = cm.get_cmap('twilight')
            actual_phases = (2 * np.pi * edge_phases) % (2 * np.pi)

            for i, (u, v) in enumerate(edges):
                if i < len(actual_phases):
                    edge_color = cmap_edges(phase_norm(actual_phases[i]))
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=[(u, v)],
                        edge_color=[edge_color],
                        arrows=True,
                        arrowsize=10,
                        width=1,
                        ax=ax,
                        alpha=0.5
                    )
        else:
            nx.draw_networkx_edges(
                G, pos,
                arrows=True,
                arrowsize=10,
                width=1,
                edge_color='gray',
                ax=ax,
                alpha=0.4
            )

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_graphs, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved network comparison to: {save_path}")
    else:
        plt.show()

    plt.close()
