"""Spectral visualization functions for magnetic Laplacian."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import networkx as nx
import torch
from pathlib import Path
from typing import Optional, Dict
import pandas as pd


def plot_eigenvalue_spectrum(
    eigenvals_standard: np.ndarray,
    eigenvals_magnetic: np.ndarray,
    title: str = "Eigenvalue Spectrum",
    save_path: Optional[Path] = None
):
    """
    Plot eigenvalue spectra for standard and magnetic Laplacian.

    Args:
        eigenvals_standard: Eigenvalues from standard Laplacian
        eigenvals_magnetic: Eigenvalues from magnetic Laplacian
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Standard Laplacian spectrum
    ax1.stem(range(len(eigenvals_standard)), eigenvals_standard, basefmt=' ')
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title('Standard Laplacian', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Highlight spectral gap
    if len(eigenvals_standard) > 1:
        gap = eigenvals_standard[1] - eigenvals_standard[0]
        ax1.axvspan(-0.5, 1.5, alpha=0.1, color='green', label=f'Gap: {gap:.3f}')
        ax1.legend()

    # Annotate spectral radius
    rho = np.max(np.abs(eigenvals_standard))
    ax1.text(0.98, 0.98, f'ρ = {rho:.3f}', transform=ax1.transAxes,
             ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Magnetic Laplacian spectrum
    ax2.stem(range(len(eigenvals_magnetic)), eigenvals_magnetic, basefmt=' ',
             linefmt='r-', markerfmt='ro')
    ax2.set_xlabel('Index', fontsize=12)
    ax2.set_ylabel('Eigenvalue', fontsize=12)
    ax2.set_title('Magnetic Laplacian', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Highlight spectral gap
    if len(eigenvals_magnetic) > 1:
        gap = eigenvals_magnetic[1] - eigenvals_magnetic[0]
        ax2.axvspan(-0.5, 1.5, alpha=0.1, color='red', label=f'Gap: {gap:.3f}')
        ax2.legend()

    # Annotate spectral radius
    rho = np.max(np.abs(eigenvals_magnetic))
    ax2.text(0.98, 0.98, f'ρ = {rho:.3f}', transform=ax2.transAxes,
             ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_eigenvector_on_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    eigenvector: np.ndarray,
    title: str = "Eigenvector Visualization",
    is_complex: bool = False,
    save_path: Optional[Path] = None
):
    """
    Overlay eigenvector values on graph layout.

    Args:
        edge_index: [2, E] tensor of directed edges
        num_nodes: Number of nodes
        eigenvector: Eigenvector to visualize
        title: Plot title
        is_complex: Whether eigenvector is complex-valued
        save_path: Optional path to save figure
    """
    # Create NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)

    # Compute layout
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))

    if is_complex:
        # For complex eigenvectors: color = phase, size = magnitude
        magnitudes = np.abs(eigenvector)
        phases = np.angle(eigenvector)

        # Normalize magnitudes for node size
        node_sizes = 300 + 2000 * (magnitudes / magnitudes.max()) if magnitudes.max() > 0 else [300] * num_nodes

        # Use HSV colormap for phase (circular)
        # Map phase from [-π, π] to [0, 1]
        phase_normalized = (phases + np.pi) / (2 * np.pi)

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color=phase_normalized,
                              cmap='hsv', vmin=0, vmax=1, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                              arrowsize=20, width=1.5, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='white',
                               font_weight='bold', ax=ax)

        # Add colorbar for phase
        sm = plt.cm.ScalarMappable(cmap='hsv',
                                  norm=Normalize(vmin=-np.pi, vmax=np.pi))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Phase (radians)', fraction=0.046, pad=0.04)
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    else:
        # For real eigenvectors: color = value (diverging), size = |value|
        values = eigenvector.real if np.iscomplexobj(eigenvector) else eigenvector
        abs_values = np.abs(values)

        # Normalize for node size
        node_sizes = 300 + 2000 * (abs_values / abs_values.max()) if abs_values.max() > 0 else [300] * num_nodes

        # Use diverging colormap (blue-white-red)
        vmax = np.max(np.abs(values))
        vmin = -vmax if vmax > 0 else -1

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color=values,
                              cmap='seismic', vmin=vmin, vmax=vmax, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                              arrowsize=20, width=1.5, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black',
                               font_weight='bold', ax=ax)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='seismic',
                                  norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Eigenvector value', fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spectrum_comparison(
    edge_index: torch.Tensor,
    num_nodes: int,
    q: float,
    graph_name: str,
    save_dir: Path,
    spectrum: Optional[Dict] = None
):
    """
    Create 4-panel comparison figure.

    Args:
        edge_index: [2, E] tensor of directed edges
        num_nodes: Number of nodes
        q: Phase parameter
        graph_name: Name of the graph
        save_dir: Directory to save figure
        spectrum: Precomputed spectrum (optional, will compute if not provided)
    """
    from ..graphs.spectral import compute_laplacian_spectrum

    if spectrum is None:
        spectrum = compute_laplacian_spectrum(edge_index, num_nodes, q)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: Graph structure
    ax1 = fig.add_subplot(gs[0, 0])
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)

    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue',
                          edgecolors='black', linewidths=2, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                          arrowsize=20, width=2, alpha=0.7, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax1)
    ax1.set_title('A: Graph Structure', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Panel B: Eigenvalue spectrum comparison
    ax2 = fig.add_subplot(gs[0, 1])
    eigenvals_std = spectrum['eigenvalues_standard']
    eigenvals_mag = spectrum['eigenvalues_magnetic']

    x_std = np.arange(len(eigenvals_std))
    x_mag = np.arange(len(eigenvals_mag)) + 0.3

    ax2.stem(x_std, eigenvals_std, basefmt=' ', label='Standard',
             linefmt='b-', markerfmt='bo')
    ax2.stem(x_mag, eigenvals_mag, basefmt=' ', label='Magnetic',
             linefmt='r-', markerfmt='ro')

    ax2.set_xlabel('Eigenvalue Index', fontsize=12)
    ax2.set_ylabel('Eigenvalue', fontsize=12)
    ax2.set_title('B: Eigenvalue Spectrum', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Panel C: Second eigenvector from standard Laplacian (Fiedler)
    ax3 = fig.add_subplot(gs[1, 0])
    eigenvec_std = spectrum['eigenvectors_standard'][:, 1] if num_nodes > 1 else spectrum['eigenvectors_standard'][:, 0]
    values_std = eigenvec_std
    abs_values = np.abs(values_std)
    node_sizes = 300 + 2000 * (abs_values / abs_values.max()) if abs_values.max() > 0 else [300] * num_nodes
    vmax = np.max(np.abs(values_std))
    vmin = -vmax if vmax > 0 else -1

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=values_std,
                          cmap='seismic', vmin=vmin, vmax=vmax, ax=ax3)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                          arrowsize=15, width=1.5, alpha=0.6, ax=ax3)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black',
                           font_weight='bold', ax=ax3)

    sm = plt.cm.ScalarMappable(cmap='seismic', norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax3, label='Value', fraction=0.046, pad=0.04)
    ax3.set_title('C: Standard Laplacian Eigenvector 2', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Panel D: Second eigenvector from magnetic Laplacian
    ax4 = fig.add_subplot(gs[1, 1])
    eigenvec_mag = spectrum['eigenvectors_magnetic'][:, 1] if num_nodes > 1 else spectrum['eigenvectors_magnetic'][:, 0]
    magnitudes = np.abs(eigenvec_mag)
    phases = np.angle(eigenvec_mag)
    node_sizes = 300 + 2000 * (magnitudes / magnitudes.max()) if magnitudes.max() > 0 else [300] * num_nodes
    phase_normalized = (phases + np.pi) / (2 * np.pi)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=phase_normalized,
                          cmap='hsv', vmin=0, vmax=1, ax=ax4)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                          arrowsize=15, width=1.5, alpha=0.6, ax=ax4)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white',
                           font_weight='bold', ax=ax4)

    sm = plt.cm.ScalarMappable(cmap='hsv', norm=Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4, label='Phase (rad)', fraction=0.046, pad=0.04)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels(['-π', '0', 'π'])
    ax4.set_title('D: Magnetic Laplacian Eigenvector 2', fontsize=14, fontweight='bold')
    ax4.axis('off')

    plt.suptitle(f'{graph_name} (q={q})', fontsize=16, fontweight='bold', y=0.98)

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path_png = save_dir / f"{graph_name}.png"
    save_path_pdf = save_dir / f"{graph_name}.pdf"
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close()


def plot_parameter_sweep(
    sweep_df: pd.DataFrame,
    graph_name: str,
    save_path: Optional[Path] = None
):
    """
    Plot how spectral properties vary with q parameter.

    Args:
        sweep_df: DataFrame with columns: q, spectral_radius_mag, spectral_gap_mag, etc.
        graph_name: Name of the graph
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot 1: Spectral radius
    ax = axes[0]
    ax.plot(sweep_df['q'], sweep_df['spectral_radius_standard'], 'b-o', label='Standard', linewidth=2)
    ax.plot(sweep_df['q'], sweep_df['spectral_radius_magnetic'], 'r-s', label='Magnetic', linewidth=2)
    ax.set_xlabel('q parameter', fontsize=12)
    ax.set_ylabel('Spectral Radius', fontsize=12)
    ax.set_title('Spectral Radius vs q', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='q=0.1')
    ax.axvline(x=0.25, color='gray', linestyle=':', alpha=0.5, label='q=0.25')

    # Plot 2: Spectral gap
    ax = axes[1]
    ax.plot(sweep_df['q'], sweep_df['spectral_gap_standard'], 'b-o', label='Standard', linewidth=2)
    ax.plot(sweep_df['q'], sweep_df['spectral_gap_magnetic'], 'r-s', label='Magnetic', linewidth=2)
    ax.set_xlabel('q parameter', fontsize=12)
    ax.set_ylabel('Spectral Gap', fontsize=12)
    ax.set_title('Spectral Gap vs q', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.25, color='gray', linestyle=':', alpha=0.5)

    # Plot 3: Spectral radius ratio
    ax = axes[2]
    ax.plot(sweep_df['q'], sweep_df['spectral_radius_ratio'], 'g-d', linewidth=2)
    ax.set_xlabel('q parameter', fontsize=12)
    ax.set_ylabel('Ratio (Magnetic / Standard)', fontsize=12)
    ax.set_title('Spectral Radius Ratio vs q', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.25, color='gray', linestyle=':', alpha=0.5)

    # Plot 4: Eigenvalue correlation
    ax = axes[3]
    ax.plot(sweep_df['q'], sweep_df['eigenvalue_correlation'], 'm-^', linewidth=2)
    ax.set_xlabel('q parameter', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Eigenvalue Correlation vs q', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.25, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim([0, 1.05])

    plt.suptitle(f'Parameter Sweep: {graph_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_eigenvector_heatmap(
    eigenvectors: np.ndarray,
    node_labels: Optional[np.ndarray] = None,
    is_complex: bool = False,
    save_path: Optional[Path] = None,
    title: str = "Eigenvector Heatmap"
):
    """
    Heatmap of eigenvectors.

    Args:
        eigenvectors: Matrix of eigenvectors (nodes x eigenvectors)
        node_labels: Optional node labels for sorting
        is_complex: Whether eigenvectors are complex-valued
        save_path: Optional path to save figure
        title: Plot title
    """
    if is_complex:
        # Create two heatmaps: magnitude and phase
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        magnitudes = np.abs(eigenvectors)
        phases = np.angle(eigenvectors)

        # Sort by labels if provided
        if node_labels is not None:
            sort_idx = np.argsort(node_labels)
            magnitudes = magnitudes[sort_idx, :]
            phases = phases[sort_idx, :]

        # Magnitude heatmap
        im1 = ax1.imshow(magnitudes, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_xlabel('Eigenvector Index', fontsize=12)
        ax1.set_ylabel('Node', fontsize=12)
        ax1.set_title('Magnitude', fontsize=14)
        plt.colorbar(im1, ax=ax1, label='|v_i|')

        # Phase heatmap
        im2 = ax2.imshow(phases, aspect='auto', cmap='twilight',
                        interpolation='nearest', vmin=-np.pi, vmax=np.pi)
        ax2.set_xlabel('Eigenvector Index', fontsize=12)
        ax2.set_ylabel('Node', fontsize=12)
        ax2.set_title('Phase', fontsize=14)
        cbar = plt.colorbar(im2, ax=ax2, label='arg(v_i)')
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels(['-π', '0', 'π'])

        plt.suptitle(title, fontsize=16, y=1.02)

    else:
        # Single heatmap for real eigenvectors
        fig, ax = plt.subplots(figsize=(10, 6))

        values = eigenvectors.real if np.iscomplexobj(eigenvectors) else eigenvectors

        # Sort by labels if provided
        if node_labels is not None:
            sort_idx = np.argsort(node_labels)
            values = values[sort_idx, :]

        vmax = np.max(np.abs(values))
        im = ax.imshow(values, aspect='auto', cmap='seismic',
                      interpolation='nearest', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('Eigenvector Index', fontsize=12)
        ax.set_ylabel('Node', fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, label='v_i')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
