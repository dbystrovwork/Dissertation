"""Adjacency matrix construction."""

from __future__ import annotations

import torch
import numpy as np


def build_matrices(edge_index: torch.Tensor, num_nodes: int, q: float = 0.1):
    """
    Build normalized adjacency matrices for GCN and MagNet.

    Args:
        edge_index: [2, E] tensor of directed edges
        num_nodes: Number of nodes
        q: Phase parameter (0.1 gives good real/imag balance)

    Returns:
        A_gcn: Symmetric normalized adjacency (real)
        A_mag: Hermitian normalized adjacency (complex)
    """
    # Symmetric adjacency (treat all edges as undirected)
    A_sym = torch.zeros(num_nodes, num_nodes)
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        A_sym[u, v] = 1.0
        A_sym[v, u] = 1.0

    # Add self-loops
    A_sym = A_sym + torch.eye(num_nodes)

    # Degree normalization
    D = A_sym.sum(dim=1)
    D_inv_sqrt = torch.zeros(num_nodes)
    D_inv_sqrt[D > 0] = 1.0 / torch.sqrt(D[D > 0])
    D_mat = torch.diag(D_inv_sqrt)

    # GCN: symmetric normalized
    A_gcn = D_mat @ A_sym @ D_mat

    # MagNet: Hermitian with phase encoding direction
    A_mag = torch.eye(num_nodes, dtype=torch.complex64)  # Self-loops
    phase = 2 * np.pi * q
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        A_mag[u, v] = np.exp(1j * phase)   # u -> v
        A_mag[v, u] = np.exp(-1j * phase)  # Hermitian conjugate

    # Same normalization
    D_mat_c = D_mat.to(torch.complex64)
    A_mag = D_mat_c @ A_mag @ D_mat_c

    return A_gcn, A_mag


def build_matrices_with_phases(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_phases: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build adjacency matrices with per-edge phase values (for CSBM).

    IMPORTANT: The CSBM generators (generate_csbm, generate_csbm_unbalanced)
    already provide BOTH directions for each undirected edge:
      - Edge i→j with phase q
      - Edge j→i with phase -q (conjugate)

    So edge_index contains all directed edges explicitly, and we just apply
    the phases as given to construct the Hermitian matrix.

    Args:
        edge_index: [2, E] tensor of directed edges (E = 2 * num_undirected_edges)
        num_nodes: Number of nodes
        edge_phases: [E] array of phase parameters q (where actual phase = 2πq)
                     Includes conjugate pairs for Hermitian property

    Returns:
        A_gcn: Symmetric normalized adjacency (real)
        A_mag: Hermitian normalized adjacency (complex) with per-edge phases

    Example:
        # CSBM provides: edges [[0,1], [1,0]], phases [0.25, -0.25]
        # This creates A[0,1] = e^(i*π/2), A[1,0] = e^(-i*π/2) automatically
    """
    # Symmetric adjacency (magnitude only, undirected)
    A_sym = torch.zeros(num_nodes, num_nodes)
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        A_sym[u, v] = 1.0

    # Make symmetric (average if both directions exist)
    A_sym = (A_sym + A_sym.T) / 2.0

    # Add self-loops
    A_sym = A_sym + torch.eye(num_nodes)

    # Degree normalization
    D = A_sym.sum(dim=1)
    D_inv_sqrt = torch.zeros(num_nodes)
    D_inv_sqrt[D > 0] = 1.0 / torch.sqrt(D[D > 0])
    D_mat = torch.diag(D_inv_sqrt)

    # GCN: symmetric normalized
    A_gcn = D_mat @ A_sym @ D_mat

    # MagNet: Hermitian with per-edge phase encoding
    A_mag = torch.eye(num_nodes, dtype=torch.complex64)  # Self-loops
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        q = edge_phases[i]
        phase = 2 * np.pi * q
        # Set A[u, v] = e^(iφ) directly
        # The conjugate A[v, u] = e^(-iφ) is set when we encounter edge [v, u] in the loop
        A_mag[u, v] = np.exp(1j * phase)

    # Same normalization
    D_mat_c = D_mat.to(torch.complex64)
    A_mag = D_mat_c @ A_mag @ D_mat_c

    return A_gcn, A_mag


def unbalance_score(edge_index: torch.Tensor, num_nodes: int, q: float = 0.1) -> float:
    """
    Compute structural unbalance score.

    Score = 1 - rho(A_mag) / rho(A_sym)

    Higher score indicates more directional information is lost
    when symmetrizing the adjacency matrix.

    Args:
        edge_index: [2, E] tensor of directed edges
        num_nodes: Number of nodes
        q: Phase parameter

    Returns:
        Unbalance score in [0, 1]
    """
    A_gcn, A_mag = build_matrices(edge_index, num_nodes, q)

    rho_gcn = np.max(np.abs(np.linalg.eigvalsh(A_gcn.numpy())))
    rho_mag = np.max(np.abs(np.linalg.eigvals(A_mag.numpy())))

    if rho_gcn == 0:
        return 0.0
    return 1 - rho_mag / rho_gcn
