"""Spectral analysis for magnetic Laplacian."""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .matrices import build_matrices


def compute_laplacian_spectrum(
    edge_index: torch.Tensor,
    num_nodes: int,
    q: float = 0.1,
    laplacian_type: str = 'normalized'
) -> Dict:
    """
    Compute eigenvalues and eigenvectors for both standard and magnetic Laplacian.

    Args:
        edge_index: [2, E] tensor of directed edges
        num_nodes: Number of nodes
        q: Phase parameter (default 0.1)
        laplacian_type: 'normalized', 'unnormalized', or 'random_walk'

    Returns:
        Dictionary containing:
            - eigenvalues_standard: Real eigenvalues of symmetric Laplacian
            - eigenvectors_standard: Real eigenvectors (columns)
            - eigenvalues_magnetic: Real eigenvalues of Hermitian magnetic Laplacian
            - eigenvectors_magnetic: Complex eigenvectors (columns)
            - spectral_gap_standard: λ_1 - λ_0
            - spectral_gap_magnetic: λ_1 - λ_0 for magnetic
    """
    # Build adjacency matrices using existing function
    A_gcn, A_mag = build_matrices(edge_index, num_nodes, q)

    # Construct Laplacians
    if laplacian_type == 'normalized':
        # L = I - A (already normalized in build_matrices)
        L_sym = torch.eye(num_nodes) - A_gcn
        L_mag = torch.eye(num_nodes, dtype=torch.complex64) - A_mag
    elif laplacian_type == 'unnormalized':
        # Need to rebuild without normalization for unnormalized case
        # For now, use normalized (can extend later if needed)
        raise NotImplementedError("Unnormalized Laplacian not yet implemented")
    elif laplacian_type == 'random_walk':
        raise NotImplementedError("Random walk Laplacian not yet implemented")
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    # Compute eigenvalues and eigenvectors
    # Use eigh for Hermitian matrices (guaranteed real eigenvalues, more stable)
    eigenvals_sym, eigenvecs_sym = np.linalg.eigh(L_sym.numpy())
    eigenvals_mag, eigenvecs_mag = np.linalg.eigh(L_mag.numpy())

    # Verify eigenvalues are real (should be for Hermitian matrices)
    if np.max(np.abs(eigenvals_mag.imag)) > 1e-8:
        print(f"Warning: Magnetic eigenvalues have imaginary part > 1e-8: {np.max(np.abs(eigenvals_mag.imag))}")

    # Extract real parts (should be essentially real already)
    eigenvals_mag = eigenvals_mag.real

    # Sort in ascending order (Laplacian convention: smallest first)
    idx_sym = np.argsort(eigenvals_sym)
    eigenvals_sym = eigenvals_sym[idx_sym]
    eigenvecs_sym = eigenvecs_sym[:, idx_sym]

    idx_mag = np.argsort(eigenvals_mag)
    eigenvals_mag = eigenvals_mag[idx_mag]
    eigenvecs_mag = eigenvecs_mag[:, idx_mag]

    # Compute spectral gaps (difference between first two eigenvalues)
    spectral_gap_sym = eigenvals_sym[1] - eigenvals_sym[0] if num_nodes > 1 else 0.0
    spectral_gap_mag = eigenvals_mag[1] - eigenvals_mag[0] if num_nodes > 1 else 0.0

    return {
        'eigenvalues_standard': eigenvals_sym,
        'eigenvectors_standard': eigenvecs_sym,
        'eigenvalues_magnetic': eigenvals_mag,
        'eigenvectors_magnetic': eigenvecs_mag,
        'spectral_gap_standard': spectral_gap_sym,
        'spectral_gap_magnetic': spectral_gap_mag,
    }


def extract_phase_patterns(eigenvector_complex: np.ndarray) -> Dict:
    """
    Extract magnitude and phase components from complex eigenvector.

    Args:
        eigenvector_complex: Complex eigenvector (1D array of length num_nodes)

    Returns:
        Dictionary containing:
            - magnitudes: |v_i| for each node
            - phases: arg(v_i) in [-π, π] for each node
            - phase_mean: Mean phase
            - phase_std: Standard deviation of phases
    """
    magnitudes = np.abs(eigenvector_complex)
    phases = np.angle(eigenvector_complex)  # Returns values in [-π, π]

    # Compute phase statistics
    phase_mean = np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))

    # Phase standard deviation (circular statistics)
    # R = mean resultant length
    R = np.sqrt(np.mean(np.cos(phases))**2 + np.mean(np.sin(phases))**2)
    phase_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.pi

    return {
        'magnitudes': magnitudes,
        'phases': phases,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
    }


def compare_spectra(eigenvals_sym: np.ndarray, eigenvals_mag: np.ndarray) -> Dict:
    """
    Compare eigenvalue distributions between symmetric and magnetic Laplacian.

    Args:
        eigenvals_sym: Eigenvalues from standard Laplacian
        eigenvals_mag: Eigenvalues from magnetic Laplacian

    Returns:
        Dictionary of comparison metrics:
            - spectral_radius_ratio: max|λ_mag| / max|λ_sym|
            - spectral_gap_ratio: (λ1_mag - λ0_mag) / (λ1_sym - λ0_sym)
            - eigenvalue_correlation: Pearson correlation between ordered eigenvalues
            - mean_eigenvalue_diff: Mean absolute difference
            - max_eigenvalue_diff: Maximum absolute difference
    """
    # Spectral radius ratio (relates to unbalance_score)
    rho_sym = np.max(np.abs(eigenvals_sym))
    rho_mag = np.max(np.abs(eigenvals_mag))
    spectral_radius_ratio = rho_mag / rho_sym if rho_sym > 0 else 0.0

    # Spectral gap ratio
    gap_sym = eigenvals_sym[1] - eigenvals_sym[0] if len(eigenvals_sym) > 1 else 0.0
    gap_mag = eigenvals_mag[1] - eigenvals_mag[0] if len(eigenvals_mag) > 1 else 0.0
    spectral_gap_ratio = gap_mag / gap_sym if gap_sym > 0 else 0.0

    # Eigenvalue correlation (Pearson)
    if len(eigenvals_sym) > 1:
        correlation = np.corrcoef(eigenvals_sym, eigenvals_mag)[0, 1]
    else:
        correlation = 1.0

    # Absolute differences
    mean_diff = np.mean(np.abs(eigenvals_mag - eigenvals_sym))
    max_diff = np.max(np.abs(eigenvals_mag - eigenvals_sym))

    return {
        'spectral_radius_ratio': spectral_radius_ratio,
        'spectral_gap_ratio': spectral_gap_ratio,
        'eigenvalue_correlation': correlation,
        'mean_eigenvalue_diff': mean_diff,
        'max_eigenvalue_diff': max_diff,
        'spectral_radius_standard': rho_sym,
        'spectral_radius_magnetic': rho_mag,
        'spectral_gap_standard': gap_sym,
        'spectral_gap_magnetic': gap_mag,
    }


def parameter_sweep_q(
    edge_index: torch.Tensor,
    num_nodes: int,
    q_values: np.ndarray
) -> pd.DataFrame:
    """
    Sweep through q parameter values and track spectral properties.

    Args:
        edge_index: [2, E] tensor of directed edges
        num_nodes: Number of nodes
        q_values: Array of q values to test (e.g., [0.01, 0.05, 0.1, ..., 0.5])

    Returns:
        DataFrame with columns: q, spectral_radius_mag, spectral_gap_mag,
                               spectral_radius_std, spectral_gap_std, etc.
    """
    results = []

    for q in q_values:
        # Compute spectrum for this q value
        spectrum = compute_laplacian_spectrum(edge_index, num_nodes, q)

        # Compare spectra
        comparison = compare_spectra(
            spectrum['eigenvalues_standard'],
            spectrum['eigenvalues_magnetic']
        )

        # Store results
        results.append({
            'q': q,
            'spectral_radius_standard': comparison['spectral_radius_standard'],
            'spectral_radius_magnetic': comparison['spectral_radius_magnetic'],
            'spectral_radius_ratio': comparison['spectral_radius_ratio'],
            'spectral_gap_standard': comparison['spectral_gap_standard'],
            'spectral_gap_magnetic': comparison['spectral_gap_magnetic'],
            'spectral_gap_ratio': comparison['spectral_gap_ratio'],
            'eigenvalue_correlation': comparison['eigenvalue_correlation'],
            'mean_eigenvalue_diff': comparison['mean_eigenvalue_diff'],
            'max_eigenvalue_diff': comparison['max_eigenvalue_diff'],
        })

    return pd.DataFrame(results)


def compute_eigenvector_localization(eigenvector: np.ndarray) -> float:
    """
    Compute Inverse Participation Ratio (IPR) for eigenvector localization.

    IPR = Σ|v_i|^4 / (Σ|v_i|^2)^2

    Values close to 1/N indicate delocalized (spread evenly across nodes).
    Values close to 1 indicate localized (concentrated on few nodes).

    Args:
        eigenvector: Eigenvector (real or complex)

    Returns:
        IPR value in [1/N, 1]
    """
    magnitudes = np.abs(eigenvector)
    numerator = np.sum(magnitudes**4)
    denominator = np.sum(magnitudes**2)**2

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_phase_coherence(eigenvector_complex: np.ndarray) -> float:
    """
    Measure phase coherence of complex eigenvector.

    Coherence = |<exp(iφ)>| where φ are the phases.
    Values close to 1 indicate aligned phases.
    Values close to 0 indicate random phases.

    Args:
        eigenvector_complex: Complex eigenvector

    Returns:
        Coherence value in [0, 1]
    """
    phases = np.angle(eigenvector_complex)
    mean_exp = np.mean(np.exp(1j * phases))
    return np.abs(mean_exp)
