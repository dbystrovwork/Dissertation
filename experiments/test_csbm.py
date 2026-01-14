"""
Quick test of CSBM generators to verify correct behavior.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.graphs import generate_csbm, generate_csbm_unbalanced, build_matrices_with_phases


def test_balanced_csbm():
    """Test balanced phase-structured CSBM (eta=0)."""
    print("=" * 80)
    print("Test 1: Balanced Phase-Structured CSBM (eta=0)")
    print("=" * 80)

    edge_index, labels, meta = generate_csbm(
        n_communities=2,
        community_size=30,
        p_in=0.6,
        p_out=0.1,
        eta=0.0,
        sub_communities_per_comm=[3, 3],
        seed=42
    )

    print(f"Nodes: {len(labels)}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Level-1 communities: {meta['n_communities']}")
    print(f"Sub-communities: {meta['sub_communities']}")
    print(f"Eta (mixing): {meta['eta']}")

    # Check phase structure
    phases = meta['edge_phases']
    unique_phases = np.unique(np.round(phases, 3))
    print(f"\nUnique phases (rounded): {unique_phases}")
    print(f"Expected for 3 sub-communities: [0, 0.333, 0.667, -0.333, -0.667]")

    # Build matrices
    n = len(labels)
    A_gcn, A_mag = build_matrices_with_phases(edge_index, n, phases)
    print(f"\nGCN matrix shape: {A_gcn.shape}, dtype: {A_gcn.dtype}")
    print(f"MagNet matrix shape: {A_mag.shape}, dtype: {A_mag.dtype}")

    # Check Hermitian property
    is_hermitian = np.allclose(A_mag.numpy(), A_mag.numpy().conj().T, atol=1e-6)
    print(f"Hermitian check: {'PASS' if is_hermitian else 'FAIL'}")

    return edge_index, labels, meta


def test_phase_mixed_csbm():
    """Test phase-mixed CSBM (eta>0)."""
    print("\n" + "=" * 80)
    print("Test 2: Phase-Mixed CSBM (eta=0.3)")
    print("=" * 80)

    edge_index, labels, meta = generate_csbm(
        n_communities=2,
        community_size=30,
        p_in=0.6,
        p_out=0.1,
        eta=0.3,  # 30% of phases randomized
        sub_communities_per_comm=[3, 3],
        seed=42
    )

    print(f"Nodes: {len(labels)}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Eta (mixing): {meta['eta']}")

    # Check phase distribution
    phases = meta['edge_phases']
    unique_phases = np.unique(np.round(phases, 3))
    print(f"\nUnique phases (more varied due to mixing): {len(unique_phases)} distinct values")
    print(f"Sample phases: {unique_phases[:10]}")

    # Build matrices
    n = len(labels)
    A_gcn, A_mag = build_matrices_with_phases(edge_index, n, phases)

    # Check Hermitian
    is_hermitian = np.allclose(A_mag.numpy(), A_mag.numpy().conj().T, atol=1e-6)
    print(f"Hermitian check: {'PASS' if is_hermitian else 'FAIL'}")

    return edge_index, labels, meta


def test_unbalanced_csbm():
    """Test strictly unbalanced CSBM."""
    print("\n" + "=" * 80)
    print("Test 3: Strictly Unbalanced CSBM")
    print("=" * 80)

    edge_index, labels, meta = generate_csbm_unbalanced(
        n_communities=2,
        community_size=30,
        p_in=0.6,
        p_out=0.1,
        seed=42
    )

    print(f"Nodes: {len(labels)}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Eta: {meta['eta']} (not applicable for unbalanced)")

    # Check phase distribution (should be random)
    phases = meta['edge_phases']
    print(f"\nPhase distribution (random):")
    print(f"  Min: {phases.min():.3f}")
    print(f"  Max: {phases.max():.3f}")
    print(f"  Mean: {phases.mean():.3f}")
    print(f"  Std: {phases.std():.3f}")
    print(f"  Unique values: {len(np.unique(np.round(phases, 3)))}")

    # Build matrices
    n = len(labels)
    A_gcn, A_mag = build_matrices_with_phases(edge_index, n, phases)

    # Check Hermitian
    is_hermitian = np.allclose(A_mag.numpy(), A_mag.numpy().conj().T, atol=1e-6)
    print(f"Hermitian check: {'PASS' if is_hermitian else 'FAIL'}")

    return edge_index, labels, meta


def main():
    """Run all tests."""
    print("\nCSBM Generator Tests\n")

    test_balanced_csbm()
    test_phase_mixed_csbm()
    test_unbalanced_csbm()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()