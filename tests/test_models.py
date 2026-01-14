"""Tests for GNN models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import GCN, MagNet
from src.graphs import build_matrices, generate_sbm


def test_gcn_forward():
    """Test GCN forward pass."""
    model = GCN(in_dim=8, hidden_dim=16, out_dim=4)
    x = torch.randn(10, 8)
    A = torch.eye(10) + torch.randn(10, 10) * 0.1
    A = A / A.sum(dim=1, keepdim=True)  # Normalize

    out = model(x, A)
    assert out.shape == (10, 4)
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(10), atol=1e-5)
    print("GCN forward: PASS")


def test_magnet_forward():
    """Test MagNet forward pass."""
    model = MagNet(in_dim=8, hidden_dim=16, out_dim=4)
    x = torch.randn(10, 8)
    A_mag = torch.eye(10, dtype=torch.complex64)

    out = model(x, A_mag)
    assert out.shape == (10, 4)
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(10), atol=1e-5)
    print("MagNet forward: PASS")


def test_build_matrices():
    """Test matrix construction."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    A_gcn, A_mag = build_matrices(edge_index, num_nodes=3, q=0.1)

    assert A_gcn.shape == (3, 3)
    assert A_mag.shape == (3, 3)
    assert A_mag.dtype == torch.complex64
    print("build_matrices: PASS")


def test_generate_sbm():
    """Test SBM generation."""
    edge_index, labels = generate_sbm(n=20, k=4, p_in=0.5, p_out=0.1)

    assert edge_index.shape[0] == 2
    assert labels.shape == (20,)
    assert labels.max() < 4
    print("generate_sbm: PASS")


if __name__ == "__main__":
    test_gcn_forward()
    test_magnet_forward()
    test_build_matrices()
    test_generate_sbm()
    print("\nAll tests passed!")
