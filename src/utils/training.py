"""Training utilities."""

import torch
import torch.nn.functional as F


def train_eval(
    model: torch.nn.Module,
    x: torch.Tensor,
    A: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4
) -> float:
    """
    Train model and return test accuracy.

    Args:
        model: GNN model
        x: Node features [N, F]
        A: Adjacency matrix [N, N]
        labels: Node labels [N]
        train_mask: Boolean mask for training nodes
        test_mask: Boolean mask for test nodes
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization

    Returns:
        Test accuracy as float
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(x, A)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x, A).argmax(dim=1)
        acc = (pred[test_mask] == labels[test_mask]).float().mean()

    return acc.item()
