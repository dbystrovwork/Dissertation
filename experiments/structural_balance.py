"""
Structural Balance Experiment

Tests whether MagNet outperforms GCN on structurally unbalanced graphs.
Hypothesis: MagNet advantage correlates with unbalance score.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.models import GCN, MagNet
from src.graphs import build_matrices, unbalance_score, generate_sbm
from src.utils import train_eval


def run_experiment(num_trials: int = 10, q: float = 0.1) -> pd.DataFrame:
    """Run full experiment comparing GCN vs MagNet."""
    results = []
    n, k = 100, 4

    for name, balanced in [("Balanced", True), ("Unbalanced", False)]:
        print(f"\n{name}:")
        for trial in range(num_trials):
            # Generate graph
            edge_index, labels = generate_sbm(n, k, 0.3, 0.05, balanced)
            A_gcn, A_mag = build_matrices(edge_index, n, q)

            # Features and masks
            x = torch.randn(n, 16)
            perm = torch.randperm(n)
            train_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[perm[:60]] = True
            test_mask = ~train_mask

            # Compute unbalance score
            score = unbalance_score(edge_index, n, q)

            # Train GCN
            gcn = GCN(16, 32, k)
            acc_gcn = train_eval(gcn, x, A_gcn, labels, train_mask, test_mask)

            # Train MagNet
            magnet = MagNet(16, 32, k)
            acc_mag = train_eval(magnet, x, A_mag, labels, train_mask, test_mask)

            delta = acc_mag - acc_gcn

            results.append({
                "config": name,
                "trial": trial,
                "unbalance": score,
                "gcn": acc_gcn,
                "magnet": acc_mag,
                "delta": delta
            })

            print(f"  {trial}: unbal={score:.3f}, GCN={acc_gcn:.3f}, MagNet={acc_mag:.3f}, d={delta:+.3f}")

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize results."""
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    summary = df.groupby("config")[["unbalance", "gcn", "magnet", "delta"]].mean()
    print(summary.round(3))

    # Statistics
    corr, p_corr = stats.pearsonr(df["unbalance"], df["delta"])
    print(f"\nCorrelation (unbalance vs delta): r={corr:.3f}, p={p_corr:.4f}")

    bal = df[df["config"] == "Balanced"]["delta"]
    unbal = df[df["config"] == "Unbalanced"]["delta"]
    t, p = stats.ttest_ind(unbal, bal)
    print(f"T-test: t={t:.3f}, p={p:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"Balanced": "blue", "Unbalanced": "red"}
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg]
        ax1.scatter(sub["unbalance"], sub["delta"], label=cfg,
                    c=colors[cfg], alpha=0.7, s=80)
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Unbalance Score")
    ax1.set_ylabel("MagNet - GCN Accuracy")
    ax1.set_title("Structural Unbalance vs Performance Gap")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    x_pos = np.arange(2)
    ax2.bar(x_pos - 0.175, summary["gcn"], 0.35, label="GCN", color="steelblue")
    ax2.bar(x_pos + 0.175, summary["magnet"], 0.35, label="MagNet", color="coral")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(summary.index)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "structural_balance_results.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nFigure saved to: {output_path}")

    # Conclusion
    print("\n" + "=" * 60)
    if corr > 0.2 and unbal.mean() > bal.mean():
        print("SUPPORTS hypothesis: MagNet advantage on unbalanced graphs")
    elif corr < -0.2:
        print("CONTRADICTS hypothesis")
    else:
        print("INCONCLUSIVE - need more trials or different setup")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Structural Balance & GNN Experiment")
    print("=" * 60)

    # Run experiment
    df = run_experiment(num_trials=10, q=0.1)

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / "structural_balance_results.csv", index=False)

    # Analyze
    analyze_results(df, output_dir)


if __name__ == "__main__":
    main()
