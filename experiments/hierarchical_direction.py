"""
Hierarchical Direction Experiment

Tests whether MagNet outperforms GCN when direction encodes meaningful information.
Uses hierarchical DAGs where edge direction = information flow through levels.
Node label = hierarchy level, so direction is class-relevant.
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
from src.graphs import build_matrices, unbalance_score, generate_hierarchical_dag
from src.utils import train_eval


def run_experiment(num_trials: int = 10, q: float = 0.5) -> pd.DataFrame:
    """Run experiment comparing GCN vs MagNet on hierarchical DAGs."""
    results = []
    n = 100

    configs = [
        ("3-Level", 3),
        ("4-Level", 4),
        ("5-Level", 5),
        ("6-Level", 6),
    ]

    for name, num_levels in configs:
        print(f"\n{name}:")
        for trial in range(num_trials):
            # Generate hierarchical DAG
            edge_index, labels = generate_hierarchical_dag(
                n=n,
                num_levels=num_levels,
                p_forward=0.3,
                p_skip=0.1,
                p_lateral=0.05
            )

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
            gcn = GCN(16, 32, num_levels)
            acc_gcn = train_eval(gcn, x, A_gcn, labels, train_mask, test_mask)

            # Train MagNet
            magnet = MagNet(16, 32, num_levels)
            acc_mag = train_eval(magnet, x, A_mag, labels, train_mask, test_mask)

            delta = acc_mag - acc_gcn

            results.append({
                "config": name,
                "num_levels": num_levels,
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
    summary = df.groupby("config")[["num_levels", "unbalance", "gcn", "magnet", "delta"]].mean()
    print(summary.round(3))

    # Overall stats
    print(f"\nOverall mean delta: {df['delta'].mean():.3f}")
    print(f"MagNet wins: {(df['delta'] > 0).sum()}/{len(df)} trials")

    # T-test: is delta significantly > 0?
    t, p = stats.ttest_1samp(df["delta"], 0, alternative='greater')
    print(f"T-test (delta > 0): t={t:.3f}, p={p:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: unbalance vs delta
    colors = plt.cm.viridis(np.linspace(0, 1, len(df["config"].unique())))
    for i, cfg in enumerate(df["config"].unique()):
        sub = df[df["config"] == cfg]
        ax1.scatter(sub["unbalance"], sub["delta"], label=cfg,
                    c=[colors[i]], alpha=0.7, s=80)
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Unbalance Score")
    ax1.set_ylabel("MagNet - GCN Accuracy")
    ax1.set_title("Hierarchical DAGs: Direction Encodes Class Info")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar: comparison by config
    x_pos = np.arange(len(summary))
    ax2.bar(x_pos - 0.175, summary["gcn"], 0.35, label="GCN", color="steelblue")
    ax2.bar(x_pos + 0.175, summary["magnet"], 0.35, label="MagNet", color="coral")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(summary.index, rotation=15)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Comparison by Hierarchy Depth")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "hierarchical_direction_results.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nFigure saved to: {output_path}")

    # Conclusion
    print("\n" + "=" * 60)
    mean_delta = df["delta"].mean()
    if mean_delta > 0.02 and p < 0.05:
        print(f"SUCCESS: MagNet outperforms GCN (mean delta={mean_delta:.3f}, p={p:.4f})")
        print("Direction encodes class-relevant information")
    elif mean_delta < -0.02:
        print("UNEXPECTED: GCN outperforms MagNet")
    else:
        print("INCONCLUSIVE: No significant difference")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Hierarchical Direction Experiment")
    print("=" * 60)

    # Run experiment
    df = run_experiment(num_trials=10, q=0.5)

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / "hierarchical_direction_results.csv", index=False)

    # Analyze
    analyze_results(df, output_dir)


if __name__ == "__main__":
    main()
