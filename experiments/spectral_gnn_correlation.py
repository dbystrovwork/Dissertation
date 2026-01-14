"""
Spectral-GNN Correlation Experiment

Tests the hypothesis that spectral properties of the magnetic Laplacian
predict when MagNet outperforms GCN. Extends hierarchical_direction.py
with spectral metric computation and correlation analysis.

Core Hypothesis: Spectral gap ratio correlates with MagNet-GCN performance gap.
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
from scipy.stats import pearsonr

from src.models import GCN, MagNet
from src.graphs import build_matrices, unbalance_score, generate_hierarchical_dag
from src.graphs.spectral import compute_laplacian_spectrum, compare_spectra
from src.utils import train_eval


def run_experiment(num_trials: int = 10, q: float = 0.5) -> pd.DataFrame:
    """Run experiment comparing GCN vs MagNet on hierarchical DAGs with spectral analysis."""
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

            # Build matrices
            A_gcn, A_mag = build_matrices(edge_index, n, q)

            # Compute spectral properties
            spectrum = compute_laplacian_spectrum(edge_index, n, q)
            comparison = compare_spectra(
                spectrum['eigenvalues_standard'],
                spectrum['eigenvalues_magnetic']
            )

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
                "delta": delta,
                # Spectral metrics
                "spectral_radius_ratio": comparison['spectral_radius_ratio'],
                "spectral_gap_ratio": comparison['spectral_gap_ratio'],
                "eigenvalue_correlation": comparison['eigenvalue_correlation'],
            })

            print(f"  {trial}: unbal={score:.3f}, gap_ratio={comparison['spectral_gap_ratio']:.3f}, "
                  f"GCN={acc_gcn:.3f}, MagNet={acc_mag:.3f}, d={delta:+.3f}")

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize results with spectral correlation."""
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    summary = df.groupby("config")[["num_levels", "unbalance", "gcn", "magnet", "delta",
                                     "spectral_gap_ratio", "eigenvalue_correlation"]].mean()
    print(summary.round(3))

    # Overall stats
    print(f"\nOverall mean delta: {df['delta'].mean():.3f}")
    print(f"MagNet wins: {(df['delta'] > 0).sum()}/{len(df)} trials")

    # T-test: is delta significantly > 0?
    t, p = stats.ttest_1samp(df["delta"], 0, alternative='greater')
    print(f"T-test (delta > 0): t={t:.3f}, p={p:.4f}")

    # Spectral-Performance Correlations
    print("\n" + "=" * 60)
    print("SPECTRAL-PERFORMANCE CORRELATIONS")
    print("=" * 60)

    r_gap, p_gap = pearsonr(df['spectral_gap_ratio'], df['delta'])
    r_radius, p_radius = pearsonr(df['spectral_radius_ratio'], df['delta'])
    r_corr, p_corr = pearsonr(df['eigenvalue_correlation'], df['delta'])

    print(f"Gap Ratio vs Delta: r={r_gap:.3f}, p={p_gap:.4f}")
    print(f"Radius Ratio vs Delta: r={r_radius:.3f}, p={p_radius:.4f}")
    print(f"Eigenvalue Corr vs Delta: r={r_corr:.3f}, p={p_corr:.4f}")

    # Interpretation
    print("\nInterpretation:")
    if abs(r_gap) > 0.4 and p_gap < 0.05:
        print(f"  [OK] STRONG correlation between spectral gap ratio and performance ({r_gap:.3f})")
        print("  Hypothesis H1 VALIDATED: Spectral properties predict GNN performance")
    elif abs(r_gap) > 0.2:
        print(f"  MODERATE correlation ({r_gap:.3f}) - some predictive power")
    else:
        print(f"  WEAK correlation ({r_gap:.3f}) - spectral gap may not be the key metric")

    # Plot - 3 panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(df["config"].unique())))

    # Panel 1: Unbalance vs delta (original)
    for i, cfg in enumerate(df["config"].unique()):
        sub = df[df["config"] == cfg]
        ax1.scatter(sub["unbalance"], sub["delta"], label=cfg,
                    c=[colors[i]], alpha=0.7, s=80)
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Unbalance Score")
    ax1.set_ylabel("MagNet - GCN Accuracy")
    ax1.set_title("Direction Encodes Class Info")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Bar comparison (original)
    x_pos = np.arange(len(summary))
    ax2.bar(x_pos - 0.175, summary["gcn"], 0.35, label="GCN", color="steelblue")
    ax2.bar(x_pos + 0.175, summary["magnet"], 0.35, label="MagNet", color="coral")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(summary.index, rotation=15)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Comparison by Hierarchy Depth")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Spectral gap ratio vs delta (NEW!)
    for i, cfg in enumerate(df["config"].unique()):
        sub = df[df["config"] == cfg]
        ax3.scatter(sub["spectral_gap_ratio"], sub["delta"], label=cfg,
                    c=[colors[i]], alpha=0.7, s=80)
    ax3.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax3.set_xlabel("Spectral Gap Ratio (magnetic/standard)")
    ax3.set_ylabel("MagNet - GCN Accuracy")
    ax3.set_title(f"Spectral Prediction (r={r_gap:.3f}, p={p_gap:.4f})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add trend line if correlation is significant
    if abs(r_gap) > 0.2 and p_gap < 0.05:
        x_trend = np.linspace(df['spectral_gap_ratio'].min(), df['spectral_gap_ratio'].max(), 100)
        z = np.polyfit(df['spectral_gap_ratio'], df['delta'], 1)
        p = np.poly1d(z)
        ax3.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label=f"Trend (r={r_gap:.3f})")

    plt.tight_layout()
    output_path = output_dir / "spectral_gnn_correlation_results.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nFigure saved to: {output_path}")

    # Conclusion
    print("\n" + "=" * 60)
    mean_delta = df["delta"].mean()
    if mean_delta > 0.02 and p < 0.05:
        print(f"SUCCESS: MagNet outperforms GCN (mean delta={mean_delta:.3f}, p={p:.4f})")
        print("Direction encodes class-relevant information")

        if abs(r_gap) > 0.4 and p_gap < 0.05:
            print(f"\nBONUS: Strong spectral-performance correlation (r={r_gap:.3f}, p={p_gap:.4f})")
            print("Spectral properties PREDICT when MagNet wins!")
    elif mean_delta < -0.02:
        print("UNEXPECTED: GCN outperforms MagNet")
    else:
        print("INCONCLUSIVE: No significant difference")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Spectral-GNN Correlation Experiment")
    print("=" * 60)
    print("\nTesting hypothesis: Spectral gap ratio predicts MagNet advantage")

    # Run experiment
    df = run_experiment(num_trials=10, q=0.5)

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / "spectral_gnn_correlation_results.csv", index=False)

    # Analyze
    analyze_results(df, output_dir)


if __name__ == "__main__":
    main()
