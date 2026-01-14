"""
CSBM Balanced vs Unbalanced: MagNet Performance Comparison

Tests three key scenarios:
1. Balanced Phase-Structured CSBM (eta=0): Clean phase offsets, structurally balanced
2. Phase-Mixed CSBM (eta>0): Partial randomization, intermediate balance
3. Strictly Unbalanced CSBM: Random phases, frustrated cycles

Hypothesis: MagNet shows greater advantage on strictly unbalanced graphs
than on balanced phase-structured graphs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

from src.models import GCN, MagNet
from src.graphs import generate_csbm, generate_csbm_unbalanced, build_matrices_with_phases
from src.utils import train_eval


def run_experiment(num_trials: int = 20, output_dir: Path = None) -> pd.DataFrame:
    """
    Run CSBM comparison experiment.

    Three configurations:
    - Balanced (eta=0): Deterministic phase structure
    - Mixed (eta=0.2): 20% phase randomization
    - Unbalanced: Completely random phases

    Args:
        num_trials: Number of trials per configuration
        output_dir: Directory to save results

    Returns:
        DataFrame with results
    """
    results = []

    # Configuration: (name, generator_func, kwargs)
    configs = [
        ("Balanced-eta0.0", generate_csbm, {
            "n_communities": 2,
            "community_size": 60,
            "p_in": 0.6,
            "p_out": 0.1,
            "eta": 0.0,
            "sub_communities_per_comm": [3, 3]
        }),
        ("Mixed-eta0.1", generate_csbm, {
            "n_communities": 2,
            "community_size": 60,
            "p_in": 0.6,
            "p_out": 0.1,
            "eta": 0.1,
            "sub_communities_per_comm": [3, 3]
        }),
        ("Mixed-eta0.2", generate_csbm, {
            "n_communities": 2,
            "community_size": 60,
            "p_in": 0.6,
            "p_out": 0.1,
            "eta": 0.2,
            "sub_communities_per_comm": [3, 3]
        }),
        ("Unbalanced", generate_csbm_unbalanced, {
            "n_communities": 2,
            "community_size": 60,
            "p_in": 0.6,
            "p_out": 0.1,
        }),
    ]

    print("=" * 80)
    print("CSBM Balanced vs Unbalanced Experiment")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Trials per config: {num_trials}")
    print(f"  Nodes per graph: 120 (2 communities × 60 nodes)")
    print(f"  Task: Level-1 community classification\n")

    for name, gen_func, kwargs in configs:
        print(f"\n{name}:")

        for trial in range(num_trials):
            # Generate graph
            edge_index, labels, metadata = gen_func(**kwargs, seed=trial)
            n = len(labels)
            num_classes = len(torch.unique(labels))

            # Build matrices with per-edge phases
            edge_phases = metadata['edge_phases']
            A_gcn, A_mag = build_matrices_with_phases(edge_index, n, edge_phases)

            # Random features (16-dimensional)
            x = torch.randn(n, 16)

            # Train/test split (60/40)
            perm = torch.randperm(n)
            train_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[perm[:int(0.6 * n)]] = True
            test_mask = ~train_mask

            # Compute phase diversity (std of phases)
            phase_diversity = np.std(edge_phases)

            # Train GCN
            gcn = GCN(16, 32, num_classes)
            acc_gcn = train_eval(gcn, x, A_gcn, labels, train_mask, test_mask, epochs=200)

            # Train MagNet
            magnet = MagNet(16, 32, num_classes)
            acc_mag = train_eval(magnet, x, A_mag, labels, train_mask, test_mask, epochs=200)

            delta = acc_mag - acc_gcn

            results.append({
                "config": name,
                "trial": trial,
                "num_nodes": n,
                "num_edges": edge_index.shape[1],
                "num_classes": num_classes,
                "eta": metadata['eta'] if 'eta' in metadata and metadata['eta'] is not None else -1.0,
                "phase_diversity": phase_diversity,
                "gcn": acc_gcn,
                "magnet": acc_mag,
                "delta": delta,
            })

            status = "+" if delta > 0 else "-"
            print(f"  {trial}: phase_div={phase_diversity:.3f}, "
                  f"GCN={acc_gcn:.3f}, MagNet={acc_mag:.3f}, delta={delta:+.3f} [{status}]")

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame, output_dir: Path):
    """
    Analyze and visualize results.

    Tests:
    1. Does MagNet outperform GCN overall?
    2. Is delta larger for unbalanced vs balanced?
    3. Does phase diversity correlate with performance?
    """
    print("\n" + "=" * 80)
    print("RESULTS BY CONFIGURATION")
    print("=" * 80)

    # Summary by config
    summary = df.groupby("config")[["phase_diversity", "gcn", "magnet", "delta"]].agg(["mean", "std"])
    print(summary.round(3))

    # Group by balance type
    df["balance_type"] = df["config"].apply(
        lambda x: "Balanced" if "eta0.0" in x else ("Mixed" if "Mixed" in x else "Unbalanced")
    )

    print("\n" + "=" * 80)
    print("RESULTS BY BALANCE TYPE")
    print("=" * 80)

    balance_summary = df.groupby("balance_type")[["phase_diversity", "gcn", "magnet", "delta"]].agg(["mean", "std"])
    print(balance_summary.round(3))

    # Statistical tests
    print(f"\n{'=' * 80}")
    print("STATISTICAL TESTS")
    print(f"{'=' * 80}")

    balanced = df[df["balance_type"] == "Balanced"]["delta"]
    unbalanced = df[df["balance_type"] == "Unbalanced"]["delta"]

    print(f"\nBalanced (n={len(balanced)}):")
    print(f"  Mean delta: {balanced.mean():.4f} ± {balanced.std():.4f}")

    print(f"\nUnbalanced (n={len(unbalanced)}):")
    print(f"  Mean delta: {unbalanced.mean():.4f} ± {unbalanced.std():.4f}")

    # T-test: unbalanced > balanced
    t, p = stats.ttest_ind(unbalanced, balanced, alternative='greater')
    print(f"\nT-test (unbalanced > balanced): t={t:.3f}, p={p:.4f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((balanced.std()**2 + unbalanced.std()**2) / 2)
    if pooled_std > 0:
        cohens_d = (unbalanced.mean() - balanced.mean()) / pooled_std
        print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    else:
        cohens_d = 0
        print("Effect size: Unable to compute (std=0)")

    # Correlation: phase diversity vs delta
    r, p_corr = pearsonr(df["phase_diversity"], df["delta"])
    print(f"\nCorrelation (phase diversity vs delta): r={r:.3f}, p={p_corr:.4f}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if p < 0.05 and unbalanced.mean() > balanced.mean():
        print(f"[VALIDATED] Hypothesis confirmed: MagNet shows greater advantage on unbalanced graphs")
        print(f"  Mean delta difference: {unbalanced.mean() - balanced.mean():+.3f} (p={p:.4f})")
        if abs(cohens_d) > 0.5:
            print(f"  Effect size is MEDIUM-LARGE (d={cohens_d:.3f})")
        elif abs(cohens_d) > 0.2:
            print(f"  Effect size is SMALL-MEDIUM (d={cohens_d:.3f})")
    elif p < 0.05 and unbalanced.mean() < balanced.mean():
        print(f"[UNEXPECTED] Balanced graphs show larger MagNet advantage")
        print(f"  This contradicts the hypothesis!")
    else:
        print(f"[INCONCLUSIVE] No significant difference (p={p:.4f})")
        print(f"  Possible reasons:")
        print(f"  - Task structure (community classification) may dominate")
        print(f"  - Need different task or larger graphs")

    # Create visualization
    create_visualization(df, summary, balance_summary, t, p, output_dir)


def create_visualization(df: pd.DataFrame, summary: pd.DataFrame,
                         balance_summary: pd.DataFrame, t: float, p: float,
                         output_dir: Path):
    """Create 4-panel visualization."""
    fig = plt.figure(figsize=(20, 10))

    # Panel 1: Box plot by config
    ax1 = fig.add_subplot(2, 2, 1)
    configs = df["config"].unique()
    data_by_config = [df[df["config"] == cfg]["delta"].values for cfg in configs]
    bp = ax1.boxplot(data_by_config, labels=configs, patch_artist=True)

    # Color by type
    colors = {
        "Balanced-eta0.0": "steelblue",
        "Mixed-eta0.1": "lightcoral",
        "Mixed-eta0.2": "coral",
        "Unbalanced": "darkred"
    }
    for patch, cfg in zip(bp['boxes'], configs):
        patch.set_facecolor(colors.get(cfg, 'gray'))

    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax1.set_title(f"Performance Gap by Config\n(t={t:.2f}, p={p:.4f})",
                  fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Phase diversity vs delta
    ax2 = fig.add_subplot(2, 2, 2)
    for cfg in configs:
        sub = df[df["config"] == cfg]
        ax2.scatter(sub["phase_diversity"], sub["delta"],
                   label=cfg, color=colors.get(cfg, 'gray'), alpha=0.6, s=60)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Phase Diversity (std)", fontsize=12)
    ax2.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax2.set_title("Phase Diversity vs Performance", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Bar chart by balance type
    ax3 = fig.add_subplot(2, 2, 3)
    balance_types = ["Balanced", "Mixed", "Unbalanced"]
    mean_gcn = [balance_summary.loc[bt, ("gcn", "mean")] if bt in balance_summary.index else 0
                for bt in balance_types]
    mean_magnet = [balance_summary.loc[bt, ("magnet", "mean")] if bt in balance_summary.index else 0
                   for bt in balance_types]

    x_pos = np.arange(len(balance_types))
    width = 0.35
    ax3.bar(x_pos - width/2, mean_gcn, width, label="GCN", color="steelblue", alpha=0.8)
    ax3.bar(x_pos + width/2, mean_magnet, width, label="MagNet", color="darkred", alpha=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(balance_types)
    ax3.set_ylabel("Accuracy", fontsize=12)
    ax3.set_title("Model Comparison by Balance Type", fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Delta distribution by balance type
    ax4 = fig.add_subplot(2, 2, 4)
    balance_data = [df[df["balance_type"] == bt]["delta"].values
                    for bt in balance_types if bt in df["balance_type"].values]
    labels_present = [bt for bt in balance_types if bt in df["balance_type"].values]
    bp2 = ax4.boxplot(balance_data, labels=labels_present, patch_artist=True)

    type_colors = {"Balanced": "steelblue", "Mixed": "coral", "Unbalanced": "darkred"}
    for patch, bt in zip(bp2['boxes'], labels_present):
        patch.set_facecolor(type_colors.get(bt, 'gray'))

    ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax4.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax4.set_title("Performance Gap by Balance Type", fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "csbm_balance_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CSBM Balanced vs Unbalanced Experiment")
    parser.add_argument('--num-trials', type=int, default=20, help='Number of trials per config')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run experiment
    df = run_experiment(num_trials=args.num_trials, output_dir=output_dir)

    # Save results
    df.to_csv(output_dir / "csbm_balance_results.csv", index=False)
    print(f"\nResults saved to: {output_dir / 'csbm_balance_results.csv'}")

    # Analyze
    analyze_results(df, output_dir)


if __name__ == "__main__":
    main()