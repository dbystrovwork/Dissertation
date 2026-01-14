"""
CSBM Sub-Community Classification Experiment

Tests whether MagNet can leverage phase structure to distinguish sub-communities.

Key Difference from Previous Experiment:
- Previous: Level-1 community classification (too easy for GCN)
- This: Level-2 sub-community classification (defined by phase structure)

Hypothesis: MagNet performs better on balanced graphs where phases are structured,
worse on unbalanced graphs where phases are random.

Expected Results:
- Balanced (eta=0): MagNet > GCN (phases encode sub-community structure)
- Mixed (eta>0): MagNet advantage decreases as phases randomize
- Unbalanced: MagNet ≈ GCN (random phases provide no signal)
"""

from __future__ import annotations

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
    Run CSBM sub-community classification experiment.

    Task: Classify nodes into their sub-community (level-2 labels)
    - Balanced: Sub-communities have structured phase offsets
    - Unbalanced: Phases are completely random

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
    print("CSBM Sub-Community Classification Experiment")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Trials per config: {num_trials}")
    print(f"  Nodes per graph: 120 (2 communities × 60 nodes)")
    print(f"  Task: Sub-community classification (6 classes: 2 communities × 3 sub-communities)")
    print(f"  Why this task: Sub-communities are defined by phase structure\n")

    for name, gen_func, kwargs in configs:
        print(f"\n{name}:")

        for trial in range(num_trials):
            # Generate graph
            edge_index, level1_labels, metadata = gen_func(**kwargs, seed=trial)

            # KEY CHANGE: Use sub-community labels as the task
            sub_labels = metadata['sub_community_labels']
            n = len(sub_labels)
            num_classes = len(torch.unique(sub_labels))

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
            acc_gcn = train_eval(gcn, x, A_gcn, sub_labels, train_mask, test_mask, epochs=200)

            # Train MagNet
            magnet = MagNet(16, 32, num_classes)
            acc_mag = train_eval(magnet, x, A_mag, sub_labels, train_mask, test_mask, epochs=200)

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
            print(f"  Trial {trial}: phase_div={phase_diversity:.3f}, "
                  f"GCN={acc_gcn:.3f}, MagNet={acc_mag:.3f}, delta={delta:+.3f} [{status}]")

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame, output_dir: Path):
    """
    Analyze and visualize results.

    Tests:
    1. Does MagNet outperform GCN on balanced graphs?
    2. Does performance degrade with increasing phase randomness?
    3. Does phase diversity (negatively) correlate with delta?
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
    print(f"  GCN accuracy: {df[df['balance_type'] == 'Balanced']['gcn'].mean():.4f}")
    print(f"  MagNet accuracy: {df[df['balance_type'] == 'Balanced']['magnet'].mean():.4f}")

    print(f"\nUnbalanced (n={len(unbalanced)}):")
    print(f"  Mean delta: {unbalanced.mean():.4f} ± {unbalanced.std():.4f}")
    print(f"  GCN accuracy: {df[df['balance_type'] == 'Unbalanced']['gcn'].mean():.4f}")
    print(f"  MagNet accuracy: {df[df['balance_type'] == 'Unbalanced']['magnet'].mean():.4f}")

    # Test 1: Is MagNet better than GCN on balanced graphs?
    t_balanced, p_balanced = stats.ttest_1samp(balanced, 0, alternative='greater')
    print(f"\nTest 1: MagNet > GCN on balanced? t={t_balanced:.3f}, p={p_balanced:.4f}")
    if p_balanced < 0.05 and balanced.mean() > 0:
        print("  [YES] MagNet significantly outperforms GCN on balanced graphs")
    else:
        print("  [NO] No significant MagNet advantage on balanced graphs")

    # Test 2: Is delta higher for balanced than unbalanced?
    t, p = stats.ttest_ind(balanced, unbalanced, alternative='greater')
    print(f"\nTest 2: Balanced > Unbalanced? t={t:.3f}, p={p:.4f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((balanced.std()**2 + unbalanced.std()**2) / 2)
    if pooled_std > 0:
        cohens_d = (balanced.mean() - unbalanced.mean()) / pooled_std
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
    else:
        cohens_d = 0
        print("  Effect size: Unable to compute (std=0)")

    # Test 3: Correlation: phase diversity vs delta (should be negative)
    r, p_corr = pearsonr(df["phase_diversity"], df["delta"])
    print(f"\nTest 3: Correlation (phase diversity vs delta): r={r:.3f}, p={p_corr:.4f}")
    if p_corr < 0.05 and r < 0:
        print("  [YES] More phase randomness hurts MagNet (negative correlation)")
    else:
        print(f"  [NO] Phase diversity doesn't correlate as expected")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if p_balanced < 0.05 and balanced.mean() > 0:
        print(f"[HYPOTHESIS SUPPORTED]:")
        print(f"  - MagNet leverages structured phases on balanced graphs")
        print(f"  - Mean advantage: {balanced.mean():+.3f} (p={p_balanced:.4f})")

        if p < 0.05 and balanced.mean() > unbalanced.mean():
            print(f"  - Advantage disappears on unbalanced graphs (random phases)")
            print(f"  - Difference: {balanced.mean() - unbalanced.mean():+.3f} (p={p:.4f})")
            if abs(cohens_d) > 0.5:
                print(f"  - Effect size is MEDIUM-LARGE (d={cohens_d:.3f})")
            elif abs(cohens_d) > 0.2:
                print(f"  - Effect size is SMALL-MEDIUM (d={cohens_d:.3f})")
        else:
            print(f"  - But no significant difference from unbalanced (p={p:.4f})")
    else:
        print(f"[HYPOTHESIS NOT SUPPORTED]:")
        print(f"  - MagNet does not show advantage even on balanced graphs")
        print(f"  - Possible reasons:")
        print(f"    1. Sub-community structure still too weak")
        print(f"    2. Random features don't capture phase information")
        print(f"    3. Need different graph parameters (more nodes, higher p_in)")

    # Create visualization
    create_visualization(df, summary, balance_summary, t, p, r, p_corr, output_dir)


def create_visualization(df: pd.DataFrame, summary: pd.DataFrame,
                         balance_summary: pd.DataFrame, t: float, p: float,
                         r: float, p_corr: float, output_dir: Path):
    """Create 4-panel visualization."""
    fig = plt.figure(figsize=(20, 10))

    # Panel 1: Box plot by config
    ax1 = fig.add_subplot(2, 2, 1)
    configs = df["config"].unique()
    data_by_config = [df[df["config"] == cfg]["delta"].values for cfg in configs]
    bp = ax1.boxplot(data_by_config, labels=configs, patch_artist=True)

    # Color by type
    colors = {
        "Balanced-eta0.0": "forestgreen",
        "Mixed-eta0.1": "orange",
        "Mixed-eta0.2": "coral",
        "Unbalanced": "darkred"
    }
    for patch, cfg in zip(bp['boxes'], configs):
        patch.set_facecolor(colors.get(cfg, 'gray'))
        patch.set_alpha(0.7)

    ax1.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=2)
    ax1.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax1.set_title(f"Sub-Community Classification: Performance Gap\n(Balanced vs Unbalanced: t={t:.2f}, p={p:.4f})",
                  fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Phase diversity vs delta
    ax2 = fig.add_subplot(2, 2, 2)
    for cfg in configs:
        sub = df[df["config"] == cfg]
        ax2.scatter(sub["phase_diversity"], sub["delta"],
                   label=cfg, color=colors.get(cfg, 'gray'), alpha=0.6, s=80)

    # Add trend line
    z = np.polyfit(df["phase_diversity"], df["delta"], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(df["phase_diversity"].min(), df["phase_diversity"].max(), 100)
    ax2.plot(x_line, p_fit(x_line), "k--", alpha=0.5, linewidth=2, label=f"Trend (r={r:.3f})")

    ax2.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=2)
    ax2.set_xlabel("Phase Diversity (std)", fontsize=12)
    ax2.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax2.set_title(f"Phase Structure vs Performance\n(r={r:.3f}, p={p_corr:.4f})",
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Absolute performance comparison
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
    ax3.set_title("Absolute Performance by Balance Type", fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1])

    # Panel 4: Delta distribution by balance type
    ax4 = fig.add_subplot(2, 2, 4)
    balance_data = [df[df["balance_type"] == bt]["delta"].values
                    for bt in balance_types if bt in df["balance_type"].values]
    labels_present = [bt for bt in balance_types if bt in df["balance_type"].values]
    bp2 = ax4.boxplot(balance_data, labels=labels_present, patch_artist=True)

    type_colors = {"Balanced": "forestgreen", "Mixed": "orange", "Unbalanced": "darkred"}
    for patch, bt in zip(bp2['boxes'], labels_present):
        patch.set_facecolor(type_colors.get(bt, 'gray'))
        patch.set_alpha(0.7)

    ax4.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=2)
    ax4.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax4.set_title("Performance Gap Distribution", fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "csbm_subcommunity_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CSBM Sub-Community Classification Experiment")
    parser.add_argument('--num-trials', type=int, default=20, help='Number of trials per config')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run experiment
    df = run_experiment(num_trials=args.num_trials, output_dir=output_dir)

    # Save results
    csv_path = output_dir / "csbm_subcommunity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Analyze
    analyze_results(df, output_dir)


if __name__ == "__main__":
    main()