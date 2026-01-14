"""
Balanced vs Strictly Unbalanced Graphs: MagNet Performance Comparison

Tests the hypothesis from arxiv:2307.01813 that MagNet performs better
on strictly unbalanced graphs than on structurally balanced graphs.

Key definitions:
- Structural Balance: Phase of every cycle = 0 (mod 2π)
- Strict Unbalance: Graph has cycles with non-zero phase accumulation

For a directed cycle with n nodes and uniform phase θ = 2πq per edge:
- Balanced: n·(2πq) = 0 (mod 2π) → choose n, q such that n·q is integer
- Unbalanced: n·(2πq) ≠ 0 (mod 2π) → choose n, q such that n·q is NOT integer
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

from src.models import GCN, MagNet
from src.graphs import (
    build_matrices,
    unbalance_score,
    generate_balanced_cycle,
    generate_unbalanced_cycle,
)
from src.utils import train_eval


def compute_cycle_phase_accumulation(n: int, q: float) -> float:
    """
    Compute total phase accumulation around an n-cycle with phase parameter q.

    Args:
        n: Cycle length
        q: Phase parameter (edge phase = 2πq)

    Returns:
        phase_accumulation: Total phase mod 2π
    """
    total_phase = n * 2 * np.pi * q
    return total_phase % (2 * np.pi)


def is_balanced(phase_accumulation: float, tolerance: float = 0.1) -> bool:
    """Check if phase accumulation is close to 0 or 2π (balanced)."""
    return abs(phase_accumulation) < tolerance or abs(phase_accumulation - 2 * np.pi) < tolerance


def run_experiment(num_trials: int = 20, output_dir: Path = None) -> pd.DataFrame:
    """
    Run balanced vs unbalanced comparison experiment.

    Strategy:
    - For balanced cycles: Choose (n, q) pairs where n·q is integer
    - For unbalanced cycles: Choose (n, q) pairs where n·q is NOT integer
    - Same task structure (position mod k_classes), different balance

    Args:
        num_trials: Number of trials per configuration
        output_dir: Directory to save results

    Returns:
        DataFrame with results
    """
    results = []
    k_classes = 10  # Number of classes for node classification task

    # Configuration: (name, n, q, expected_balance)
    configs = [
        # Balanced cycles: n·q is integer
        ("Balanced-n30-q0.33", 30, 1/3, True),   # 30·(1/3) = 10
        ("Balanced-n40-q0.25", 40, 1/4, True),   # 40·(1/4) = 10
        ("Balanced-n50-q0.2", 50, 1/5, True),    # 50·(1/5) = 10

        # Unbalanced cycles: n·q is NOT integer
        ("Unbalanced-n30-q0.5", 30, 0.5, False),  # 30·0.5 = 15 (but phase = 15·2π mod 2π ≠ 0)
        ("Unbalanced-n37-q0.5", 37, 0.5, False),  # 37·0.5 = 18.5 (not integer)
        ("Unbalanced-n50-q0.3", 50, 0.3, False),  # 50·0.3 = 15 (but let's check)
    ]

    print("="*80)
    print("Balanced vs Unbalanced Cycles Experiment")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Trials per config: {num_trials}")
    print(f"  Number of classes: {k_classes}")
    print("\nPhase Accumulation Check:")

    for name, n, q, expected_balanced in configs:
        phase_acc = compute_cycle_phase_accumulation(n, q)
        actual_balanced = is_balanced(phase_acc)
        status = "[OK]" if actual_balanced == expected_balanced else "[WARNING]"
        print(f"  {name}: phase={phase_acc:.4f}, balanced={actual_balanced} {status}")

    print()

    for name, n, q, expected_balanced in configs:
        print(f"\n{name} (n={n}, q={q}):")

        for trial in range(num_trials):
            # Generate cycle (same structure, balance determined by q parameter)
            if expected_balanced:
                edge_index, labels = generate_balanced_cycle(n, k_classes)
            else:
                edge_index, labels = generate_unbalanced_cycle(n, k_classes)

            num_classes = len(torch.unique(labels))

            # Build matrices with the specified q
            A_gcn, A_mag = build_matrices(edge_index, n, q)

            # Random features
            x = torch.randn(n, 16)

            # Train/test split
            perm = torch.randperm(n)
            train_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[perm[:int(0.6 * n)]] = True
            test_mask = ~train_mask

            # Compute unbalance score
            score = unbalance_score(edge_index, n, q)

            # Compute phase accumulation
            phase_acc = compute_cycle_phase_accumulation(n, q)
            balanced_flag = is_balanced(phase_acc)

            # Train GCN
            gcn = GCN(16, 32, num_classes)
            acc_gcn = train_eval(gcn, x, A_gcn, labels, train_mask, test_mask, epochs=200)

            # Train MagNet
            magnet = MagNet(16, 32, num_classes)
            acc_mag = train_eval(magnet, x, A_mag, labels, train_mask, test_mask, epochs=200)

            delta = acc_mag - acc_gcn

            results.append({
                "config": name,
                "n": n,
                "q": q,
                "num_classes": num_classes,
                "trial": trial,
                "phase_accumulation": phase_acc,
                "is_balanced": balanced_flag,
                "unbalance_score": score,
                "gcn": acc_gcn,
                "magnet": acc_mag,
                "delta": delta,
            })

            print(f"  {trial}: phase={phase_acc:.3f}, balanced={balanced_flag}, "
                  f"unbal_score={score:.3f}, GCN={acc_gcn:.3f}, MagNet={acc_mag:.3f}, d={delta:+.3f}")

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame, output_dir: Path):
    """
    Analyze results and test hypothesis.

    Hypothesis: δ (MagNet - GCN) is larger for unbalanced than balanced graphs.
    """
    print("\n" + "="*80)
    print("RESULTS BY BALANCE TYPE")
    print("="*80)

    # Group by balance type
    summary = df.groupby("is_balanced")[["phase_accumulation", "unbalance_score", "gcn", "magnet", "delta"]].mean()
    print(summary.round(3))

    # Statistical test: unbalanced vs balanced
    balanced = df[df["is_balanced"] == True]["delta"]
    unbalanced = df[df["is_balanced"] == False]["delta"]

    print(f"\n{'='*80}")
    print("STATISTICAL TESTS")
    print(f"{'='*80}")

    print(f"Balanced graphs (n={len(balanced)}):")
    print(f"  Mean delta: {balanced.mean():.4f}")
    print(f"  Std: {balanced.std():.4f}")

    print(f"\nUnbalanced graphs (n={len(unbalanced)}):")
    print(f"  Mean delta: {unbalanced.mean():.4f}")
    print(f"  Std: {unbalanced.std():.4f}")

    # T-test: unbalanced > balanced
    t, p = stats.ttest_ind(unbalanced, balanced, alternative='greater')
    print(f"\nT-test (unbalanced > balanced): t={t:.3f}, p={p:.4f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((balanced.std()**2 + unbalanced.std()**2) / 2)
    cohens_d = (unbalanced.mean() - balanced.mean()) / pooled_std
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")

    # Correlation: phase_accumulation vs delta
    # Filter out exactly balanced (phase ≈ 0)
    df_varied = df[df["phase_accumulation"] > 0.1].copy()
    r, p_corr = pearsonr(df_varied["phase_accumulation"], df_varied["delta"])
    print(f"\nCorrelation (phase accumulation vs delta): r={r:.3f}, p={p_corr:.4f}")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    if p < 0.05 and unbalanced.mean() > balanced.mean():
        print(f"[OK] HYPOTHESIS VALIDATED: MagNet shows greater advantage on unbalanced graphs")
        print(f"     Mean delta difference: {unbalanced.mean() - balanced.mean():+.3f} (p={p:.4f})")
        if abs(cohens_d) > 0.5:
            print(f"     Effect size is MEDIUM-LARGE (d={cohens_d:.3f})")
    elif p < 0.05 and unbalanced.mean() < balanced.mean():
        print(f"[!!] UNEXPECTED: Balanced graphs show larger MagNet advantage")
        print(f"     This contradicts the hypothesis from arxiv:2307.01813")
    else:
        print(f"[~] INCONCLUSIVE: No significant difference (p={p:.4f})")
        print(f"    Possible reasons:")
        print(f"    - Task structure dominates over balance/unbalance")
        print(f"    - Need more trials or different graph types")
        print(f"    - Phase parameter q may not be optimal")

    # Create visualization
    create_visualization(df, summary, t, p, output_dir)


def create_visualization(df: pd.DataFrame, summary: pd.DataFrame, t: float, p: float, output_dir: Path):
    """Create 3-panel visualization of results."""
    fig = plt.figure(figsize=(18, 5))

    # Panel 1: Box plot - delta by balance type
    ax1 = fig.add_subplot(1, 3, 1)
    balanced_data = df[df["is_balanced"] == True]["delta"]
    unbalanced_data = df[df["is_balanced"] == False]["delta"]
    bp = ax1.boxplot([balanced_data, unbalanced_data],
                      labels=["Balanced", "Unbalanced"],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax1.set_title(f"Performance Gap\n(t={t:.2f}, p={p:.4f})", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Scatter - phase accumulation vs delta
    ax2 = fig.add_subplot(1, 3, 2)
    for is_bal in [True, False]:
        sub = df[df["is_balanced"] == is_bal]
        label = "Balanced" if is_bal else "Unbalanced"
        color = 'steelblue' if is_bal else 'coral'
        ax2.scatter(sub["phase_accumulation"], sub["delta"],
                   label=label, color=color, alpha=0.6, s=60)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Phase Accumulation (mod 2π)", fontsize=12)
    ax2.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax2.set_title("Phase Frustration vs Performance", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Bar chart - GCN vs MagNet by balance type
    ax3 = fig.add_subplot(1, 3, 3)
    x_pos = np.arange(len(summary))
    width = 0.35
    ax3.bar(x_pos - width/2, summary["gcn"], width, label="GCN", color='steelblue', alpha=0.8)
    ax3.bar(x_pos + width/2, summary["magnet"], width, label="MagNet", color='coral', alpha=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(["Balanced", "Unbalanced"])
    ax3.set_ylabel("Accuracy", fontsize=12)
    ax3.set_title("Model Comparison", fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "balance_vs_unbalance_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Balanced vs Unbalanced Graphs Experiment")
    parser.add_argument('--num-trials', type=int, default=20, help='Number of trials per config')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run experiment
    df = run_experiment(num_trials=args.num_trials, output_dir=output_dir)

    # Save results
    df.to_csv(output_dir / "balance_vs_unbalance_results.csv", index=False)

    # Analyze
    analyze_results(df, output_dir)


if __name__ == "__main__":
    main()
