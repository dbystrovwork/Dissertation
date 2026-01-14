"""
Spectral-GNN Multi-Graph Correlation Experiment (Phase 3)

Tests the hypothesis that spectral properties of the magnetic Laplacian
predict when MagNet outperforms GCN across MULTIPLE graph families.

This extends spectral_gnn_correlation.py to include:
- Cycles (directed, mixed bidirectional, undirected)
- Trees (directed, random orientation)
- Small-world graphs (varying rewiring probability)
- Hierarchical DAGs (baseline from Phase 2)

Goal: Determine if spectral-performance correlation is graph-family specific.
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
    generate_hierarchical_dag,
    generate_directed_cycle,
    generate_cycle_with_bidir,
    generate_tree,
    generate_small_world,
)
from src.graphs.spectral import compute_laplacian_spectrum, compare_spectra
from src.utils import train_eval


# Graph configurations
GRAPH_CONFIGS = [
    # Hierarchical (baseline from Phase 2)
    ("Hierarchical-3", generate_hierarchical_dag, {"n": 100, "num_levels": 3, "p_forward": 0.3, "p_skip": 0.1, "p_lateral": 0.05}),
    ("Hierarchical-4", generate_hierarchical_dag, {"n": 100, "num_levels": 4, "p_forward": 0.3, "p_skip": 0.1, "p_lateral": 0.05}),

    # Cycles (expect varied spectral properties)
    ("Cycle-Directed", generate_directed_cycle, {"n": 100}),
    ("Cycle-50Bidir", generate_cycle_with_bidir, {"n": 100, "bidir_frac": 0.5}),
    ("Cycle-Undirected", generate_cycle_with_bidir, {"n": 100, "bidir_frac": 1.0}),

    # Trees (expect small spectral difference like paths)
    ("Tree-Binary-Directed", generate_tree, {"n": 100, "branching": 2, "directed": True}),
    ("Tree-Binary-Random", generate_tree, {"n": 100, "branching": 2, "directed": False}),

    # Small world (intermediate)
    ("SmallWorld-p0.1", generate_small_world, {"n": 100, "k": 4, "p": 0.1, "directed": True}),
    ("SmallWorld-p0.3", generate_small_world, {"n": 100, "k": 4, "p": 0.3, "directed": True}),
]


def run_experiment(num_trials: int = 10, q: float = 0.5, graph_types: list = None) -> pd.DataFrame:
    """Run experiment across multiple graph types."""
    results = []

    # Filter configs if specific graph types requested
    if graph_types:
        configs = [c for c in GRAPH_CONFIGS if c[0] in graph_types]
    else:
        configs = GRAPH_CONFIGS

    for graph_name, gen_func, kwargs in configs:
        print(f"\n{'='*60}")
        print(f"{graph_name}")
        print(f"{'='*60}")

        for trial in range(num_trials):
            # Generate graph
            edge_index, labels = gen_func(**kwargs)
            n = len(labels)

            # Handle potential size mismatch (e.g., tree truncation)
            if n != kwargs.get('n', n):
                print(f"  Note: Graph has {n} nodes (requested {kwargs.get('n', n)})")

            # Determine number of classes
            num_classes = len(torch.unique(labels))

            # Build matrices
            A_gcn, A_mag = build_matrices(edge_index, n, q)

            # Compute spectral properties
            try:
                spectrum = compute_laplacian_spectrum(edge_index, n, q)
                comparison = compare_spectra(
                    spectrum['eigenvalues_standard'],
                    spectrum['eigenvalues_magnetic']
                )
            except Exception as e:
                print(f"  [WARNING] Trial {trial}: Spectral computation failed: {e}")
                continue

            # Features and masks
            x = torch.randn(n, 16)
            perm = torch.randperm(n)
            train_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[perm[:int(0.6 * n)]] = True
            test_mask = ~train_mask

            # Compute unbalance score
            score = unbalance_score(edge_index, n, q)

            # Train GCN
            gcn = GCN(16, 32, num_classes)
            acc_gcn = train_eval(gcn, x, A_gcn, labels, train_mask, test_mask)

            # Train MagNet
            magnet = MagNet(16, 32, num_classes)
            acc_mag = train_eval(magnet, x, A_mag, labels, train_mask, test_mask)

            delta = acc_mag - acc_gcn

            results.append({
                "graph_type": graph_name,
                "trial": trial,
                "num_nodes": n,
                "num_classes": num_classes,
                "unbalance": score,
                "gcn": acc_gcn,
                "magnet": acc_mag,
                "delta": delta,
                # Spectral metrics
                "spectral_radius_ratio": comparison['spectral_radius_ratio'],
                "spectral_gap_ratio": comparison['spectral_gap_ratio'],
                "eigenvalue_correlation": comparison['eigenvalue_correlation'],
            })

            print(f"  {trial}: n={n}, classes={num_classes}, gap_ratio={comparison['spectral_gap_ratio']:.3f}, "
                  f"GCN={acc_gcn:.3f}, MagNet={acc_mag:.3f}, d={delta:+.3f}")

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize results with multi-graph breakdown."""
    # Summary by graph type
    print("\n" + "=" * 80)
    print("RESULTS BY GRAPH TYPE")
    print("=" * 80)
    summary = df.groupby("graph_type")[["num_nodes", "num_classes", "unbalance", "gcn", "magnet", "delta",
                                         "spectral_gap_ratio", "eigenvalue_correlation"]].mean()
    print(summary.round(3))

    # Overall stats
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Overall mean delta: {df['delta'].mean():.3f}")
    print(f"MagNet wins: {(df['delta'] > 0).sum()}/{len(df)} trials ({100*(df['delta'] > 0).sum()/len(df):.1f}%)")

    # T-test: is delta significantly > 0?
    t, p = stats.ttest_1samp(df["delta"], 0, alternative='greater')
    print(f"T-test (delta > 0): t={t:.3f}, p={p:.4f}")

    # Spectral-Performance Correlations
    print("\n" + "=" * 80)
    print("SPECTRAL-PERFORMANCE CORRELATIONS (All Graphs)")
    print("=" * 80)

    # Remove outliers (gap_ratio > 100 indicates numerical issues)
    df_clean = df[df['spectral_gap_ratio'] < 100].copy()
    print(f"Removed {len(df) - len(df_clean)} outlier trials")

    r_gap, p_gap = pearsonr(df_clean['spectral_gap_ratio'], df_clean['delta'])
    r_radius, p_radius = pearsonr(df_clean['spectral_radius_ratio'], df_clean['delta'])
    r_corr, p_corr = pearsonr(df_clean['eigenvalue_correlation'], df_clean['delta'])

    print(f"Gap Ratio vs Delta: r={r_gap:.3f}, p={p_gap:.4f}")
    print(f"Radius Ratio vs Delta: r={r_radius:.3f}, p={p_radius:.4f}")
    print(f"Eigenvalue Corr vs Delta: r={r_corr:.3f}, p={p_corr:.4f}")

    # Interpretation
    print("\nInterpretation:")
    if abs(r_gap) > 0.4 and p_gap < 0.05:
        print(f"  [OK] STRONG correlation between spectral gap ratio and performance ({r_gap:.3f})")
        print("  Hypothesis H1 VALIDATED: Spectral properties predict GNN performance!")
    elif abs(r_gap) > 0.2:
        print(f"  MODERATE correlation ({r_gap:.3f}) - some predictive power")
    else:
        print(f"  WEAK correlation ({r_gap:.3f}) - spectral gap may not be universal predictor")

    # Grouped correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATIONS BY GRAPH FAMILY")
    print("=" * 80)

    families = {
        "Hierarchical": ["Hierarchical-3", "Hierarchical-4"],
        "Cycles": ["Cycle-Directed", "Cycle-50Bidir", "Cycle-Undirected"],
        "Trees": ["Tree-Binary-Directed", "Tree-Binary-Random"],
        "SmallWorld": ["SmallWorld-p0.1", "SmallWorld-p0.3"],
    }

    for family, graph_types in families.items():
        family_df = df_clean[df_clean['graph_type'].isin(graph_types)]
        if len(family_df) > 3:  # Need at least 3 points for correlation
            r_fam, p_fam = pearsonr(family_df['spectral_gap_ratio'], family_df['delta'])
            print(f"{family:15s}: r={r_fam:+.3f}, p={p_fam:.4f}, n={len(family_df)}")

    # Create visualization
    create_multigraph_visualization(df_clean, summary, r_gap, p_gap, output_dir)

    # Conclusion
    print("\n" + "=" * 80)
    mean_delta = df["delta"].mean()
    if mean_delta > 0.02 and p < 0.05:
        print(f"SUCCESS: MagNet outperforms GCN overall (mean delta={mean_delta:.3f}, p={p:.4f})")

        if abs(r_gap) > 0.3 and p_gap < 0.05:
            print(f"\nBONUS: Spectral-performance correlation detected (r={r_gap:.3f}, p={p_gap:.4f})")
            print("Spectral properties show predictive power across graph families!")
        else:
            print(f"\nNote: Weak spectral correlation (r={r_gap:.3f}) suggests task-structure alignment")
            print("dominates over pure spectral properties.")
    else:
        print(f"Mixed results: Mean delta={mean_delta:.3f}, p={p:.4f}")
    print("=" * 80)


def create_multigraph_visualization(df: pd.DataFrame, summary: pd.DataFrame, r_gap: float, p_gap: float, output_dir: Path):
    """Create comprehensive 4-panel visualization."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Define color map for graph families
    family_colors = {
        "Hierarchical-3": "#1f77b4",
        "Hierarchical-4": "#1f77b4",
        "Cycle-Directed": "#ff7f0e",
        "Cycle-50Bidir": "#ff7f0e",
        "Cycle-Undirected": "#ff7f0e",
        "Tree-Binary-Directed": "#2ca02c",
        "Tree-Binary-Random": "#2ca02c",
        "SmallWorld-p0.1": "#d62728",
        "SmallWorld-p0.3": "#d62728",
    }

    # Panel 1: Delta by graph type (box plot)
    ax1 = fig.add_subplot(gs[0, 0])
    graph_types = df['graph_type'].unique()
    data_by_type = [df[df['graph_type'] == gt]['delta'].values for gt in graph_types]
    bp = ax1.boxplot(data_by_type, labels=graph_types, patch_artist=True)
    for patch, gt in zip(bp['boxes'], graph_types):
        patch.set_facecolor(family_colors.get(gt, '#999999'))
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax1.set_title("Performance Gap by Graph Type", fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Spectral gap ratio vs delta (scatter)
    ax2 = fig.add_subplot(gs[0, 1])
    for gt in graph_types:
        sub = df[df['graph_type'] == gt]
        ax2.scatter(sub['spectral_gap_ratio'], sub['delta'],
                   label=gt, color=family_colors.get(gt, '#999999'),
                   alpha=0.6, s=60)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Spectral Gap Ratio (magnetic/standard)", fontsize=12)
    ax2.set_ylabel("MagNet - GCN Accuracy", fontsize=12)
    ax2.set_title(f"Spectral Prediction (r={r_gap:.3f}, p={p_gap:.4f})", fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Add trend line if correlation is significant
    if abs(r_gap) > 0.2 and p_gap < 0.1:
        x_trend = np.linspace(df['spectral_gap_ratio'].min(), df['spectral_gap_ratio'].max(), 100)
        z = np.polyfit(df['spectral_gap_ratio'], df['delta'], 1)
        p = np.poly1d(z)
        ax2.plot(x_trend, p(x_trend), "r--", alpha=0.7, linewidth=2)

    # Panel 3: Bar chart comparison
    ax3 = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(len(summary))
    width = 0.35
    ax3.bar(x_pos - width/2, summary['gcn'], width, label='GCN', color='steelblue', alpha=0.8)
    ax3.bar(x_pos + width/2, summary['magnet'], width, label='MagNet', color='coral', alpha=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(summary.index, rotation=45, ha='right')
    ax3.set_ylabel("Accuracy", fontsize=12)
    ax3.set_title("Model Comparison by Graph Type", fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Spectral gap ratio by graph type
    ax4 = fig.add_subplot(gs[1, 1])
    gap_by_type = [df[df['graph_type'] == gt]['spectral_gap_ratio'].values for gt in graph_types]
    bp2 = ax4.boxplot(gap_by_type, labels=graph_types, patch_artist=True)
    for patch, gt in zip(bp2['boxes'], graph_types):
        patch.set_facecolor(family_colors.get(gt, '#999999'))
    ax4.set_ylabel("Spectral Gap Ratio", fontsize=12)
    ax4.set_title("Spectral Diversity Across Graph Types", fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "spectral_gnn_multigraph_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-graph spectral-GNN correlation experiment")
    parser.add_argument('--num-trials', type=int, default=10, help='Number of trials per graph type')
    parser.add_argument('--q', type=float, default=0.5, help='Phase parameter')
    parser.add_argument('--graph-types', nargs='+', default=None,
                       help='Specific graph types to test (default: all)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Graph Spectral-GNN Correlation Experiment (Phase 3)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Trials per graph: {args.num_trials}")
    print(f"  Phase parameter q: {args.q}")
    print(f"  Graph types: {args.graph_types if args.graph_types else 'all'}")

    # Run experiment
    df = run_experiment(num_trials=args.num_trials, q=args.q, graph_types=args.graph_types)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / "spectral_gnn_multigraph_results.csv", index=False)

    # Analyze
    analyze_results(df, output_dir)


if __name__ == "__main__":
    main()
