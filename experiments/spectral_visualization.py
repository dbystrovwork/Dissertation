"""Spectral visualization experiments for magnetic Laplacian."""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.graphs.generators import generate_small_graph
from src.graphs.spectral import (
    compute_laplacian_spectrum,
    compare_spectra,
    parameter_sweep_q,
)
from src.visualization.spectral_plots import (
    plot_eigenvalue_spectrum,
    plot_spectrum_comparison,
    plot_parameter_sweep,
)


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_output_directories(output_dir: Path):
    """Create directory structure for results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graphs").mkdir(exist_ok=True)
    (output_dir / "sweeps").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)


def generate_all_pedagogical_graphs():
    """Generate all pedagogical graphs for analysis."""
    graph_names = [
        "triangle_cycle",
        "path_4",
        "diamond",
        "cycle_4",
        "star_out",
        "star_in",
        "hexagon",
        "two_triangles",
        "hierarchical_3level",
        "complete_4",
    ]

    graphs = {}
    for name in graph_names:
        edge_index, num_nodes, description = generate_small_graph(name)
        graphs[name] = (edge_index, num_nodes, description)

    return graphs


def save_spectrum_data(name: str, spectrum: dict, output_dir: Path):
    """Save spectral data to CSV."""
    import pandas as pd

    # Eigenvalues
    df_eigenvals = pd.DataFrame({
        'index': range(len(spectrum['eigenvalues_standard'])),
        'eigenvalue_standard': spectrum['eigenvalues_standard'],
        'eigenvalue_magnetic': spectrum['eigenvalues_magnetic'],
    })
    df_eigenvals.to_csv(output_dir / f"{name}_eigenvalues.csv", index=False)

    # Eigenvectors (save first 3 for brevity)
    num_to_save = min(3, spectrum['eigenvectors_standard'].shape[1])
    evec_data = {'node': range(spectrum['eigenvectors_standard'].shape[0])}

    for i in range(num_to_save):
        evec_data[f'evec_std_{i}'] = spectrum['eigenvectors_standard'][:, i]
        evec_mag = spectrum['eigenvectors_magnetic'][:, i]
        evec_data[f'evec_mag_{i}_real'] = evec_mag.real
        evec_data[f'evec_mag_{i}_imag'] = evec_mag.imag
        evec_data[f'evec_mag_{i}_magnitude'] = np.abs(evec_mag)
        evec_data[f'evec_mag_{i}_phase'] = np.angle(evec_mag)

    df_eigenvecs = pd.DataFrame(evec_data)
    df_eigenvecs.to_csv(output_dir / f"{name}_eigenvectors.csv", index=False)


def print_summary(name: str, description: str, spectrum: dict):
    """Print summary of spectral properties."""
    print(f"\n{'='*80}")
    print(f"Graph: {name}")
    print(f"Description: {description}")
    print(f"{'='*80}")

    eigenvals_std = spectrum['eigenvalues_standard']
    eigenvals_mag = spectrum['eigenvalues_magnetic']

    print(f"\nNumber of nodes: {len(eigenvals_std)}")
    print(f"\nStandard Laplacian:")
    print(f"  Spectral radius: {np.max(np.abs(eigenvals_std)):.4f}")
    print(f"  Spectral gap: {spectrum['spectral_gap_standard']:.4f}")
    print(f"  First 3 eigenvalues: {eigenvals_std[:3]}")

    print(f"\nMagnetic Laplacian:")
    print(f"  Spectral radius: {np.max(np.abs(eigenvals_mag)):.4f}")
    print(f"  Spectral gap: {spectrum['spectral_gap_magnetic']:.4f}")
    print(f"  First 3 eigenvalues: {eigenvals_mag[:3]}")

    # Comparison
    comparison = compare_spectra(eigenvals_std, eigenvals_mag)
    print(f"\nComparison:")
    print(f"  Spectral radius ratio: {comparison['spectral_radius_ratio']:.4f}")
    print(f"  Spectral gap ratio: {comparison['spectral_gap_ratio']:.4f}")
    print(f"  Eigenvalue correlation: {comparison['eigenvalue_correlation']:.4f}")


def compare_balanced_vs_unbalanced(output_dir: Path):
    """Compare balanced and unbalanced graph variants."""
    print("\n" + "="*80)
    print("Comparative Analysis: Balanced vs Unbalanced")
    print("="*80)

    # For now, just print a message - can extend later
    print("\nThis analysis compares structural properties of balanced vs unbalanced graphs.")
    print("Future extension: Generate balanced/unbalanced variants and compare spectra.")


def compare_symmetric_vs_asymmetric(output_dir: Path):
    """Compare symmetric and asymmetric graph variants."""
    print("\n" + "="*80)
    print("Comparative Analysis: Symmetric vs Asymmetric")
    print("="*80)

    # For now, just print a message - can extend later
    print("\nThis analysis compares symmetric vs asymmetric edge patterns.")
    print("Future extension: Generate symmetric/asymmetric variants and compare spectra.")


def create_master_figure(graphs: dict, q: float, save_path: Path):
    """Create master summary figure with all graphs."""
    print("\n" + "="*80)
    print("Creating Master Figure")
    print("="*80)

    # For now, just print a message - can extend later with grid layout
    print(f"\nMaster figure generation: {save_path}")
    print("Future extension: Grid layout with all pedagogical graphs.")


def main(args):
    """Main experiment workflow."""
    print("="*80)
    print("Magnetic Laplacian Spectral Visualization")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  q parameter: {args.q}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Parameter sweep: {args.sweep_q}")
    print(f"  Graphs to analyze: {args.graphs if args.graphs != 'all' else 'all'}")

    # Setup
    set_random_seed(42)
    output_dir = Path(args.output_dir)
    create_output_directories(output_dir)

    # Generate all pedagogical graphs
    print("\n" + "="*80)
    print("Generating Pedagogical Graphs")
    print("="*80)
    graphs = generate_all_pedagogical_graphs()
    print(f"\nGenerated {len(graphs)} graphs:")
    for name, (edge_index, num_nodes, description) in graphs.items():
        print(f"  - {name}: {num_nodes} nodes, {edge_index.shape[1]} edges")

    # Filter graphs if specified
    if args.graphs != 'all':
        selected_graphs = {name: graphs[name] for name in args.graphs if name in graphs}
        if len(selected_graphs) == 0:
            print(f"\nWarning: None of the specified graphs found. Using all graphs.")
            selected_graphs = graphs
        else:
            graphs = selected_graphs
            print(f"\nAnalyzing {len(graphs)} selected graphs: {list(graphs.keys())}")

    # Analyze each graph
    print("\n" + "="*80)
    print("Analyzing Graphs")
    print("="*80)

    for name, (edge_index, num_nodes, description) in graphs.items():
        print(f"\nProcessing {name}...")

        # Compute spectral properties
        spectrum = compute_laplacian_spectrum(edge_index, num_nodes, q=args.q)

        # Create comparison figure (4-panel)
        plot_spectrum_comparison(
            edge_index, num_nodes, args.q, name,
            output_dir / "graphs", spectrum=spectrum
        )

        # Save numerical data
        save_spectrum_data(name, spectrum, output_dir / "data")

        # Print summary
        print_summary(name, description, spectrum)

        print(f"  [OK] Saved to {output_dir / 'graphs' / name}.png")

    # Parameter sweep
    if args.sweep_q:
        print("\n" + "="*80)
        print("Parameter Sweep Analysis")
        print("="*80)

        sweep_graphs = ["triangle_cycle", "cycle_4", "diamond"]
        # Only sweep graphs that exist in our filtered set
        sweep_graphs = [name for name in sweep_graphs if name in graphs]

        if len(sweep_graphs) == 0:
            print("\nWarning: No default sweep graphs found in selected set. Skipping sweep.")
        else:
            q_values = np.linspace(0.01, 0.5, 20)
            print(f"\nSweeping q from {q_values[0]} to {q_values[-1]} ({len(q_values)} values)")
            print(f"Graphs: {sweep_graphs}")

            for name in sweep_graphs:
                print(f"\n  Processing {name}...")
                edge_index, num_nodes, _ = graphs[name]

                sweep_df = parameter_sweep_q(edge_index, num_nodes, q_values)

                # Plot
                plot_parameter_sweep(
                    sweep_df, name,
                    output_dir / "sweeps" / f"{name}_sweep.png"
                )

                # Save data
                sweep_df.to_csv(output_dir / "data" / f"{name}_sweep.csv", index=False)

                print(f"    [OK] Saved sweep results")

    # Comparative analysis
    compare_balanced_vs_unbalanced(output_dir)
    compare_symmetric_vs_asymmetric(output_dir)

    # Master figure
    create_master_figure(graphs, args.q, output_dir / "master_figure.png")

    # Final summary
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir.absolute()}")
    print(f"\nOutput files:")
    print(f"  - Individual graph analyses: {output_dir / 'graphs'}")
    print(f"  - Numerical data (CSV): {output_dir / 'data'}")
    if args.sweep_q and len(sweep_graphs) > 0:
        print(f"  - Parameter sweeps: {output_dir / 'sweeps'}")
    print(f"\nTo view results, navigate to the output directory and open the PNG files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize spectral properties of magnetic Laplacian"
    )
    parser.add_argument(
        '--q', type=float, default=0.1,
        help='Phase parameter (default: 0.1)'
    )
    parser.add_argument(
        '--graphs', nargs='+', default='all',
        help='Which graphs to analyze (default: all). Options: triangle_cycle, path_4, diamond, cycle_4, star_out, star_in, hexagon, two_triangles, hierarchical_3level, complete_4'
    )
    parser.add_argument(
        '--sweep-q', action='store_true',
        help='Run parameter sweep over q values'
    )
    parser.add_argument(
        '--output-dir', default='results/spectral_viz',
        help='Output directory for results (default: results/spectral_viz)'
    )

    args = parser.parse_args()
    main(args)
