# Magnetic Laplacian Spectral Analysis Guide

## Overview

This framework provides tools to visualize and understand the eigenvectors and eigenvalues of the magnetic Laplacian, comparing them against the standard graph Laplacian. The implementation focuses on pedagogically valuable small graphs (3-10 nodes) with comprehensive visualizations.

## Quick Start

### Running the Complete Analysis

```bash
cd Dissertation
python experiments/spectral_visualization.py --q 0.1 --sweep-q --output-dir results/spectral_viz
```

### Analyzing Specific Graphs

```bash
python experiments/spectral_visualization.py --q 0.1 --graphs triangle_cycle path_4 diamond
```

### Parameter Sweep Only

```bash
python experiments/spectral_visualization.py --q 0.1 --sweep-q --graphs cycle_4
```

## Command Line Arguments

- `--q`: Phase parameter (default: 0.1)
- `--graphs`: Specific graphs to analyze (default: all)
  - Options: triangle_cycle, path_4, diamond, cycle_4, star_out, star_in, hexagon, two_triangles, hierarchical_3level, complete_4
- `--sweep-q`: Run parameter sweep over q values from 0.01 to 0.5
- `--output-dir`: Output directory for results (default: results/spectral_viz)

## Generated Outputs

### Directory Structure

```
results/spectral_viz/
├── graphs/               # 4-panel comparison figures for each graph
│   ├── triangle_cycle.png/.pdf
│   ├── path_4.png/.pdf
│   └── ...
├── sweeps/              # Parameter sweep plots
│   ├── triangle_cycle_sweep.png
│   ├── cycle_4_sweep.png
│   └── diamond_sweep.png
├── data/                # CSV files with numerical results
│   ├── triangle_cycle_eigenvalues.csv
│   ├── triangle_cycle_eigenvectors.csv
│   ├── triangle_cycle_sweep.csv
│   └── ...
└── comparisons/         # Future: comparative analysis figures
```

### Figure Descriptions

#### 4-Panel Comparison Figures (`graphs/*.png`)

Each graph gets a comprehensive 4-panel visualization:

- **Panel A**: Graph structure with directed edges
- **Panel B**: Eigenvalue spectrum comparison (standard vs magnetic)
- **Panel C**: Second eigenvector from standard Laplacian (Fiedler vector)
  - Real-valued, shown with diverging colormap (blue-white-red)
  - Node size indicates magnitude
- **Panel D**: Second eigenvector from magnetic Laplacian
  - Complex-valued, shown with phase (HSV colormap) and magnitude (size)
  - Node color indicates phase angle (-π to π)

#### Parameter Sweep Plots (`sweeps/*_sweep.png`)

4-panel plots showing how spectral properties vary with q parameter:

1. **Spectral Radius**: Maximum eigenvalue magnitude vs q
2. **Spectral Gap**: Difference between first two eigenvalues vs q
3. **Spectral Radius Ratio**: Ratio of magnetic to standard radius
4. **Eigenvalue Correlation**: Correlation between ordered eigenvalues

Key insights:
- As q increases, directional information becomes more prominent
- Optimal q varies by graph structure
- Correlation typically decreases with increasing q

## Pedagogical Graphs

### 1. triangle_cycle (3 nodes)
- **Structure**: 0→1→2→0 (directed cycle)
- **Key Insight**: Simplest non-trivial cycle showing direction encoding via phase
- **Expected**: Magnetic spectrum differs significantly from standard

### 2. path_4 (4 nodes)
- **Structure**: 0→1→2→3 (directed path)
- **Key Insight**: Hierarchical flow without cycles - eigenvectors show gradient
- **Expected**: Spectra are nearly identical (no cyclic structure)

### 3. diamond (4 nodes)
- **Structure**: 0→{1,2}, {1,2}→3 (two parallel paths)
- **Key Insight**: Multiple pathways demonstrating eigenvector localization
- **Expected**: Eigenvectors distinguish between the two paths

### 4. cycle_4 (4 nodes)
- **Structure**: 0→1→2→3→0 (simple cycle)
- **Key Insight**: Simple cycle ideal for parameter sweeps
- **Expected**: Strong q-dependence in spectral properties

### 5. star_out (5 nodes)
- **Structure**: 0→{1,2,3,4} (broadcast pattern)
- **Key Insight**: Hub centrality (broadcast) - eigenvector localization on hub
- **Expected**: First eigenvector concentrated on node 0

### 6. star_in (5 nodes)
- **Structure**: {1,2,3,4}→0 (aggregation pattern)
- **Key Insight**: Contrast with star_out, same structure but reversed
- **Expected**: Similar spectrum but opposite eigenvector orientation

### 7. hexagon (6 nodes)
- **Structure**: 0→1→2→3→4→5→0 (6-node cycle)
- **Key Insight**: Larger cycle for structural balance exploration
- **Expected**: Even more pronounced differences than triangle

### 8. two_triangles (6 nodes)
- **Structure**: Two triangles with bidirectional bridge
- **Key Insight**: Community structure detection via Fiedler vector
- **Expected**: Second eigenvector separates communities

### 9. hierarchical_3level (9 nodes)
- **Structure**: 3 levels × 3 nodes, edges flow forward
- **Key Insight**: Miniature hierarchical DAG - eigenvectors align with levels
- **Expected**: Eigenvectors encode hierarchical position

### 10. complete_4 (4 nodes)
- **Structure**: All edges bidirectional (fully connected)
- **Key Insight**: Baseline - no directional bias
- **Expected**: Magnetic spectrum similar to standard (all edges reciprocal)

## Understanding the Results

### Key Observations from Experimental Run

1. **Cycles show large differences**:
   - triangle_cycle: spectral radius ratio = 1.28, gap ratio = 0.47
   - cycle_4: spectral radius ratio = 0.90, gap ratio = 0.22
   - hexagon: spectral radius ratio = 0.96, gap ratio = 0.21

2. **DAGs are nearly identical**:
   - path_4: spectral radius ratio = 1.00, gap ratio = 1.00
   - hierarchical_3level: spectral radius ratio = 1.00, gap ratio = 1.00
   - diamond: spectral radius ratio = 1.00, gap ratio = 1.00

3. **Stars show no directional bias**:
   - star_out & star_in: identical spectra despite reversed directions
   - This is because the magnetic phase encoding preserves the undirected structure

### Interpreting Eigenvalue Correlation

- **Correlation ≈ 1.0**: Spectra are very similar (e.g., path_4, diamond)
- **Correlation ≈ 0.8-0.9**: Moderate differences (e.g., triangle_cycle, cycle_4)
- Lower correlation indicates more directional information captured by magnetic Laplacian

### Interpreting Spectral Gap Ratio

- **Ratio ≈ 1.0**: Magnetic Laplacian has same gap as standard
- **Ratio < 1.0**: Magnetic Laplacian has smaller gap (cycles)
- Smaller gap in magnetic Laplacian indicates directional asymmetry

## Using the Framework for Your Dissertation

### 1. Understanding Direction Encoding

Compare Panel C (standard eigenvector) vs Panel D (magnetic eigenvector) to see:
- How phase patterns encode edge directions
- How magnitude patterns differ from real-valued case
- Which eigenvectors are most affected by direction

### 2. Optimal q Parameter Selection

Use parameter sweeps to determine:
- Which q maximizes directional information for your graph type
- How robust spectral properties are to q variation
- Trade-offs between directional encoding and numerical stability

### 3. Connecting to GNN Performance

Use these insights to explain:
- Why MagNet outperforms GCN on structurally unbalanced graphs
- How spectral gap relates to information propagation
- Why certain graph structures benefit more from magnetic Laplacian

### 4. Thesis Figures

The generated figures are publication-quality (300 DPI) and available in both PNG and PDF:
- Use 4-panel figures to illustrate specific graph examples
- Use parameter sweeps to show q-dependence
- CSV data can be used for custom analysis in Jupyter notebooks

## Programmatic Usage

### Computing Spectrum

```python
from src.graphs.generators import generate_small_graph
from src.graphs.spectral import compute_laplacian_spectrum

# Generate graph
edge_index, num_nodes, description = generate_small_graph("triangle_cycle")

# Compute spectrum
spectrum = compute_laplacian_spectrum(edge_index, num_nodes, q=0.1)

# Access results
eigenvals_std = spectrum['eigenvalues_standard']
eigenvals_mag = spectrum['eigenvalues_magnetic']
eigenvecs_std = spectrum['eigenvectors_standard']
eigenvecs_mag = spectrum['eigenvectors_magnetic']  # Complex-valued
```

### Creating Custom Visualizations

```python
from src.visualization.spectral_plots import plot_eigenvector_on_graph

# Visualize specific eigenvector
plot_eigenvector_on_graph(
    edge_index, num_nodes,
    eigenvecs_mag[:, 1],  # Second eigenvector
    title="Magnetic Fiedler Vector",
    is_complex=True,
    save_path="my_figure.png"
)
```

### Parameter Sweep

```python
from src.graphs.spectral import parameter_sweep_q
import numpy as np

q_values = np.linspace(0.01, 0.5, 50)
sweep_df = parameter_sweep_q(edge_index, num_nodes, q_values)

# Access results
print(sweep_df[['q', 'spectral_radius_ratio', 'eigenvalue_correlation']])
```

## Next Steps

1. **Explore the visualizations**: Open the PNG files in `results/spectral_viz/graphs/`
2. **Examine the data**: Look at CSV files to see numerical values
3. **Try different q values**: Run with different --q parameters to see effects
4. **Add custom graphs**: Extend `generate_small_graph()` with your own structures
5. **Create Jupyter notebook**: Use the framework interactively for exploration

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the Dissertation directory:
```bash
cd Dissertation
python experiments/spectral_visualization.py
```

### No Module Named 'src'
The script adds the parent directory to Python path, but if issues persist:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Dissertation"
```

### Memory Issues with Large Graphs
The framework is designed for small pedagogical graphs (3-10 nodes). For larger graphs:
- Reduce the number of eigenvectors saved
- Skip eigenvector heatmaps
- Process graphs individually instead of all at once

## Citation

If you use this framework in your dissertation, consider acknowledging:
- NetworkX for graph layouts
- NumPy/SciPy for eigenvalue computations
- Matplotlib for visualizations
- The magnetic Laplacian construction from your supervisor's work

## Contact

For questions or issues with this framework, refer to:
- The implementation plan: `~/.claude/plans/misty-leaping-liskov.md`
- Source code documentation in module docstrings
- Your dissertation supervisor
