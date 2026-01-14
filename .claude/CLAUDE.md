# Dissertation: Networks with Complex Weights

## Project Overview
14-week dissertation on spectral properties of complex-weighted graphs and their applications to GNNs.

## Key Concepts
- **Hermitian/Magnetic adjacency**: Complex edge weights encoding direction via phase
- **Structural balance**: Unbalance score = 1 - rho(A_mag)/rho(A_sym)
- **GCN vs MagNet**: Testing when directional information helps

## Code Structure
```
src/
├── models/      # GCN, MagNet implementations
├── graphs/      # Matrix construction, graph generators
└── utils/       # Training utilities
experiments/     # Runnable experiment scripts
notebooks/       # Jupyter notebooks for exploration
scripts/         # RunPod and utility scripts
```

## Running Experiments

```bash
# Local
python experiments/structural_balance.py

# Remote (RunPod) - set RUNPOD_HOST, RUNPOD_PORT first
./scripts/run_remote.sh experiments/structural_balance.py
```

## Current Hypothesis
MagNet outperforms GCN on structurally unbalanced graphs (graphs with random edge directions creating odd cycles).

## Key Files
- `src/models/gcn.py` - Standard GCN
- `src/models/magnet.py` - Magnetic GNN
- `src/graphs/matrices.py` - Adjacency matrix construction
- `experiments/structural_balance.py` - Main experiment
