#!/bin/bash
# Setup RunPod environment
# Run this once when you create a new pod

set -e

echo "=== Installing dependencies ==="
pip install torch numpy scipy pandas matplotlib pyyaml

echo "=== Creating dissertation directory ==="
mkdir -p ~/dissertation

echo "=== Setup complete ==="
echo "You can now run experiments with:"
echo "  ./scripts/run_remote.sh experiments/structural_balance.py"
