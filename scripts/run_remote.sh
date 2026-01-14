#!/bin/bash
# Run experiment on RunPod
# Usage: ./scripts/run_remote.sh <script.py> [args]
#
# Required environment variables:
#   RUNPOD_HOST - Pod IP address
#   RUNPOD_PORT - SSH port (default: 22)
#   RUNPOD_USER - SSH user (default: root)

set -e

RUNPOD_HOST="${RUNPOD_HOST:?Error: RUNPOD_HOST not set}"
RUNPOD_PORT="${RUNPOD_PORT:-22}"
RUNPOD_USER="${RUNPOD_USER:-root}"

SCRIPT="${1:?Error: No script specified}"
shift
ARGS="$@"

REMOTE_DIR="~/dissertation"

echo "=== Syncing code to RunPod ==="
rsync -avz --exclude 'results/' --exclude '.git/' --exclude '__pycache__/' \
    -e "ssh -p $RUNPOD_PORT" \
    . ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/

echo "=== Running: python $SCRIPT $ARGS ==="
ssh -p $RUNPOD_PORT ${RUNPOD_USER}@${RUNPOD_HOST} \
    "cd ${REMOTE_DIR} && python $SCRIPT $ARGS"

echo "=== Syncing results back ==="
rsync -avz -e "ssh -p $RUNPOD_PORT" \
    ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/results/ ./results/

echo "=== Done ==="
