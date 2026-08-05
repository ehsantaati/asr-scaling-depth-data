#!/bin/bash
# Single training run. For the revision rerun campaign use the job queue instead
# (scripts/queue_launch.sh); this wrapper is kept for one-off / debugging runs.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PY="$(require_py)"

CONFIG_PATH=${1:-}
DEVICE=${2:-0}

if [ -z "$CONFIG_PATH" ]; then
  echo "Usage: $0 <config_path> [device_id]"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: Config file not found: $CONFIG_PATH"
  exit 1
fi

echo "Running experiment with config $CONFIG_PATH on device $DEVICE (python: $PY)"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$DEVICE" \
WANDB_MODE="${WANDB_MODE:-offline}" \
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
  "$PY" train.py --config_path "$CONFIG_PATH"
