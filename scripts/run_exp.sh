#!/bin/bash
# Single training run. For the revision rerun campaign use the job queue instead
# (scripts/queue_launch.sh); this wrapper is kept for one-off / debugging runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"

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

if [ ! -x "$PY" ]; then
  echo "Error: interpreter not found at $PY (set PY=... to override)"
  exit 1
fi

echo "Running experiment with config $CONFIG_PATH on device $DEVICE"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$DEVICE" \
WANDB_MODE="${WANDB_MODE:-offline}" \
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
  "$PY" train.py --config_path "$CONFIG_PATH"
