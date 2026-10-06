#!/bin/bash
# Run one explicitly selected training experiment.
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

CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

case "$CONFIG_PATH" in
  *.yaml|*.yml) ;;
  *) echo "Error: configuration must be a .yaml or .yml file: $CONFIG_PATH" >&2; exit 1 ;;
esac

require_dependencies "$PY"

echo "Running experiment with config $CONFIG_PATH on device $DEVICE (python: $PY)"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$DEVICE" \
WANDB_MODE="${WANDB_MODE:-offline}" \
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
  "$PY" train.py --config_path "$CONFIG_PATH"
