#!/bin/bash
ID=$1
DEVICE=${2:-0}

if [ -z "$ID" ]; then
  echo "Usage: $0 <experiment_id> [device_id]"
  exit 1
fi

# Find config file starting with ID
CONFIG_FILE=$(find exps -name "${ID}_*.yaml" | head -n 1)

if [ -z "$CONFIG_FILE" ]; then
  echo "Error: No config file found for ID $ID in exps/"
  exit 1
fi

echo "Running multi-seed experiment $ID with config $CONFIG_FILE on device $DEVICE"
# Assuming run_multiseeds.py is in the root directory as moved previously
poetry run python run_multiseeds.py --config_path "$CONFIG_FILE" --device "$DEVICE"
