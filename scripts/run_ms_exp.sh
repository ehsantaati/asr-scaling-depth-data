#!/bin/bash
CONFIG_PATH=$1
DEVICE=${2:-0}

# Handle case where user passes --config_path as first arg
if [ "$CONFIG_PATH" == "--config_path" ]; then
  CONFIG_PATH=$2
  DEVICE=${3:-0}
fi

if [ -z "$CONFIG_PATH" ]; then
  echo "Usage: $0 <config_path> [device_id]"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: Config file not found: $CONFIG_PATH"
  exit 1
fi

echo "Running multi-seed experiment with config $CONFIG_PATH on device $DEVICE"
poetry run python run_multiseeds.py --config_path "$CONFIG_PATH" --device "$DEVICE"
