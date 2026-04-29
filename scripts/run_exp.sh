#!/bin/bash
CONFIG_PATH=$1
DEVICE=${2:-0}

if [ -z "$CONFIG_PATH" ]; then
  echo "Usage: $0 <config_path> [device_id]"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: Config file not found: $CONFIG_PATH"
  exit 1
fi

echo "Running experiment with config $CONFIG_PATH on device $DEVICE"
CUDA_VISIBLE_DEVICES=$DEVICE poetry run python train.py --config_path "$CONFIG_PATH"
