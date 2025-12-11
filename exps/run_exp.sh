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

echo "Running experiment $ID with config $CONFIG_FILE on device $DEVICE"
CUDA_VISIBLE_DEVICES=$DEVICE poetry run python train.py --config_path "$CONFIG_FILE"
