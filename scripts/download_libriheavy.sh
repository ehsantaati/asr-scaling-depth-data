#!/bin/bash
# Download libriheavy validation and test sets (cuts files only)
# Usage: bash scripts/download_libriheavy.sh

set -e  # Exit on error

# Base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/libriheavy"

echo "Downloading libriheavy validation and test sets..."
echo "Target directory: $DATA_DIR"

# Create directory
mkdir -p "$DATA_DIR"

# Base URL
BASE_URL="https://huggingface.co/datasets/pkufool/libriheavy/resolve/main"

# Download validation file
echo ""
echo "Downloading validation set (cuts file)..."
wget -nc "$BASE_URL/libriheavy_cuts_dev.jsonl.gz" -P "$DATA_DIR/"

# Download test files
echo ""
echo "Downloading test set (cuts files)..."
wget -nc "$BASE_URL/libriheavy_cuts_test_clean.jsonl.gz" -P "$DATA_DIR/"
wget -nc "$BASE_URL/libriheavy_cuts_test_other.jsonl.gz" -P "$DATA_DIR/"

echo ""
echo "Download complete!"
echo "Files saved to: $DATA_DIR"
echo ""
echo "Downloaded files:"
echo "  - libriheavy_cuts_dev.jsonl.gz (validation)"
echo "  - libriheavy_cuts_test_clean.jsonl.gz (test clean)"
echo "  - libriheavy_cuts_test_other.jsonl.gz (test other)"
echo ""
echo "Note: Audio files are embedded in the HuggingFace dataset."
echo "Text is extracted from the 'supervisions' field automatically."
