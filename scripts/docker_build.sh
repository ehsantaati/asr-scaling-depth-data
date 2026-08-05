#!/bin/bash
# Build the image with the correct context (the repo root), whatever directory you
# happen to be in. Exists because `docker build -f docker/Dockerfile ... docker` --
# the VS Code Docker extension's default, which uses the Dockerfile's own directory
# as the context -- fails with "/pyproject.toml: not found".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${TAG:-asr-data-scaling:latest}"

cd "$REPO_ROOT"
echo "Building $TAG   (context: $REPO_ROOT, dockerfile: docker/Dockerfile)"
exec docker build -f docker/Dockerfile -t "$TAG" "$@" .
