#!/bin/bash
# Download QM9 checkpoints for all baselines
set -e

BASELINES_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$BASELINES_DIR/scripts"

echo "=== Downloading all QM9 checkpoints ==="
echo ""

for method in geodiff geomol mcf tordiff etflow; do
    echo "--- Downloading: $method ---"
    bash "$SCRIPTS_DIR/${method}_download.sh"
    echo ""
done

echo "=== All checkpoints downloaded ==="
