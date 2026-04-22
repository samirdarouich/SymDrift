#!/bin/bash
# Install conda environments for all baselines
set -e

BASELINES_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$BASELINES_DIR/scripts"

echo "=== Setting up all baseline environments ==="
echo ""

for method in mcf tordiff etflow; do
    echo "--- Setting up: $method ---"
    bash "$SCRIPTS_DIR/${method}_setup.sh"
    echo ""
done

echo "=== All environments set up ==="
