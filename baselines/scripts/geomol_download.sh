#!/bin/bash
# GeoMol: no public pretrained checkpoint is available.
# This script checks whether the model exists and prints instructions if not.

REPO="$(cd "$(dirname "$0")/../GeoMol" 2>/dev/null && pwd)" || {
    echo "[GeoMol] ERROR: GeoMol repo not found."
    exit 1
}

CKPT="$REPO/trained_models/qm9/best_model.pt"
CONFIG="$REPO/trained_models/qm9/model_parameters.yml"

if [ -f "$CKPT" ] && [ -f "$CONFIG" ]; then
    echo "[GeoMol] Checkpoint found at $CKPT — nothing to download."
    exit 0
fi

echo ""
echo "[GeoMol] *** No public checkpoint available ***"
echo ""
echo "GeoMol does not distribute a pretrained QM9 model publicly."
echo "Options:"
echo ""
echo "  1. Request weights from the authors:"
echo "     https://github.com/PattanaikL/GeoMol/issues"
echo ""
echo "  2. Train from scratch (requires QM9 data):"
echo "     conda run -n geomol python $REPO/train.py \\"
echo "         --dataset qm9 --data_dir $REPO/data/QM9"
echo ""
echo "Once you have the weights, place them at:"
echo "  $CKPT"
echo "  $CONFIG"
echo ""
exit 1
