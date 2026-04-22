#!/bin/bash
# Download Torsional Diffusion QM9 checkpoint and test data from Google Drive.
# Drive folder: https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7
set -e

REPO="$(cd "$(dirname "$0")/../torsional-diffusion" 2>/dev/null && pwd)" || {
    echo "[TorsionalDiff] ERROR: torsional-diffusion repo not found."
    exit 1
}

CKPT="$REPO/workdir/qm9_default/best_model.pt"
if [ -f "$CKPT" ]; then
    echo "[TorsionalDiff] Checkpoint already exists at $CKPT — skipping."
    exit 0
fi

command -v gdown >/dev/null 2>&1 || {
    echo "[TorsionalDiff] gdown not found. Installing..."
    pip install -q gdown
}

echo "[TorsionalDiff] Downloading workdir (checkpoint) from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7" \
    -O "$REPO" --remaining-ok

echo "[TorsionalDiff] Checkpoint downloaded. Verifying..."
if [ -f "$CKPT" ]; then
    echo "[TorsionalDiff] Checkpoint found at $CKPT"
else
    echo "[TorsionalDiff] WARNING: Expected checkpoint at $CKPT not found."
    echo "  Check the downloaded folder structure in $REPO/"
fi
