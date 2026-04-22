#!/bin/bash
# Download ETFlow QM9 checkpoint from Zenodo (record 14226681).
set -e

REPO="$(cd "$(dirname "$0")/../ETFlow" 2>/dev/null && pwd)" || {
    echo "[ETFlow] ERROR: ETFlow repo not found."
    exit 1
}

CKPT_DIR="$REPO/ckpts"
CKPT_FILE="$CKPT_DIR/qm9_o3.ckpt"
mkdir -p "$CKPT_DIR"

if [ -f "$CKPT_FILE" ]; then
    echo "[ETFlow] Checkpoint already exists at $CKPT_FILE — skipping."
    exit 0
fi

echo "[ETFlow] Downloading QM9 checkpoint from Zenodo (record 14226681)..."
# The Zenodo record lists the file as qm9-o3 checkpoint
wget -q --show-progress \
    "https://zenodo.org/records/14226681/files/qm9_o3.ckpt" \
    -O "$CKPT_FILE" || {
    echo "[ETFlow] Direct wget failed. Trying via Python etflow API (auto-download)..."
    conda run -n etflow python3 -c "
from etflow import BaseFlow
model = BaseFlow.from_default(model='qm9-o3')
print('[ETFlow] Checkpoint auto-downloaded and cached by etflow.')
"
    exit 0
}

echo "[ETFlow] Checkpoint saved to $CKPT_FILE"
