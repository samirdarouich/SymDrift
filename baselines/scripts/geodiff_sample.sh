#!/bin/bash
# GeoDiff: sample 50 QM9 test molecules and report wall-clock timing.
# Uses --start_idx 800 --end_idx 850 (50 molecules from QM9 test split).

REPO="$(cd "$(dirname "$0")/../GeoDiff" 2>/dev/null && pwd)" || {
    echo "[GeoDiff] ERROR: GeoDiff repo not found."
    exit 1
}

CKPT="$REPO/logs/qm9_default/checkpoints/1000000.pt"
if [ ! -f "$CKPT" ]; then
    echo "[GeoDiff] ERROR: Checkpoint not found at $CKPT. Run geodiff_download.sh first."
    exit 1
fi

N_SAMPLES=50

echo "[GeoDiff] Sampling $N_SAMPLES molecules (start=800, end=850)..."

conda run -n geodiff python3 - <<PYEOF
import subprocess, time, sys, os
os.chdir("$REPO")
cmd = [
    "python", "test.py",
    "$CKPT",
    "--start_idx", "800",
    "--end_idx", "850",
    "--w_global", "0.3",
]
start = time.perf_counter()
result = subprocess.run(cmd)
elapsed = time.perf_counter() - start
n = $N_SAMPLES
print(f"[GeoDiff] Total: {elapsed:.2f}s | Avg/sample: {elapsed/n*1000:.1f}ms")
sys.exit(result.returncode)
PYEOF
