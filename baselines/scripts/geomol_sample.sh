#!/bin/bash
# GeoMol: sample 50 QM9 test molecules and report wall-clock timing.

REPO="$(cd "$(dirname "$0")/../GeoMol" 2>/dev/null && pwd)" || {
    echo "[GeoMol] ERROR: GeoMol repo not found."
    exit 1
}

CKPT="$REPO/trained_models/qm9/best_model.pt"
if [ ! -f "$CKPT" ]; then
    echo "[GeoMol] ERROR: Checkpoint not found at $CKPT. See geomol_download.sh for instructions."
    exit 1
fi

TEST_CSV_FULL="$REPO/data/QM9/test_smiles.csv"
if [ ! -f "$TEST_CSV_FULL" ]; then
    echo "[GeoMol] ERROR: Test CSV not found at $TEST_CSV_FULL"
    exit 1
fi

# Create a 50-molecule subset (header + 50 rows)
TMPDIR_LOCAL=$(mktemp -d)
TEST_CSV_50="$TMPDIR_LOCAL/test_50.csv"
{ head -1 "$TEST_CSV_FULL"; tail -n +2 "$TEST_CSV_FULL" | head -50; } > "$TEST_CSV_50"

N_SAMPLES=50
echo "[GeoMol] Sampling $N_SAMPLES molecules..."

conda run -n geomol python3 - <<PYEOF
import subprocess, time, sys, os
os.chdir("$REPO")
cmd = [
    "python", "generate_confs.py",
    "--trained_model_dir", "trained_models/qm9/",
    "--test_csv", "$TEST_CSV_50",
    "--dataset", "qm9",
    "--out", "$TMPDIR_LOCAL/geomol_out.pkl",
]
start = time.perf_counter()
result = subprocess.run(cmd)
elapsed = time.perf_counter() - start
n = $N_SAMPLES
print(f"[GeoMol] Total: {elapsed:.2f}s | Avg/sample: {elapsed/n*1000:.1f}ms")
sys.exit(result.returncode)
PYEOF

rm -rf "$TMPDIR_LOCAL"
