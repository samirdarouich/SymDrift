#!/bin/bash
# GeoMol: benchmark average inference speed using random weights.
# Uses the existing model_parameters.yml from the repo.

REPO="$(cd "$(dirname "$0")/../GeoMol" 2>/dev/null && pwd)" || {
    echo "[GeoMol] ERROR: GeoMol repo not found. Run clone_baselines.sh first."
    exit 1
}

conda run -n geomol python3 - "$REPO" <<'PYEOF'
import sys, os, time, tempfile, torch
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

import yaml
from model.model import GeoMol
from model.featurization import featurize_mol_from_smiles
from model.inference import construct_conformers
from torch_geometric.data import Batch

# Load config (already in repo, no download needed)
with open("trained_models/qm9/model_parameters.yml") as f:
    model_parameters = yaml.full_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoMol(**model_parameters).to(device)  # random weights
model.eval()

SMILES = [
    "C", "CC", "CCC", "CO", "CCO", "CN", "CCN", "C=C", "C=O", "C=N",
    "C#N", "CC#N", "C1CC1", "C1CCC1", "C1CO1", "C1CCO1", "C1CN1", "C1CCN1",
    "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "c1ccoc1", "CF", "CCF", "C(F)(F)F",
    "CC(C)O", "CC(C)=O", "CC(C)N", "CC(C)C", "CCC=O",
    "CCCC", "CCCN", "CCCO", "C=CC", "CC=C", "CC=O", "C#CC",
    "C1CCCC1", "C1CCCO1", "C1CCNC1", "Cc1ccncc1", "Cc1ccoc1",
    "CC(F)F", "CCCl", "CCBr", "CCI",
    "NCC(=O)O", "CC(N)C(=O)O", "OCC(O)CO",
    "CC1CC1", "C1CC1C", "CC(C)(C)C", "CCCCN", "CCCCO",
    "CC(O)CO", "C1CCCCC1", "CC(C)CC",
]

def make_batch(smiles, device):
    data_list = []
    for smi in smiles:
        d = featurize_mol_from_smiles(smi, dataset="qm9")
        if d is not None:
            data_list.append(d)
    if not data_list:
        return None
    return Batch.from_data_list(data_list).to(device)

N_WARMUP = 5
N_BENCH = 50

print("[GeoMol] Warming up...", flush=True)
with torch.no_grad():
    batch = make_batch(SMILES[:N_WARMUP], device)
    if batch is not None:
        model(batch, inference=True, n_model_confs=1)

print(f"[GeoMol] Benchmarking {N_BENCH} molecules...", flush=True)
with torch.no_grad():
    batch = make_batch(SMILES[:N_BENCH], device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    model(batch, inference=True, n_model_confs=1)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

n = len(batch.ptr) - 1  # actual number of molecules processed
print(f"[GeoMol] Avg inference speed: {elapsed/n*1000:.1f} ms/sample  (total {elapsed:.2f}s for {n} samples)")
PYEOF
