#!/bin/bash
# GeoMol: benchmark average inference speed using random weights.

REPO="$(cd "$(dirname "$0")/../GeoMol" 2>/dev/null && pwd)" || {
    echo "[GeoMol] ERROR: GeoMol repo not found."
    exit 1
}

PYTHON=$(conda info --envs 2>/dev/null | awk '$1=="geomol"{print $NF"/bin/python3"}')

if [ ! -x "$PYTHON" ]; then
    echo "[GeoMol] ERROR: conda env 'geomol' not found at $PYTHON. Run geomol_setup.sh first."
    exit 1
fi

"$PYTHON" - "$REPO" <<'PYEOF'
import sys, os, time, torch
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

import yaml
from model.model import GeoMol
from model.featurization import featurize_mol_from_smiles
from torch_geometric.data import Batch

with open("trained_models/qm9/model_parameters.yml") as f:
    model_parameters = yaml.full_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoMol(**model_parameters).to(device)
model.eval()

SMILES = [
    "C","CC","CCC","CO","CCO","CN","CCN","C=C","C=O","C=N",
    "C#N","CC#N","C1CC1","C1CCC1","C1CO1","C1CCO1","C1CN1","C1CCN1",
    "c1ccccc1","Cc1ccccc1","c1ccncc1","c1ccoc1","CF","CCF","C(F)(F)F",
    "CC(C)O","CC(C)=O","CC(C)N","CC(C)C","CCC=O",
    "CCCC","CCCN","CCCO","C=CC","CC=C","CC=O","C#CC",
    "C1CCCC1","C1CCCO1","C1CCNC1","Cc1ccncc1","Cc1ccoc1",
    "CC(F)F","CCCl","CCBr","NCC(=O)O","CC(N)C(=O)O",
    "OCC(O)CO","CC1CC1","CCCCN","CCCCO","CC(C)(C)C",
]

def make_batch(smiles, device):
    data_list = [d for smi in smiles if (d := featurize_mol_from_smiles(smi, dataset="qm9")) is not None]
    if not data_list:
        return None
    batch = Batch.from_data_list(data_list).to(device)
    # PyG 2.x collates non-tensor dict attributes in unexpected ways;
    # get_neighbor_ids expects a plain list of per-molecule dicts.
    batch.neighbors = [{k: v.to(device) for k, v in d.neighbors.items()} for d in data_list]
    return batch

print("[GeoMol] Warming up...", flush=True)
with torch.no_grad():
    b = make_batch(SMILES[:5], device)
    if b: model(b, inference=True, n_model_confs=1)

N = 50
print(f"[GeoMol] Benchmarking {N} molecules...", flush=True)
with torch.no_grad():
    b = make_batch(SMILES[:N], device)
    if device == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()
    model(b, inference=True, n_model_confs=1)
    if device == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

n = int(b.ptr.shape[0]) - 1
print(f"[GeoMol] Avg inference speed: {elapsed/n*1000:.1f} ms/sample  (total {elapsed:.2f}s for {n} samples)")
PYEOF
