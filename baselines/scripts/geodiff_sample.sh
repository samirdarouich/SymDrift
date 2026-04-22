#!/bin/bash
# GeoDiff: benchmark average inference speed using random weights.
# Creates a synthetic QM9-like dataset and times langevin_dynamics_sample().

REPO="$(cd "$(dirname "$0")/../GeoDiff" 2>/dev/null && pwd)" || {
    echo "[GeoDiff] ERROR: GeoDiff repo not found. Run clone_baselines.sh first."
    exit 1
}

conda run -n geodiff python3 - "$REPO" <<'PYEOF'
import sys, os, time, tempfile, torch
import numpy as np
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from easydict import EasyDict
import yaml
from models.epsnet import get_model

# Load QM9 config from repo
with open("configs/qm9_default.yml") as f:
    config = EasyDict(yaml.safe_load(f))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(config.model).to(device)
model.eval()

# QM9 SMILES (≤9 heavy atoms, CHNOF only)
SMILES = [
    "C", "CC", "CCC", "CO", "CCO", "CN", "CCN", "C=C", "C=O", "C=N",
    "C#N", "CC#N", "C1CC1", "C1CCC1", "C1CO1", "C1CCO1", "C1CN1", "C1CCN1",
    "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "c1ccoc1", "CF", "CCF", "C(F)(F)F",
    "CC(C)O", "CC(C)=O", "CC(C)N", "CC(C)C", "CCC=O",
    "CCCC", "CCCN", "CCCO", "C=CC", "CC=C", "CC=O", "C#CC",
    "C1CCCC1", "C1CCCO1", "C1CCNC1", "Cc1ccncc1", "Cc1ccoc1",
    "CC(F)F", "CCCl", "CCBr", "CCI",
    "NCC(=O)O", "CC(N)C(=O)O", "OCC(O)CO", "CC(O)C(=O)O",
    "CC1CC1", "C1CC1C", "CC(C)(C)C",
]

from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_batch(smi_list, device):
    """Create a minimal PyG-compatible batch directly from SMILES."""
    from torch_geometric.data import Data, Batch

    data_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        # Atom type: atomic number (truncated to vocab used by GeoDiff)
        atom_type = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
        # Build bond edges
        src, dst, etype = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = int(bond.GetBondTypeAsDouble())  # 1,2,3
            src += [i, j]; dst += [j, i]; etype += [bt, bt]
        if not src:  # single atom
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_type = torch.tensor(etype, dtype=torch.long)
        data = Data(atom_type=atom_type, edge_index=edge_index, edge_type=edge_type,
                    num_nodes=n)
        data.num_graphs = 1
        data_list.append(data)
    return Batch.from_data_list(data_list).to(device)

N_WARMUP = 5
N_BENCH = 50

print("[GeoDiff] Warming up...", flush=True)
with torch.no_grad():
    batch = smiles_to_batch(SMILES[:N_WARMUP], device)
    pos_init = torch.randn(batch.num_nodes, 3, device=device)
    model.langevin_dynamics_sample(
        atom_type=batch.atom_type, pos_init=pos_init,
        bond_index=batch.edge_index, bond_type=batch.edge_type,
        batch=batch.batch, num_graphs=batch.num_graphs,
        extend_order=False, n_steps=5000, step_lr=1e-6,
        w_global=0.3, global_start_sigma=0.5,
    )

print(f"[GeoDiff] Benchmarking {N_BENCH} molecules...", flush=True)
with torch.no_grad():
    batch = smiles_to_batch(SMILES[:N_BENCH], device)
    pos_init = torch.randn(batch.num_nodes, 3, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.langevin_dynamics_sample(
        atom_type=batch.atom_type, pos_init=pos_init,
        bond_index=batch.edge_index, bond_type=batch.edge_type,
        batch=batch.batch, num_graphs=batch.num_graphs,
        extend_order=False, n_steps=5000, step_lr=1e-6,
        w_global=0.3, global_start_sigma=0.5,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

print(f"[GeoDiff] Avg inference speed: {elapsed/N_BENCH*1000:.1f} ms/sample  (total {elapsed:.2f}s for {N_BENCH} samples)")
PYEOF
