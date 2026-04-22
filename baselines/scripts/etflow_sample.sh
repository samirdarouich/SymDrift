#!/bin/bash
# ETFlow: benchmark average inference speed using random weights.
# Instantiates BaseFlow from qm9-base.yaml config with random weights.

REPO="$(cd "$(dirname "$0")/../ETFlow" 2>/dev/null && pwd)" || {
    echo "[ETFlow] ERROR: ETFlow repo not found. Run clone_baselines.sh first."
    exit 1
}

conda run -n etflow python3 - "$REPO" <<'PYEOF'
import sys, os, time, torch
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from etflow.models.model import BaseFlow

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate from qm9-base.yaml model_args (random weights)
model = BaseFlow(
    network_type="TorchMDDynamics",
    hidden_channels=160,
    num_layers=20,
    num_rbf=64,
    rbf_type="expnorm",
    trainable_rbf=True,
    activation="silu",
    neighbor_embedding=True,
    cutoff_lower=0.0,
    cutoff_upper=10.0,
    max_z=100,
    node_attr_dim=10,
    edge_attr_dim=1,
    attn_activation="silu",
    num_heads=8,
    distance_influence="both",
    reduce_op="sum",
    qk_norm=True,
    output_layer_norm=True,
    clip_during_norm=True,
    so3_equivariant=True,
    sigma=0.1,
    prior_type="gaussian",
).to(device)
model.eval()

SMILES = [
    "C", "CC", "CCC", "CO", "CCO", "CN", "CCN", "C=C", "C=O", "C=N",
    "C#N", "CC#N", "C1CC1", "C1CCC1", "C1CO1", "C1CCO1", "C1CN1", "C1CCN1",
    "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "c1ccoc1", "CF", "CCF", "C(F)(F)F",
    "CC(C)O", "CC(C)=O", "CC(C)N", "CC(C)C", "CCC=O",
    "CCCC", "CCCN", "CCCO", "C=CC", "CC=C", "CC=O", "C#CC",
    "C1CCCC1", "C1CCCO1", "C1CCNC1", "Cc1ccncc1", "Cc1ccoc1",
    "CC(F)F", "CCCl", "CCBr",
    "NCC(=O)O", "CC(N)C(=O)O", "OCC(O)CO",
    "CC1CC1", "CCCCN", "CCCCO",
    "CC(O)CO", "CC(C)(C)C", "CCCCC", "CCCC=O", "c1ccc(C)cc1",
]

def smiles_to_data(smi, device):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = float(bond.GetBondTypeAsDouble())
        src += [i, j]; dst += [j, i]; eattr += [bt, bt]
    if src:
        bond_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float).unsqueeze(1)
    else:
        bond_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    # node_attr: one-hot-like, dim=10
    node_attr = torch.zeros(z.size(0), 10)
    for i, an in enumerate(z.tolist()):
        idx = min(an - 1, 9)
        node_attr[i, idx] = 1.0
    pos = torch.randn(z.size(0), 3)
    return Data(z=z, pos=pos, bond_index=bond_index,
                edge_attr=edge_attr, node_attr=node_attr)

def make_batch(smiles, device):
    data_list = [d for smi in smiles if (d := smiles_to_data(smi, device)) is not None]
    if not data_list:
        return None
    b = Batch.from_data_list(data_list)
    return (b.z.to(device), b.bond_index.to(device),
            b.batch.to(device), b.node_attr.to(device), b.edge_attr.to(device))

N_WARMUP = 5
N_BENCH = 50

print("[ETFlow] Warming up...", flush=True)
with torch.no_grad():
    z, bi, batch, na, ea = make_batch(SMILES[:N_WARMUP], device)
    model.sample(z=z, bond_index=bi, batch=batch,
                 node_attr=na, edge_attr=ea, n_timesteps=50,
                 sampler_type="ode", t_min=0.0001, t_max=0.9999)

print(f"[ETFlow] Benchmarking {N_BENCH} molecules...", flush=True)
z, bi, batch, na, ea = make_batch(SMILES[:N_BENCH], device)
if device == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    model.sample(z=z, bond_index=bi, batch=batch,
                 node_attr=na, edge_attr=ea, n_timesteps=50,
                 sampler_type="ode", t_min=0.0001, t_max=0.9999)
if device == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

n = int(batch.max().item()) + 1
print(f"[ETFlow] Avg inference speed: {elapsed/n*1000:.1f} ms/sample  (total {elapsed:.2f}s for {n} samples)")
PYEOF
