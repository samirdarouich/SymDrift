from tspath.datasets import ReactionDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
import torch
from tspath.model import GotenNet
import numpy as np
import logging
from tspath.utils import sample_noise_like

logging.basicConfig(level=logging.INFO)

orig_rdb7_data = np.load("/home/samirdarouich/projects/TS/goflow_lean/data/RDB7/processed_data/data.pkl", allow_pickle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = "rdb7"
split_identifier = "random"
dataset = ReactionDataset(
    source=dataset_name,
    root=f"/home/samirdarouich/projects/TS_physics/tspath/data/{dataset_name}",
)

# Sanity check the data processing (only differnce should be the edge types and atom features)
for i in range(len(orig_rdb7_data)):
    data = dataset[i]
    orig_data = orig_rdb7_data[i]

    rxn = data.rxn.item()
    rxn_orig = orig_data.rxn_index
    assert rxn == rxn_orig, f"Reaction index mismatch at index {i}: {rxn} vs {rxn_orig}"
    assert (data.x == orig_data.atom_type).all(), f"Atom numbers mismatch at index {i}"
    ts_pos = orig_data.pos - orig_data.pos.mean(axis=0)
    assert torch.allclose(data.pos_ts, ts_pos, atol=1e-5), f"Positions mismatch at index {i}"
    assert (orig_data.edge_index == data.bonded_edge_index).all(), f"Bonded edge index mismatch at index {i}"
    
logging.info("Data processing sanity check passed for all samples!")

dataloader = GeometricDataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
)

model = GotenNet()
model.to(device)
model.train()

for batch_idx, batch in enumerate(dataloader):
    batch = batch.to(device)
    prior = sample_noise_like(batch.pos_ts, batch.batch)
    batch.pos = prior
    x = model(batch)
    break
logging.info(f"Model forward pass successful, output shape: {x.shape}")