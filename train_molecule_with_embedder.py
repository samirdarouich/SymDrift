import os

import matplotlib.pyplot as plt
from ase.io import write
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import MoleculeDataset, CompositionBatchSampler
from tspath.generative import DriftingField
from tspath.model import EGNN, EquiformerV2, PaiNN
from tspath.utils import sample_noise_like, batch_inputs_to_atoms

@torch.no_grad()
def sample(model, batch, n_samples=None):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()
    batch_sampling = batch.clone()

    # Sample prior noise
    n_atoms = batch.num_atoms[0].item()
    d = batch.pos.shape[1]
    if n_samples is not None:
        batch_ = torch.arange(n_samples, device=device).repeat_interleave(n_atoms)
        dummy = torch.zeros((n_samples * n_atoms, d), dtype=torch.float, device=device)
        x = batch.x[:n_atoms].repeat(n_samples)
        num_atoms = batch.num_atoms[0].repeat(n_samples * n_atoms)
        z = sample_noise_like(dummy, batch_)
    else:
        batch_ = batch.batch
        z = sample_noise_like(batch.pos, batch_)
        x = batch.x
        num_atoms = batch.num_atoms
    batch_sampling.pos = z
    batch_sampling.batch = batch_
    batch_sampling.x = x
    batch_sampling.num_atoms = num_atoms

    # generate samples
    x = model(batch_sampling)

    if was_training:
        model.train()

    batch_sampling.pos_generated = x
    atoms_samples = batch_inputs_to_atoms(batch_sampling, "pos_generated")
    return atoms_samples


def visualize(model, batch, current_epoch, n_samples=None, outdir=None):
    epoch = current_epoch
    atoms_samples = sample(model, batch, n_samples=n_samples)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        if isinstance(epoch, int):
            epoch_str = f"{epoch:04d}"
        else:
            epoch_str = str(epoch)
        write(f"{plot_dir}/epoch_{epoch_str}.png", atoms_samples[0])
        write(f"{plot_dir}/epoch_{epoch_str}.xyz", atoms_samples)
    plt.close()

class EGNNEncoder(nn.Module):
    def __init__(self, egnn_model, latent_dim=128):
        super().__init__()
        self.egnn = egnn_model  # your EGNN backbone
        self.projector = nn.Sequential(
            nn.Linear(egnn_model.out_node_nf, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, data):

        # (num_nodes, hidden_dim)
        h, _ = self.egnn(data)

        # pool to molecule embedding (B, hidden_dim)
        h_pool = global_mean_pool(h, data.batch)

        # (B, latent_dim)
        z = self.projector(h_pool)

        # normalize for contrastive stability
        z = nn.functional.normalize(z, dim=-1)

        return z
    
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "t1x_eq_C5H8O"
dataset = MoleculeDataset(
    source=dataset_name,
    root="/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq",
    split="train",
    split_identifier="debug",
)

batch_sampler = CompositionBatchSampler(
    dataset,
    k=1,  # number of chunks per batch
    n=64,  # chunk size (number of samples per chunk)
    shuffle=True,
    drop_last=False,
    resample=True,
    seed=42,
)
dataloader = GeometricDataLoader(
    dataset, batch_sampler=batch_sampler, shuffle=False
)

model_type = "egnn"
model_dict = {
    "painn": PaiNN(),
    "egnn": EGNN(),
    "equiformerv2": EquiformerV2(
        max_radius=11.0,
        num_distance_basis=64,
        num_layers=3,
        lmax_list=[2],
    ),
}
model = model_dict[model_type]
model.to(device)

# Load in pretrained embedder
embedder = EGNNEncoder(EGNN(out_node_nf=128), latent_dim=128).to(device)
ckpt = torch.load("/home/samirdarouich/projects/TS_physics/tspath/runs/embedder/checkpoints/best_model.pt")
embedder.load_state_dict(ckpt["state_dict"])
embedder.to(device)
embedder.eval()
for param in embedder.parameters():
    param.requires_grad = False

outdir = f"runs/molecule/dataset_{dataset_name}/{model_type}/embedding"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

drifting_field = DriftingField(temperature=0.15,)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

losses = []
model.train()
n_epochs = 500
n_samples = 1000
pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y = batch.pos.clone()

        # Sample noise
        z = sample_noise_like(batch.pos, batch.batch)
        batch.pos = z

        # Call the model
        x = model(batch)
        
        #! invariant embedding
        embedding_batch = batch.clone()
        embedding_batch.pos = x
        x_embedded = embedder(embedding_batch)
        
        embedding_batch.pos = y
        y_embedded = embedder(embedding_batch)
        
        # Call the drift
        V, *_ = drifting_field(
            x_embedded,
            y_embedded,
            x_embedded,
            batch.num_atoms[0],
            atomic_numbers=batch.x,
        )

        x_drifted = (x_embedded + V).detach()

        loss = torch.nn.functional.mse_loss(x_embedded, x_drifted)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})

        if epoch % (n_epochs // 10) == 0 and batch_idx == 0 and epoch > 0:
            visualize(
                model, batch, current_epoch=epoch, n_samples=n_samples, outdir=outdir
            )
            
    # if (epoch < 101 and epoch % 5 == 0) or (epoch>100 and epoch % 20 == 0):
    #     torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/epoch_{epoch}.pt")

visualize(model, batch, current_epoch="final", n_samples=n_samples, outdir=outdir)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")
