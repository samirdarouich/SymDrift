import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from tspath.datasets import MoleculeDataset
from tspath.model import EGNN
from tspath.utils import batch_center_systems, sample_noise_like
import os

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


class MolecularAugment:
    def __init__(self, noise_std=0.05, mask_ratio=0.15):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def __call__(self, data):
        data = data.clone()

        # 1) coordinate noise
        data.pos = data.pos + self.noise_std * sample_noise_like(data.pos, data.batch)
        data.pos = batch_center_systems(data.pos, data.batch)  # re-center after noise

        # 2) node feature masking
        N = data.x.size(0)
        mask = torch.rand(N, device=data.pos.device) < self.mask_ratio
        data.x[mask] = 0

        return data


def info_nce_loss(z1, z2, temperature=0.1):
    """
    z1, z2: (B, d)
    """
    B = z1.size(0)

    logits = z1 @ z2.T  # (B, B)
    logits = logits / temperature

    labels = torch.arange(B, device=z1.device)

    return nn.CrossEntropyLoss()(logits, labels)


def train(model, loader, optimizer, temperature=0.1):
    model.train()
    total_loss = 0
    augment = MolecularAugment()

    for data in loader:
        data = data.to(device)

        data1 = augment(data)
        data2 = augment(data)

        optimizer.zero_grad()

        z1 = model(data1)
        z2 = model(data2)

        loss = info_nce_loss(z1, z2, temperature)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, temperature=0.1):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            z = model(data)

            # For validation, we can compute the loss between two random augmentations
            augment = MolecularAugment()
            data_aug = augment(data)
            z_aug = model(data_aug)

            loss = info_nce_loss(z, z_aug, temperature)
            total_loss += loss.item()

    return total_loss / len(loader)


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MoleculeDataset(
    source="t1x_eq_all_structures",
    root="/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq",
    split="train",
)

dataloader = GeometricDataLoader(
    dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42)
)


dataset_val = MoleculeDataset(
    source="t1x_eq_all_structures",
    root="/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq",
    split="val",
)

dataloader_val = GeometricDataLoader(
    dataset_val,
    batch_size=64,
    shuffle=False,
    generator=torch.Generator().manual_seed(42),
)

augment = MolecularAugment()

backbone = EGNN(out_node_nf=128)
model = EGNNEncoder(backbone, latent_dim=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

outdir = "runs/embedder"
ckpt_dir = f"{outdir}/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

n_epochs = 500
best_val_loss = float("inf")
pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
for epoch in pbar:
    train_loss = train(model, dataloader, optimizer, temperature=0.1)
    val_loss = validate(model, dataloader_val, temperature=0.1)
    pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/best_model.pt")

torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")
