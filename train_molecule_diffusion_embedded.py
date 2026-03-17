import math
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import MoleculeDataset, CompositionBatchSampler
from tspath.model import PaiNN, EGNN, GaussianMomentDescriptor
from tspath.utils import sample_noise_like, batch_inputs_to_atoms
from tspath.alignment import get_rmsd_batched
from tspath.analysis import get_validity
from torch_geometric.nn import global_mean_pool
import json
import numpy as np
from ase.io import write
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def create_batch_object(batch, n_samples):
    
    batch_sampling = Data()
    # Sample prior noise
    n_atoms = batch.num_atoms[0].item()
    d = batch.pos.shape[1]

    # Sample each sample in the batch n_samples times
    B = batch.batch.max().item() + 1
    total_samples = B * n_samples
    
    batch_ = torch.arange(total_samples, device=device).repeat_interleave(n_atoms)
    dummy = torch.zeros((total_samples * n_atoms, d), dtype=torch.float, device=device)
    num_atoms = batch.num_atoms[0].repeat(total_samples)

    orig_pos = batch.pos
    orig_pos.view(B,n_atoms, d)
    orig_pos_expanded = orig_pos.unsqueeze(1).repeat(1, n_samples, 1, 1)  # (B, n_samples, n_atoms, d)
    orig_pos_expanded = orig_pos_expanded.view(total_samples * n_atoms, d)
    batch_sampling.pos_gt = orig_pos_expanded
    
    # repeat the atomic numbers for each sample
    x_reshaped = batch.x.view(B, n_atoms)          # (B, n_atoms)
    x_expanded = x_reshaped.unsqueeze(1).repeat(1, n_samples, 1)  # (B, n_samples, n_atoms)
    x_expanded = x_expanded.reshape(B * n_samples * n_atoms)

    z = sample_noise_like(dummy, batch_)
    
    batch_sampling.pos = z
    batch_sampling.batch = batch_
    batch_sampling.x = x_expanded
    batch_sampling.num_atoms = num_atoms
    return batch_sampling

def cov_mat(gen_emb, ref_emb, threshold):
    """
    Returns coverage and matching metrics.
    """
    D = torch.cdist(ref_emb, gen_emb)
    # for each reference point, find the closest generated point
    min_dist = D.min(dim=1).values
    # if the closest generated point is within the threshold, it's covered
    cov = (min_dist < threshold).float().mean()
    
    # matching metric: average distance to closest generated point (lower is better)
    mat = min_dist.mean()

    return cov, mat

def invariant_distance_embedder(xs, ys, atomic_numbers):
    """
    xs, ys: (B, N, 3)
    atomic_numbers: (B, N)

    returns:
        scalar loss
    """

    B, N, _ = xs.shape
    
    
    # Pairwise distance matrices
    Dx = torch.cdist(xs, xs)  # (B,N,N)
    Dy = torch.cdist(ys, ys)  # (B,N,N)

    Z = atomic_numbers

    unique_types = torch.unique(Z)

    dxs = []
    dys = []
    for Zi in unique_types:
        for Zj in unique_types:

            mask_i = (Z == Zi)[:, :, None]  # (B,N,1)
            mask_j = (Z == Zj)[:, None, :]  # (B,1,N)

            pair_mask = mask_i & mask_j     # (B,N,N)
            # n_interactions = pair_mask.sum().item() // B

            # i, j = torch.triu_indices(n_interactions, n_interactions, offset=min(1, n_interactions - 1), device=device)

            # # flatten pair distances
            # dx = Dx[pair_mask].view(B,n_interactions,n_interactions)[:,i,j]
            # dy = Dy[pair_mask].view(B,n_interactions,n_interactions)[:,i,j]
            dx = Dx[pair_mask].view(B, -1)
            dy = Dy[pair_mask].view(B, -1)

            dx = torch.sort(dx, dim=1)[0]
            dy = torch.sort(dy, dim=1)[0]

            dxs.append(dx)
            dys.append(dy)

    dxs = torch.cat(dxs, dim=1)
    dys = torch.cat(dys, dim=1)

    return dxs, dys

def edm_sampler(
    net, data, num_steps=32, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, dtype=torch.float32, seed=None
):  
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    B = data.batch.max().item() + 1
    
    # Guided denoiser.
    def denoise(x, t):
        data.pos = x
        data.t = t.view(-1,1).repeat(B,1)[data.batch]
        Dx = net(data).to(dtype)
        return Dx
    
    # Get initial noise sample
    noise = sample_noise_like(data.pos, data.batch)
    data.noise = noise
    
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * sample_noise_like(x_cur, data.batch)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def visualize_samples(model, dataloader, outdir, epoch):
    # sample one batch
    batch = next(iter(dataloader)).to(device)
    data = create_batch_object(batch, n_samples=1000//batch.num_graphs)
    gt = data.pos_gt.clone()
    with torch.no_grad():
        prediction = edm_sampler(model, data, seed=42)
    
    B = data.batch.max().item() + 1
    # Determine whether to use brute-force permutations
    unique_types = torch.unique(data.x.view(B, -1)[0], dim=0, return_counts=True)
    no_permutations = B
    for t, count in zip(*unique_types):
        no_permutations *= math.factorial(count.item())

    if no_permutations < 1e5:
        use_brute_force_permutations = True
    else:
        use_brute_force_permutations = False

    rmsd = get_rmsd_batched(
        gt.view(B, -1, 3), prediction.view(B, -1, 3), 
        atomic_numbers=data.x.view(B, -1), 
        align=True, permute=True, brute_force_permutations=use_brute_force_permutations
    )
    data.predicted_pos = prediction
    atoms = batch_inputs_to_atoms(data, "predicted_pos")
    metrics_generated = get_validity(atoms)
    for i, atom in enumerate(atoms):
        atom.info["rmsd"] = rmsd[i].item()
    atoms_noise = batch_inputs_to_atoms(data, "noise")
    write(f"{outdir}/noise.png", atoms_noise[0])
    write(f"{outdir}/epoch_{epoch}.png", atoms[0])
    write(f"{outdir}/epoch_{epoch}.xyz", atoms)
    
    
    prediction_embed, gt_embed = embedder_fn(embedder_type, prediction, gt, data.x, data.edge_index, data.batch)
    
    gt_embed = torch.nn.functional.normalize(gt_embed, dim=-1)
    prediction_embed = torch.nn.functional.normalize(prediction_embed, dim=-1)
    threshold = torch.quantile(gt_embed, 0.1)
    cov, mat = cov_mat(prediction_embed, gt_embed, threshold=threshold)

    pca = PCA(n_components=2)
    y_2d = pca.fit_transform(gt_embed)
    x_2d = pca.transform(prediction_embed)

    plt.figure(figsize=(8, 6))
    plt.title(f"PCA variance: {sum(pca.explained_variance_ratio_):.2f}, Cov: {cov:.4f}, Mat: {mat:.4f}")
    plt.scatter(x_2d[:, 0], x_2d[:, 1], alpha=0.75, color="red", label="predictions")
    plt.scatter(y_2d[:, 0], y_2d[:, 1], alpha=0.75, color="blue", label="target")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.savefig(f"{plot_dir}/epoch_{epoch}_pca.png")
    plt.close()
    
    with open(f"{plot_dir}/epoch_{epoch}_stats.json", "w") as f:
        json.dump({
            "rmsd": {
                "mean": rmsd.mean().item(),
                "median": rmsd.median().item()
            },
            **metrics_generated
        }, f, indent=4)
    print(f"Epoch {epoch}: Sample RMSD: {rmsd.mean().item():.4f}; Stable atoms: {metrics_generated['frac_stable_atoms']:.4f}; Stable mol: {metrics_generated['frac_stable_molecules']:.4f}")
    return atoms
    
        
def sample_sigma(batch_size, device, P_mean=-1.2, P_std=1.2):
    rnd = torch.randn(batch_size, device=device)
    return torch.exp(P_mean + P_std * rnd)


class EDMDiffusion(nn.Module):
    def __init__(self, model, sigma_data=0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(self, data):
        """
        data.pos : noisy coordinates
        """

        pos = data.pos
        sigma_nodes = data.t

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma_nodes ** 2 + self.sigma_data ** 2)
        c_out = sigma_nodes * self.sigma_data / (sigma_nodes ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma_nodes ** 2).sqrt()
        c_noise = sigma_nodes.flatten().log() / 4
        
        # Run the model.
        data.t = c_noise
        data.pos = (c_in * pos)
        F_x = self.model(data)
        D_x = c_skip * pos + c_out * F_x.to(torch.float32)

        return D_x

def embedder_fn(embedder_type, x, y, atomic_numbers, edge_index, batch_idx):
    if embedder_type == "gm":
        x_embed = gm_descriptor(x, edge_index, atomic_numbers)
        x_embed = global_mean_pool(x_embed, batch_idx)
        with torch.no_grad():
            y_embed = gm_descriptor(y, edge_index, atomic_numbers).detach()
            y_embed = global_mean_pool(y_embed, batch_idx)
    elif embedder_type == "distance":
        B = batch_idx.max().item() + 1
        x_ = x.view(B, -1, 3)
        y_ = y.view(B, -1, 3)
        atomic_numbers_ = atomic_numbers.view(B, -1)
        x_embed, y_embed = invariant_distance_embedder(x_, y_, atomic_numbers_)
    return x_embed, y_embed

def edm_loss(model, embedder_type, data):
    B = data.num_graphs

    x0 = data.pos.clone()

    sigma = sample_sigma(B, x0.device)

    noise = sample_noise_like(x0, data.batch)
    sigma_nodes = sigma[data.batch][:, None]

    x_noisy = x0 + sigma_nodes * noise

    data.pos = x_noisy
    data.t = sigma[data.batch][:, None]

    x_pred = model(data)

    weight = (sigma.view(-1,1)**2 + model.sigma_data**2) / (
        (sigma.view(-1,1) * model.sigma_data) ** 2
    )

    # Embed before computing loss
    x_pred_embed, x0_embed = embedder_fn(embedder_type, x_pred, x0, data.x, data.edge_index, data.batch)
    loss = weight * (x_pred_embed - x0_embed) ** 2

    return loss.mean()


def train(model, embedder_type, loader, optimizer):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()

        loss = edm_loss(model, embedder_type, data)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
        if embedder_type == "gm":
            torch.nn.utils.clip_grad_norm_(gm_descriptor.parameters(), max_norm=1.0)
    
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

    
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
# dataset_name = "t1x_eq_CHN3O" #! 6 ATOMS
# dataset_name = "t1x_eq_C3H2N2O2" #! 9 ATOMS
dataset_name = "t1x_eq_C5H8O" #! 14 ATOMS
# dataset_name = "qm9_C3H2N2O2" #! 9 ATOMS
# dataset_name = "qm9_C5H4N2O2" #! 13 ATOMS
data_folder = "qm9" if dataset_name.startswith("qm9") else "transition1x_eq"

split_identifier = None
# split_identifier = "debug"

dataset = MoleculeDataset(
    source=dataset_name,
    root=f"/home/samirdarouich/projects/TS_physics/tspath/data/{data_folder}",
    split="train",
    identifier="identifier" if dataset_name.startswith("qm9") else "rxn",
    split_identifier=split_identifier,
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

model_type = "painn"
model_dict = {
    "painn": PaiNN(sphere_channels=256,num_layers=9, max_radius=11.0),
    "egnn": EGNN(sphere_channels=256, num_layers=5),
}
backbone = model_dict[model_type]
model = EDMDiffusion(backbone, sigma_data=0.5).to(device)

embedder_type = "distance"
if embedder_type == "gm":
    #! GM descriptor
    n_radial = 4
    n_basis = 4
    n_contr = 4 # 4 distances # 8 distances + angle
    gm_descriptor = GaussianMomentDescriptor(
        n_radial=n_radial, n_basis=n_basis, max_radius=6.0, n_contr=n_contr, use_atom_type_embeddings=True
    )
    gm_descriptor.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 5e-5},
            {"params": gm_descriptor.parameters(), "lr": 1e-4},
        ],
        weight_decay=0.0
    )
    embedder_str = f"embedder_gm/n_radial_{n_radial}_n_basis_{n_basis}_n_contr_{n_contr}"
elif embedder_type == "distance":
    #! Invariant distance embedder
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
    embedder_str = "embedder_distance"

outdir = f"runs/molecule/dataset_{dataset_name}/split_{split_identifier}/{model_type}/diffuson_baseline/{embedder_str}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

n_steps = 500_000
n_epochs = n_steps // len(dataloader)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
for epoch in pbar:
    train_loss = train(model, embedder_type, dataloader, optimizer)
    pbar.set_postfix({"loss": train_loss})
    
    if epoch % (n_epochs//10) == 0 and epoch > 0:
        visualize_samples(model, dataloader, plot_dir, epoch)
        
    scheduler.step()

atoms_samples = visualize_samples(model, dataloader, plot_dir, "final")
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")


data_atoms = []
batch_sampler = CompositionBatchSampler(
    dataset,
    k=1,
    n=64,
    shuffle=False,
    drop_last=False,
    resample=False,
    seed=42,
)
dataloader = GeometricDataLoader(dataset, batch_sampler=batch_sampler)
for batch in dataloader:
    atoms = batch_inputs_to_atoms(batch, "pos")
    data_atoms.extend(atoms)
metrics_dataset = get_validity(data_atoms)
metrics_generated = get_validity(atoms_samples)
for metric_name in metrics_dataset.keys():
    print(f"{metric_name}: Dataset: {metrics_dataset[metric_name]:.4f}, Generated: {metrics_generated[metric_name]:.4f}")
with open(f"{plot_dir}/metrics.json", "w") as f:
    json.dump({"dataset": metrics_dataset, "generated": metrics_generated}, f, indent=4)
