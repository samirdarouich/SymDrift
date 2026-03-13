import os

import matplotlib.pyplot as plt
from ase.io import write
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import MoleculeDataset, CompositionBatchSampler
from tspath.generative import DriftingField
from tspath.model import EGNN, PaiNN, GVPModel, GaussianMomentDescriptor
from tspath.utils import sample_noise_like, batch_inputs_to_atoms
from tspath.alignment import get_rmsd_batched
from tspath.analysis import get_validity
from torch_geometric.data import Data
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from torch_geometric.nn import global_mean_pool
import json
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)

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

@torch.no_grad()
def sample(model, batch, n_samples):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()

    # Sample prior noise n_samples * B
    B = batch.batch.max().item() + 1
    batch_sampling = create_batch_object(batch, n_samples)

    # generate samples
    x = model(batch_sampling)

    # Target
    n_samples_total = B * n_samples
    n_atoms = batch.num_atoms[0].item()
    x_sample = x.view(n_samples_total, n_atoms, -1)
    # repeat each batch elements positions n_samples times to match the shape of x_sample
    x_target = batch.pos_orig.view(B, n_atoms, -1).unsqueeze(1).repeat(1, n_samples, 1, 1).view(n_samples_total, n_atoms, -1)

    rmsd = get_rmsd_batched(
        x_sample, x_target, atomic_numbers=batch_sampling.x.view(-1, n_atoms),
        align=True, permute=True, brute_force_permutations=False
    )
    
    x_embedded, _ = embedder_fn(embedder_type, x, x, batch_sampling.x, batch_sampling.edge_index, batch_sampling.batch)
    
    if was_training:
        model.train()

    batch_sampling.pos_generated = x
    atoms_noise = batch_inputs_to_atoms(batch_sampling, "pos")
    atoms_samples = batch_inputs_to_atoms(batch_sampling, "pos_generated")
    
    for i, atom in enumerate(atoms_samples):
        atom.info["rmsd"] = rmsd[i].item()
    return atoms_samples, atoms_noise, rmsd, x_embedded.cpu()

        
def visualize(model, batch, current_step, n_samples=None, outdir=None, y_embedded=None):
    step = current_step
    atoms_samples, atoms_noise, rmsd, x_embedded = sample(model, batch, n_samples=n_samples)
    metrics_generated = get_validity(atoms_samples)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        if isinstance(step, int):
            step_str = f"{step:04d}"
        else:
            step_str = str(step)
        write(f"{plot_dir}/noise.png", atoms_noise[0])
        write(f"{plot_dir}/noise.xyz", atoms_noise)
        write(f"{plot_dir}/step_{step_str}.png", atoms_samples[0])
        write(f"{plot_dir}/step_{step_str}.xyz", atoms_samples)
        print(f"Epoch {epoch}: Sample RMSD: {rmsd.mean().item():.4f}; Stable atoms: {metrics_generated['frac_stable_atoms']:.4f}; Stable mol: {metrics_generated['frac_stable_molecules']:.4f}")
    plt.close()
    if y_embedded is not None:
        
        y_embedded = torch.nn.functional.normalize(y_embedded, dim=-1)
        x_embedded = torch.nn.functional.normalize(x_embedded, dim=-1)
        threshold = torch.quantile(y_embedded, 0.1)
        cov, mat = cov_mat(x_embedded, y_embedded, threshold=threshold)

        pca = PCA(n_components=2)
        y_2d = pca.fit_transform(y_embedded)
        x_2d = pca.transform(x_embedded)
        
        plt.figure(figsize=(8, 6))
        plt.title(f"PCA variance: {sum(pca.explained_variance_ratio_):.2f}, Cov: {cov:.4f}, Mat: {mat:.4f}")
        plt.scatter(x_2d[:, 0], x_2d[:, 1], alpha=0.75, color="red", label="predictions")
        plt.scatter(y_2d[:, 0], y_2d[:, 1], alpha=0.75, color="blue", label="target")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.savefig(f"{plot_dir}/step_{step_str}_pca.png")
        plt.close()
        
        metrics_generated["cov"] = cov.item()
        metrics_generated["mat"] = mat.item()
        
    with open(f"{plot_dir}/step_{step_str}_stats.json", "w") as f:
            json.dump({
                "rmsd": {
                    "mean": rmsd.mean().item(),
                    "median": rmsd.median().item()
                },
                **metrics_generated
            }, f, indent=4)
    return atoms_samples

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

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
# dataset_name = "t1x_eq_CHN3O" #! 6 ATOMS
# dataset_name = "t1x_eq_C3H2N2O2" #! 9 ATOMS
# dataset_name = "t1x_eq_C5H8O" #! 14 ATOMS
# dataset_name = "qm9_C3H2N2O2" #! 9 ATOMS
dataset_name = "qm9_C5H4N2O2" #! 13 ATOMS
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

model_type = "painn"
model_dict = {
    "painn": PaiNN(sphere_channels=256,num_layers=9, max_radius=11.0),# PaiNN(),#PaiNN(sphere_channels=256,num_layers=6), PaiNN(sphere_channels=256,num_layers=9),
    "egnn": EGNN(sphere_channels=256, num_layers=5),
    "gvp": GVPModel(sphere_channels=256,num_layers=9, max_radius=11.0),
}
model = model_dict[model_type]
model.to(device)

print(f"Dataset: {dataset_name}, Model: {model_type}, ")

embedder_type = "gm"
if embedder_type == "gm":
    #! GM descriptor
    n_radial = 4
    n_basis = 4
    n_contr = 4 # distances # 8 distances + angle
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

only_pos_drift = True
drift_str = ""
if split_identifier is None:
    drift_str = "pos_drift" if only_pos_drift else "full_drift"
    
normalize_drift = True
temperatures = [0.05]
temp_str = "_".join([f"{t:.2f}" for t in temperatures])
outdir = f"runs/molecule/dataset_{dataset_name}/split_{split_identifier}/{model_type}/temp_{temp_str}/norm_{normalize_drift}/{embedder_str}/{drift_str}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

drifting_field = DriftingField(
    temperatures=temperatures,
    normalize_drift=normalize_drift,
)

batch_size = 64
batch_sampler = CompositionBatchSampler(
    dataset,
    k=1,  # number of chunks per batch
    n=batch_size,  # chunk size (number of samples per chunk)
    shuffle=True,
    drop_last=False,
    resample=True,
    seed=42,
)
dataloader = GeometricDataLoader(
    dataset, batch_sampler=batch_sampler, shuffle=False
)

losses = []
model.train()

n_steps = 75000
n_epochs = n_steps // len(dataloader)

scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
step_count = 0
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y = batch.pos.clone()
        batch.pos_orig = y.clone()
        
        # Sample bz negative samples
        batch = batch.clone()
        z = sample_noise_like(batch.pos, batch.batch)
        batch.pos = z

        # Call the model
        x = model(batch)
        
        if x.isnan().any():
            raise ValueError("NaN detected in model output.")
        
        # Encode samples and targets
        x_embedded, y_embedded = embedder_fn(
            embedder_type, x, y, batch.x, batch.edge_index, batch.batch
        )
        
        if x_embedded.isnan().any():
            raise ValueError("NaN detected in embedded model output.")
        
        
        # Call the drift
        V, drift_pos, drift_neg, *_ = drifting_field(
            x_embedded.detach(),
            y_embedded,
            x_embedded.detach(),
        )
        
        if split_identifier == "debug" or only_pos_drift:
            v_pos = drift_pos[0]
            v_norm = torch.sqrt(torch.mean(v_pos**2))
            v_pos = v_pos / (v_norm + 1e-8)
            x_drifted = (x_embedded + v_pos).detach()
        else:
            x_drifted = (x_embedded + V).detach()
        loss = torch.nn.functional.mse_loss(x_embedded, x_drifted)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if embedder_type == "gm":
            torch.nn.utils.clip_grad_norm_(gm_descriptor.parameters(), max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(
            {
                "step": step_count, 
                "loss": loss.item(),
                "mse(V)": torch.sqrt(torch.mean(V**2)).item(),
                "mse(pos_drift)": torch.sum(torch.sqrt(torch.mean(drift_pos**2, dim=(1,2)))).item(), 
                "mse(neg_drift)": torch.sum(torch.sqrt(torch.mean(drift_neg**2, dim=(1,2)))).item(),
            }
        )

        if step_count % (n_steps // 10) == 0 and step_count > 0:
            # Sample in total 1000 samples. This will produce per batch item n_samples, 
            # which will result in 1000 samples overall
            n_samples = 1000 // batch.num_graphs 
            visualize(
                model, batch, current_step=step_count, n_samples=n_samples, outdir=outdir, y_embedded=y_embedded.cpu()
            )
        step_count += 1

    # Update learning rate scheduler at the end of each epoch
    scheduler.step()

n_samples = 1000 // batch.num_graphs
atoms_samples = visualize(model, batch, current_step="final", n_samples=n_samples, outdir=outdir, y_embedded=y_embedded.cpu())
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")

data_atoms = []
batch_sampler = CompositionBatchSampler(
    dataset,
    k=1,
    n=batch_size,
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
    
    
    
    
