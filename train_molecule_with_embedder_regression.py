import os

import matplotlib.pyplot as plt
from ase.io import write
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import MoleculeDataset, CompositionBatchSampler
from tspath.generative import DriftingField
from tspath.model import EGNN, PaiNN, GaussianMomentDescriptor
from tspath.utils import sample_noise_like, batch_inputs_to_atoms
from tspath.analysis import get_validity
from torch_geometric.data import Data
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from torch_geometric.nn import global_mean_pool
import json

logging.basicConfig(level=logging.INFO)

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
def sample(model, batch, n_samples, gm_descriptor, mini_batch_size=64):
    """Generate samples by integrating the learned flow field in mini-batches."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()

    B = batch.batch.max().item() + 1
    n_samples_total = B * n_samples

    all_x = []
    all_z = []
    all_x_atomic = []
    all_losses = []
    samples_remaining = n_samples
    while samples_remaining > 0:
        mb = min(samples_remaining, mini_batch_size)
        batch_mb = create_batch_object(batch, mb)
        x_mb = model(batch_mb)[-1]  # final layer prediction for sampling

        # Compute GM embedding loss for this mini-batch
        x_feat = gm_descriptor(x_mb, batch_mb.edge_index, batch_mb.x, return_all_layers=False)
        y = batch.pos_orig.clone()
        y_feat = gm_descriptor(y, batch.edge_index, batch.x, return_all_layers=False)

        xl_pooled = global_mean_pool(x_feat, batch_mb.batch)          # (B*mb, features)
        yl_pooled = global_mean_pool(y_feat, batch.batch)              # (B, features)
        yl_pooled = yl_pooled.unsqueeze(1).repeat(1, mb, 1).view(B * mb, -1)
        per_sample_loss = (xl_pooled - yl_pooled).pow(2).mean(dim=-1)
        all_losses.append(per_sample_loss.cpu())

        all_x.append(x_mb.cpu())
        all_z.append(batch_mb.pos.cpu())
        all_x_atomic.append(batch_mb.x.cpu())
        samples_remaining -= mb

    x = torch.cat(all_x, dim=0)
    z = torch.cat(all_z, dim=0)
    x_atomic = torch.cat(all_x_atomic, dim=0)
    losses = torch.cat(all_losses, dim=0)

    if was_training:
        model.train()

    # Rebuild a single batch_sampling for atom conversion (all on CPU)
    batch_sampling = create_batch_object(batch, n_samples)
    batch_sampling = batch_sampling.cpu()
    batch_sampling.pos = z
    batch_sampling.x = x_atomic
    batch_sampling.pos_generated = x
    atoms_noise = batch_inputs_to_atoms(batch_sampling, "pos")
    atoms_samples = batch_inputs_to_atoms(batch_sampling, "pos_generated")

    for i, atom in enumerate(atoms_samples):
        atom.info["loss"] = losses[i].item()

    print(f"GM loss of generated samples vs target: {losses.mean().item():.6f}")
    return atoms_samples, atoms_noise, losses

        
def visualize(model, batch, current_step, n_samples=None, outdir=None, gm_descriptor=None):
    step = current_step
    atoms_samples, atoms_noise, losses = sample(model, batch, n_samples=n_samples, gm_descriptor=gm_descriptor)
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
        atoms_gt = batch_inputs_to_atoms(batch.cpu(), "pos_orig")
        write(f"{plot_dir}/groundtruth.xyz", atoms_gt)
        with open(f"{plot_dir}/step_{step_str}_loss.txt", "w") as f:
            f.write(f"GM loss: mean: {losses.mean().item():.6f}, median: {losses.median().item():.6f}\n")
    plt.close()
    return atoms_samples

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
# dataset_name = "t1x_eq_CHN3O" #! 6 ATOMS
# dataset_name = "t1x_eq_C3H2N2O2" #! 9 ATOMS
dataset_name = "t1x_eq_C5H8O" #! 14 ATOMS
split_identifier = None
# split_identifier = "debug"

dataset = MoleculeDataset(
    source=dataset_name,
    root="/home/vinhtong/tspath/data/transition1x_eq",
    split="train",
    split_identifier=split_identifier,
)

model_type = "painn"
model_dict = {
    "painn": PaiNN(sphere_channels=128, num_layers=16, max_radius=11.0),# PaiNN(),#PaiNN(sphere_channels=256,num_layers=6), PaiNN(sphere_channels=256,num_layers=9),
}
model = model_dict[model_type]
model.to(device)

print(f"Dataset: {dataset_name}, Model: {model_type}, ")

#! GM descriptor
n_radial = 5
n_basis = 4
n_contr = 8
gm_descriptor = GaussianMomentDescriptor(
    n_radial=n_radial, n_basis=n_basis, max_radius=11.0, n_contr=n_contr, use_atom_type_embeddings=True, trainable=False
)
gm_descriptor.to(device)
gm_descriptor.eval()
gm_descriptor.requires_grad_(False)

normalize_drift = True
temperatures = [0.15]
temp_str = "_".join([f"{t:.2f}" for t in temperatures])
outdir = f"runs/molecule/dataset_{dataset_name}/split_{split_identifier}/{model_type}/temp_{temp_str}/norm_{normalize_drift}/embedder_gm/n_radial_{n_radial}_n_basis_{n_basis}_n_contr_{n_contr}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

drifting_field = DriftingField(
    temperatures=temperatures,
    normalize_drift=normalize_drift,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)


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

n_steps = 200000
n_epochs = n_steps // len(dataloader)

scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

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

        # Call the model — returns one prediction per PaiNN layer
        predictions = model(batch)

        # Target GM descriptor (computed once, shared across all layers)
        y_layers = gm_descriptor(y, batch.edge_index, batch.x, return_all_layers=False)

        # Sum GM loss over all PaiNN layer predictions
        loss = 0
        for pred in predictions:
            x_layers = gm_descriptor(pred, batch.edge_index, batch.x, return_all_layers=False)
            loss = loss + (x_layers - y_layers).pow(2).mean()
        loss.backward()

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"step": step_count, "loss": loss.item()})

        # breakpoint()

        if step_count % 1000 == 0:
            del predictions, y, z, x_layers, y_layers, loss
            torch.cuda.empty_cache()
            n_samples = max(1, 1000 // batch.num_graphs)
            with torch.no_grad():
                visualize(
                    model, batch, current_step=step_count, n_samples=n_samples, outdir=outdir, gm_descriptor=gm_descriptor
                )
        step_count += 1

    # Update learning rate scheduler at the end of each epoch
    scheduler.step()

n_samples = max(1, 1000 // batch.num_graphs)
atoms_samples = visualize(model, batch, current_step="final", n_samples=n_samples, outdir=outdir, gm_descriptor=gm_descriptor)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")

if split_identifier is None:
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
    
    
    
    
