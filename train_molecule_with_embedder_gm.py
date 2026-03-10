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
from tspath.alignment import get_rmsd_batched
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
        align=True, permute=True, brute_force_permutations=True
    )
    
    if was_training:
        model.train()

    batch_sampling.pos_generated = x
    atoms_noise = batch_inputs_to_atoms(batch_sampling, "pos")
    atoms_samples = batch_inputs_to_atoms(batch_sampling, "pos_generated")
    
    for i, atom in enumerate(atoms_samples):
        atom.info["rmsd"] = rmsd[i].item()
    
    print(f"RMSD of generated samples to target: {rmsd.mean().item():.4f} Å")
    return atoms_samples, atoms_noise, rmsd

        
def visualize(model, batch, current_step, n_samples=None, outdir=None):
    step = current_step
    atoms_samples, atoms_noise, rmsd = sample(model, batch, n_samples=n_samples)
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
        with open(f"{plot_dir}/step_{step_str}_rmsd.txt", "w") as f:
            f.write(f"RMSD: mean: {rmsd.mean().item():.4f} Å, median: {rmsd.median().item():.4f} Å\n")
    plt.close()
    return atoms_samples

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "t1x_eq_CHN3O" #! 6 ATOMS
# dataset_name = "t1x_eq_C3H2N2O2" #! 9 ATOMS
# dataset_name = "t1x_eq_C5H8O" #! 14 ATOMS
split_identifier = None
# split_identifier = "debug"

dataset = MoleculeDataset(
    source=dataset_name,
    root="/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq",
    split="train",
    split_identifier=split_identifier,
)

model_type = "painn"
model_dict = {
    "painn": PaiNN(sphere_channels=256,num_layers=9, max_radius=11.0),# PaiNN(),#PaiNN(sphere_channels=256,num_layers=6), PaiNN(sphere_channels=256,num_layers=9),
    "egnn": EGNN(sphere_channels=256, num_layers=5),
}
model = model_dict[model_type]
model.to(device)

print(f"Dataset: {dataset_name}, Model: {model_type}, ")

#! GM descriptor
n_radial = 5
n_basis = 4
n_contr = 8
gm_descriptor = GaussianMomentDescriptor(
    n_radial=n_radial, n_basis=n_basis, max_radius=11.0, n_contr=n_contr, use_atom_type_embeddings=True
)
gm_descriptor.to(device)

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

optimizer = torch.optim.AdamW(
    [
        {"params": model.parameters(), "lr": 1e-4},
        {"params": gm_descriptor.parameters(), "lr": 1e-4},
    ],
    weight_decay=0.1
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

n_steps = 9000
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
        
        # Encode samples and targets
        x_embedded = gm_descriptor(x, batch.edge_index, batch.x)
        x_embedded = global_mean_pool(x_embedded, batch.batch)
        
        y_embedded = gm_descriptor(y, batch.edge_index, batch.x)
        y_embedded = global_mean_pool(y_embedded, batch.batch)
        
        # Call the drift
        V, drift_pos, drift_neg, *_ = drifting_field(
            x_embedded.detach(),
            y_embedded,
            x_embedded.detach(),
        )
        
        # sum over temperatures to get final V of shape (N, d)
        if split_identifier == "debug":
            v_norm = torch.sqrt(torch.mean(drift_pos**2, dim=(1, 2)))  # (T)
            drift_pos_ = drift_pos / (v_norm[:, None, None] + 1e-8)
            drift_pos_ = drift_pos.sum(dim=0)
            drift_pos_ = drift_pos_.view_as(x_embedded)
            x_drifted = (x_embedded + drift_pos_).detach()
        else:
            x_drifted = (x_embedded + V).detach()

        loss = torch.nn.functional.mse_loss(x_embedded, x_drifted)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(
            {
                "step": step_count, 
                "mse(V)": torch.sqrt(torch.mean(V**2)).item(),
                # take mse per temperature and sum over temperatures (as done with V)
                "mse(pos_drift)": torch.sum(torch.sqrt(torch.mean(drift_pos**2, dim=(1,2)))).item(), 
                "mse(neg_drift)": torch.sum(torch.sqrt(torch.mean(drift_neg**2, dim=(1,2)))).item(),
            }
        )

        if step_count % (n_steps // 10) == 0 and step_count > 0:
            # Sample in total 1000 samples. This will produce per batch item n_samples, 
            # which will result in 1000 samples overall
            n_samples = 1000 // batch.num_graphs 
            visualize(
                model, batch, current_step=step_count, n_samples=n_samples, outdir=outdir
            )
        step_count += 1

    # Update learning rate scheduler at the end of each epoch
    scheduler.step()

n_samples = 1000 // batch.num_graphs
atoms_samples = visualize(model, batch, current_step="final", n_samples=n_samples, outdir=outdir)
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
    
    
    
    
