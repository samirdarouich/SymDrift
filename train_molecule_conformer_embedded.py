import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import write
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import ConformerDataset
from tspath.generative import DriftingField, HarmonicSampler
from tspath.model import EGNN, PaiNN, MLP, DiT, TorchMDDynamics, GaussianMomentDescriptor
from tspath.utils import sample_noise_like, batch_inputs_to_atoms
from tspath.alignment import get_rmsd_batched_scatter
from tspath.analysis import get_validity, evaluate_covmat, print_covmat_results, pca_plot, distance_embedder
from torch_geometric.nn import global_mean_pool
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import json
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph

logging.basicConfig(level=logging.INFO)

def create_batch_object(batch, n_samples, prior_type="harmonic"):

    # Repeat each graph in the batch n_samples times to create a new batch for sampling
    data_list = batch.to_data_list()
    repeated_list = [data for data in data_list for _ in range(n_samples)]
    batch_sampling = Batch.from_data_list(repeated_list)
    
    # Sample from the prior
    # create dummy with correct shape (batch.pos has shape (n_conformers*n_atoms,3))
    dummy = torch.zeros((batch_sampling.x.shape[0],3), device=batch_sampling.pos.device)
    
    if prior_type == "gaussian":
        # Gaussian Prior
        z = sample_noise_like(dummy, batch_sampling.batch)
    elif prior_type == "harmonic":
        # Harmonic Prior
        z = HarmonicSampler().sample(
            size=dummy.shape, 
            edge_index=batch_sampling.bonded_edge_index,
            batch=batch_sampling.batch, 
            smiles=batch_sampling.smiles,
        )
    batch_sampling.pos = z
    return batch_sampling

@torch.no_grad()
def sample(model, batch, n_samples):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()

    # Sample prior noise n_samples * B, where B is the batch size.
    batch_sampling = create_batch_object(batch, n_samples)

    # generate samples
    x = model(batch_sampling)

    if was_training:
        model.train()

    batch_sampling.pos_generated = x
    atoms_noise = batch_inputs_to_atoms(batch_sampling, "pos")
    atoms_samples = batch_inputs_to_atoms(batch_sampling, "pos_generated")

    return atoms_samples, atoms_noise

        
def visualize(model, batch, current_step, n_samples=None, outdir=None):
    step = current_step
    atoms_samples, atoms_noise = sample(model, batch, n_samples=n_samples)
    metrics_generated = get_validity(atoms_samples)
    
    # compute coverage and recall for different rmsd thresholds (not using hydrogens)
    results = evaluate_covmat(
        atoms_samples, 
        dataset_atoms, 
        thresholds=[0.1, 0.2, 0.5], 
        num_workers=8, 
        same_order=False, 
        worker_fn_type="rmsd_wo_h"
    )
    
    df, metrics = print_covmat_results(results, step, threshold=0.2)

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
        pca_plot(
            ref=dataset_atoms, 
            samples=atoms_samples, 
            embedding_style="invariant_distance", 
            save_path=f"{plot_dir}/step_{step_str}_pca.png"
        )
        with open(f"{plot_dir}/step_{step_str}_stats.json", "w") as f:
            json.dump({
                **metrics_generated,
                **metrics
            }, f, indent=4
            )
        print(f"Epoch {epoch}: Stable atoms: {metrics_generated['frac_stable_atoms']:.4f}; Stable mol: {metrics_generated['frac_stable_molecules']:.4f}")
    plt.close()
    return atoms_samples

def embedder_fn(embedder_type, x, atomic_numbers, batch_idx):
    if embedder_type == "gm":
        edge_index = radius_graph(x, r=gm_descriptor.r_max, batch=batch_idx)
        x_embedded = gm_descriptor(x, edge_index, atomic_numbers)
        x_embedded = global_mean_pool(x_embedded, batch_idx)
    elif embedder_type == "distance":
        B = batch_idx.max().item() + 1
        x_ = x.view(B, -1, 3)
        atomic_numbers_ = atomic_numbers.view(B, -1)
        x_embedded = distance_embedder(x_, atomic_numbers_, invariant=True)
    return x_embedded

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "geom_qm9"
split_identifier = "geomol"
split_identifier = "geomol_debug"

dataset = ConformerDataset(
    source=dataset_name,
    root=f"/home/samirdarouich/projects/TS_physics/tspath/data/{dataset_name}",
    split="train",
    split_identifier=split_identifier,
)

dataset_atoms = dataset.get_dataset_as_atoms()
metrics_dataset = get_validity(dataset_atoms)

model_type = "painn"
aligned = True
permuted = True
brute_force_permutations = True
model_dict = {
    "painn": PaiNN(sphere_channels=256,num_layers=9, max_radius=11.0),
    "egnn": EGNN(sphere_channels=256, num_layers=5),
    "mlp": MLP(input_dim=4*6, hidden_dim=256, num_layers=9, output_dim=3*6), # d=4*n_atoms
    "dit": DiT(sphere_channels=256, num_layers=9, num_heads=8, sphere_channels_mlp=512, max_radius=11.0),
    "torchmd": TorchMDDynamics(sphere_channels=160, num_layers=9, max_radius=11.0)
}
model = model_dict[model_type]
model.to(device)

embedder_type = "distance"
if embedder_type == "gm":
    #! GM descriptor
    n_radial = 4
    n_basis = 4
    n_contr = 4 # distances # 8 distances + angle
    gm_descriptor = GaussianMomentDescriptor(
        n_radial=n_radial, n_basis=n_basis, max_radius=11.0, n_contr=n_contr, use_atom_type_embeddings=True
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
drift_str = "all_drift"
if only_pos_drift:
    drift_str = "pos_drift"
print(f"Dataset: {dataset_name}, Model: {model_type},  drift: {drift_str}")

normalize_drift = True
temperatures = [0.05]
temp_str = "_".join([f"{t:.2f}" for t in temperatures])
outdir = f"runs/conformer/dataset_{dataset_name}/split_{split_identifier}/{model_type}/temp_{temp_str}/norm_{normalize_drift}/{embedder_str}/{drift_str}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

drifting_field = DriftingField(
    temperatures=temperatures,
    normalize_drift=normalize_drift,
)

batch_size_pos = min(len(dataset), 2)
n_neg_per_pos = 32

dataloader = GeometricDataLoader(
    dataset,
    batch_size=batch_size_pos,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
)


losses = []
model.train()

n_steps = 25_000 #500_000
n_epochs = n_steps // len(dataloader)

scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
step_count = 0
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)

        # We got n_pos graphs, where each have n_conformers
        batch_sizes = batch.num_atoms * batch.num_conformers 
        conformer_idx = torch.arange(
            batch.num_graphs, device=batch_sizes.device
        ).repeat_interleave(batch_sizes)

        # create new batch mask treating each conformer as a separate graph in the batch.
        atoms_per_conf = torch.repeat_interleave(batch.num_atoms, batch.num_conformers )
        batch_conf = torch.repeat_interleave(
            torch.arange(len(atoms_per_conf),device=device), atoms_per_conf
        )
        
        # repeat atomic numbers to match the conformer of each batch conformer
        y_pos = batch.pos.clone()
        z_split = torch.split(batch.x, batch.num_atoms.tolist())
        z_pos = torch.cat([
            z_i.repeat(n_conf_i, 1)
            for z_i, n_conf_i in zip(z_split, batch.num_conformers)
        ]).view(-1)
        with torch.no_grad():
            y_pos_embedded = embedder_fn(embedder_type, y_pos, z_pos, batch_conf)
        
        
        # Sample n_neg priors per graph
        batch_neg = create_batch_object(batch, n_samples=n_neg_per_pos, prior_type="harmonic")

        # Call the model
        x = model(batch_neg)
        
        # Embeddinf of x samples
        x_embedded = embedder_fn(embedder_type, x, batch_neg.x, batch_neg.batch)
        
        # Create masks for indexing positive and negative samples corresponding to each graph in the batch
        pos_offsets = torch.cumsum(
            torch.cat([torch.tensor([0], device=device), batch.num_conformers[:-1]]),
            dim=0
        )
        neg_offsets = torch.arange(batch.num_graphs, device=device) * n_neg_per_pos
        
        # Per Class compute the drift seperately
        V_total = torch.zeros_like(x_embedded)
        V_pos_total = torch.zeros_like(x_embedded)
        for i in range(batch.num_graphs):
            # get all positive conformers of the current graph
            start = pos_offsets[i]
            end = start + batch.num_conformers[i]
            mask_pos = torch.arange(start, end, device=device)
            y_i_pos_embedded = y_pos_embedded[mask_pos]

            # get all negative conformers of the current graph
            start = neg_offsets[i]
            end = start + n_neg_per_pos
            mask_neg = torch.arange(start, end, device=device)
            x_i_embed = x_embedded[mask_neg]
            
            # Call the drift (atomic numbers will be repeated for negative samples)
            V, V_pos, V_neg, *_ = drifting_field(
                x_i_embed.detach(),
                y_i_pos_embedded,
                x_i_embed.detach(),
            )
            V_total[mask_neg] = V
            V_pos_total[mask_neg] = V_pos

        # In case only attraction
        if only_pos_drift:
            x_drifted = (x_embedded + V_pos_total).detach()
        else:
            x_drifted = (x_embedded + V_total).detach()
        
        # Compute RMSD loss
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
                "mse(V)": torch.sqrt(torch.mean(V_total**2)).item(),
                "mse(pos_drift)": torch.sqrt(torch.mean(V_pos_total**2)).item(), 
                "mse(neg_drift)": torch.sqrt(torch.mean(V_neg**2)).item(),
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
            
    # if (epoch < 101 and epoch % 5 == 0) or (epoch>100 and epoch % 20 == 0):
    #     torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/epoch_{epoch}.pt")

n_samples = 1000 // batch.num_graphs
atoms_samples = visualize(model, batch, current_step="final", n_samples=n_samples, outdir=outdir)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")

metrics_generated = get_validity(atoms_samples)
for metric_name in metrics_dataset.keys():
    print(f"{metric_name}: Dataset: {metrics_dataset[metric_name]:.4f}, Generated: {metrics_generated[metric_name]:.4f}")
with open(f"{plot_dir}/metrics.json", "w") as f:
    json.dump({"dataset": metrics_dataset, "generated": metrics_generated}, f, indent=4)