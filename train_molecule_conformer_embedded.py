import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import write
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.analysis import (
    evaluate_covmat,
    get_validity,
    pca_plot,
    print_covmat_results,
)
from tspath.datasets import ConformerDataset
from tspath.generative import DriftingField, GaussianSampler, HarmonicSampler
from tspath.model import (
    EGNN,
    MLP,
    DistanceEmbedder,
    DiT,
    GaussianMomentEmbedder,
    PaiNN,
    TorchMDDynamics,
)
from tspath.utils import batch_inputs_to_atoms

logging.basicConfig(level=logging.INFO)


def create_batch_object(batch, n_samples):

    # Repeat each graph in the batch n_samples times to create a new batch for sampling
    data_list = batch.to_data_list()
    repeated_list = [data for data in data_list for _ in range(n_samples)]
    batch_negative = Batch.from_data_list(repeated_list)

    z = prior_sampler.sample(
        size=(batch_negative.num_nodes, 3),
        edge_index=batch_negative.bonded_edge_index,
        batch=batch_negative.batch,
        smiles=batch_negative.smiles,
    )
    batch_negative.pos = z

    return batch_negative


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
    atoms_noise = batch_inputs_to_atoms(batch_sampling, "pos", ["smiles"])
    atoms_samples = batch_inputs_to_atoms(batch_sampling, "pos_generated", ["smiles"])

    return atoms_samples, atoms_noise


def visualize(model, batch, current_step, n_samples=None, outdir=None):
    step = current_step
    atoms_samples, atoms_noise = sample(model, batch, n_samples=n_samples)
    metrics_generated = get_validity(atoms_samples)

    # compute coverage and recall for different rmsd thresholds (not using hydrogens)
    results = evaluate_covmat(
        atoms_samples,
        dataset_atoms,
        thresholds=np.arange(0.05, 3.05, 0.05),
        num_workers=8,
        worker_fn_type="rmsd_rdkit_wo_h",
        ratio=2.0, # only keep at most 2*n_conformers predictions per reference
    )

    df, metrics = print_covmat_results(results, threshold=0.2)

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
        df.to_csv(f"{plot_dir}/step_{step_str}_covmat_results.csv", index=False)
        pca_plot(
            refs=dataset_atoms,
            samples=atoms_samples,
            embedder=embedder,
            save_path=f"{plot_dir}/step_{step_str}_pca.png",
        )
        with open(f"{plot_dir}/step_{step_str}_stats.json", "w") as f:
            json.dump({**metrics_generated, **metrics}, f, indent=4)
        print(
            f"Epoch {epoch}: Stable atoms: {metrics_generated['frac_stable_atoms']:.4f}; Stable mol: {metrics_generated['frac_stable_molecules']:.4f}"
        )
    plt.close()
    return atoms_samples


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "geom_qm9"
split_identifier = "geomol"
split_identifier = "geomol_debug"
split_identifier = "geomol_debug_bigger"

dataset = ConformerDataset(
    source=dataset_name,
    root=f"/home/samirdarouich/projects/TS_physics/tspath/data/{dataset_name}",
    split="train",
    split_identifier=split_identifier,
)

dataset_atoms = dataset.get_dataset_as_atoms()
metrics_dataset = get_validity(dataset_atoms)

model_type = "painn"
model_dict = {
    "painn": PaiNN(sphere_channels=256, num_layers=9, max_radius=11.0),
    "egnn": EGNN(sphere_channels=256, num_layers=5),
    "torchmd": TorchMDDynamics(sphere_channels=160, num_layers=9, max_radius=11.0),
}
model = model_dict[model_type]
model.to(device)


## define the embedder (GM or distance)
embedder_type = "gm"
if embedder_type == "gm":
    #! GM descriptor
    n_radial = 4
    n_basis = 8
    n_contr = 8  # distances # 8 distances + angle
    embedder = GaussianMomentEmbedder(
        n_radial=n_radial,
        n_basis=n_basis,
        max_radius=15.0,
        n_contr=n_contr,
        use_atom_type_embeddings=False, # no learnable atom type embeddings
    )
    embedder.to(device)

    embedder_str = (
        f"embedder_gm/n_radial_{n_radial}_n_basis_{n_basis}_n_contr_{n_contr}"
    )
elif embedder_type == "distance":
    #! Invariant distance embedder
    embedder_str = "embedder_distance"
    embedder = DistanceEmbedder(invariant=True)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

## Define the prior sampler (harmonic or gaussian)
sampler_type = "harmonic"
if sampler_type == "harmonic":
    prior_sampler = HarmonicSampler()
elif sampler_type == "gaussian":
    prior_sampler = GaussianSampler()


only_pos_drift = False
drift_str = "all_drift"
if only_pos_drift:
    drift_str = "pos_drift"
print(f"Dataset: {dataset_name}, Model: {model_type},  drift: {drift_str}")

normalize_drift = True
temperatures = [0.02, 0.05, 0.2]
temp_str = "_".join([f"{t:.2f}" for t in temperatures])
outdir = f"runs/geom_qm9/split_{split_identifier}/{model_type}/temp_{temp_str}/norm_{normalize_drift}/{embedder_str}/{drift_str}"
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

n_steps = 5_000
n_epochs = n_steps // len(dataloader)

scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
step_count = 0
for epoch in pbar:
    for batch_idx, batch_pos in enumerate(dataloader):
        optimizer.zero_grad()
        batch_pos = batch_pos.to(device)

        # Create batch mask treating each conformer as seperate graph
        batch_mask_pos = batch_pos.conformer_index
        y_pos = batch_pos.pos.clone()
        z_pos = batch_pos.x_conf.clone()
        conformer_offsets = [0] + torch.cumsum(batch_pos.num_conformers, dim=0).tolist()

        # Sample n_neg priors per graph
        batch_neg = create_batch_object(batch_pos, n_samples=n_neg_per_pos)

        # Call the model
        x = model(batch_neg)
        
        # Call the embedder for the whole batch. Embedding output is one flatten vector
        # and a mask indicating which embedding belong to which batch element
        with torch.no_grad():
            y_pos_embedded, mask_pos = embedder(
                positions=y_pos, Z=z_pos, batch=batch_mask_pos
            )
            
        x_embedded, mask_x = embedder(
            positions=x, Z=batch_neg.x, batch=batch_neg.batch
        )

        # Per class compute the drift seperately
        loss = torch.tensor(0.0, device=x.device)
        for i in range(batch_pos.num_graphs):
            # Get all embeddings corresponding to the current positive conformers
            start = conformer_offsets[i]
            end = conformer_offsets[i+1]
            mask_pos_i = torch.isin(
                mask_pos, 
                torch.arange(start, end, device=mask_pos.device)
            )
            
            # Reshape to (n_conformers_i, embed_dim_i)
            y_i_pos_embedded = y_pos_embedded[mask_pos_i].view(
                batch_pos.num_conformers[i], -1
            )

            # search negative samples corresponding to the current positive sample
            start = i * n_neg_per_pos
            end = (i + 1) * n_neg_per_pos
            mask_neg_i = torch.isin(
                mask_x, 
                torch.arange(start, end, device=mask_pos.device)
            )
            
            # Reshape to (n_neg_per_pos, embed_dim_i)
            x_i_embedded = x_embedded[mask_neg_i].view(
                n_neg_per_pos, -1
            )

            # Call the drift
            V, V_pos, V_neg, *_ = drifting_field(
                x_i_embedded.detach(),
                y_i_pos_embedded,
                x_i_embedded.detach(),
            )

            if only_pos_drift:
                x_i_drifted = (x_i_embedded + V_pos).detach()
            else:
                x_i_drifted = (x_i_embedded + V).detach()

            loss = loss + torch.nn.functional.mse_loss(x_i_embedded, x_i_drifted)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(
            {
                "step": step_count,
                "loss": loss.item(),
            }
        )

        if step_count % (n_steps // 10) == 0 and step_count > 0:
            # Sample per positive graph n_samples conformers and visualize results
            n_samples = 32
            visualize(
                model,
                batch_pos,
                current_step=step_count,
                n_samples=n_samples,
                outdir=outdir,
            )
        step_count += 1

    # Update learning rate scheduler at the end of each epoch
    scheduler.step()

# Sample per positive graph n_samples conformers and visualize results
n_samples = 32
atoms_samples = visualize(
    model, batch_pos, current_step="final", n_samples=n_samples, outdir=outdir
)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")

metrics_generated = get_validity(atoms_samples)
for metric_name in metrics_dataset.keys():
    print(
        f"{metric_name}: Dataset: {metrics_dataset[metric_name]:.4f}, Generated: {metrics_generated[metric_name]:.4f}"
    )
with open(f"{plot_dir}/metrics.json", "w") as f:
    json.dump({"dataset": metrics_dataset, "generated": metrics_generated}, f, indent=4)
