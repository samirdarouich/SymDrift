import json
import logging
import os

import matplotlib.pyplot as plt
import torch
from ase.io import write
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.alignment import get_rmsd_batched_scatter
from tspath.analysis import (
    evaluate_covmat,
    get_validity,
    pca_plot,
    print_covmat_results,
)
from tspath.datasets import ConformerDataset
from tspath.generative import EquivariantDriftingField, GaussianSampler, HarmonicSampler
from tspath.model import (
    EGNN,
    MLP,
    CosineAnnealingWarmupRestarts,
    DistanceEmbedder,
    DiT,
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
        thresholds=[0.1, 0.2, 0.5],
        num_workers=8,
        worker_fn_type="rmsd_wo_h",
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
        pca_plot(
            refs=dataset_atoms,
            samples=atoms_samples,
            embedder=DistanceEmbedder(invariant=True),
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

model_type = "dit_naive"
aligned = False
permuted = False
brute_force_permutations = False
model_dict = {
    "painn": PaiNN(sphere_channels=256, num_layers=9, max_radius=11.0),
    "egnn": EGNN(sphere_channels=256, num_layers=5),
    "mlp": MLP(
        input_dim=4 * 6, hidden_dim=256, num_layers=9, output_dim=3 * 6
    ),  # d=4*n_atoms
    "dit": DiT(
        sphere_channels=256,
        num_layers=9,
        num_heads=8,
        sphere_channels_mlp=512,
        max_radius=11.0,
    ),
    "dit_perm_eq": DiT(positional_encoding_bool=False, relative_positional_embedding_bool=False, max_radius=11.0), #positional_encoding_bool=True breaks permutation equivariance, 
    "dit_naive": DiT(
        positional_encoding_bool=True, # break permutation equivariance
        relative_positional_embedding_bool=True, # break rotation equivariance
        absolute_positional_embedding_bool=True, # break translation and rotation equivariance
        max_radius=11.0)
    ,
    "torchmd": TorchMDDynamics(sphere_channels=160, num_layers=9),
}
model = model_dict[model_type]
model.to(device)

only_pos_drift = False
drift_str = "all_drift"
if only_pos_drift:
    drift_str = "pos_drift"
print(
    f"Dataset: {dataset_name}, Model: {model_type}, Aligned: {aligned}, Permuted: {permuted}, Brute-force permutations: {brute_force_permutations}, drift: {drift_str}"
)

normalize_drift = True
temperatures = [0.02, 0.05, 0.2]
temp_str = "_".join([f"{t:.2f}" for t in temperatures])
outdir = f"runs/conformer/dataset_{dataset_name}/split_{split_identifier}/{model_type}/temp_{temp_str}/norm_{normalize_drift}/aligned_{aligned}_permuted_{permuted}_brute_force_{brute_force_permutations}/{drift_str}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

## Define the prior sampler (harmonic or gaussian)
sampler_type = "harmonic"
if sampler_type == "harmonic":
    prior_sampler = HarmonicSampler()
elif sampler_type == "gaussian":
    prior_sampler = GaussianSampler()


drifting_field = EquivariantDriftingField(
    temperatures=temperatures,
    aligned=aligned,
    permuted=permuted,
    brute_force_permutations=brute_force_permutations,
    normalize_drift=normalize_drift,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-8)

batch_size_pos = min(len(dataset), 2)
n_neg_per_pos = 32

dataloader = GeometricDataLoader(
    dataset,
    batch_size=batch_size_pos,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
    follow_batch=["x_conf"]
)


losses = []
model.train()

n_steps = 2_000
n_epochs = n_steps // len(dataloader)

scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=250_000,
    cycle_mult=1.0,
    max_lr=7e-4,
    min_lr=1e-5,
    warmup_steps=0,
    gamma=0.05,
    last_epoch=-1,
)

pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
step_count = 0
for epoch in pbar:
    for batch_idx, batch_pos in enumerate(dataloader):
        optimizer.zero_grad()
        batch_pos = batch_pos.to(device)

        # We got n_pos graphs, where each have n_conformers (n_conf_i*n_atom_i)
        y_pos = batch_pos.pos.clone()
        z_pos = batch_pos.x_conf.clone()

        # Sample n_neg priors per graph
        batch_neg = create_batch_object(
            batch_pos, n_samples=n_neg_per_pos
        )

        # Call the model
        x = model(batch_neg)

        # Per class compute the drift seperately
        V_total = torch.zeros_like(x)
        V_pos_total = torch.zeros_like(x)
        for i in range(batch_pos.num_graphs):
            # Get all conformers that correspond to the current positive sample
            mask_pos = batch_pos.x_conf_batch == i
            y_i_pos = y_pos[mask_pos]
            z_i_pos = z_pos[mask_pos]
            
            # search negative samples corresponding to the current positive sample
            start = i * n_neg_per_pos
            end = (i + 1) * n_neg_per_pos
            mask_neg = (batch_neg.batch >= start) & (batch_neg.batch < end)
            x_i = x[mask_neg]
            z_i_neg = batch_neg.x[mask_neg]

            # Compute the drift
            V, V_pos, V_neg, *_ = drifting_field(
                x_i.detach(),  # avoid unnecessary gradient tracking
                y_i_pos,
                x_i.detach(),  # avoid unnecessary gradient tracking
                batch_pos.num_atoms[i],
                atomic_numbers_pos=z_i_pos,
                atomic_numbers_neg=z_i_neg,
            )
            V_total[mask_neg] = V
            V_pos_total[mask_neg] = V_pos

        # In case only attraction
        if only_pos_drift:
            x_drifted = (x + V_pos_total).detach()
        else:
            x_drifted = (x + V_total).detach()

        # Compute RMSD loss
        loss = get_rmsd_batched_scatter(x, x_drifted, batch_neg.batch).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
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
            n_samples = 1000 // batch_pos.num_graphs
            visualize(
                model,
                batch_pos,
                current_step=step_count,
                n_samples=n_samples,
                outdir=outdir,
            )
        step_count += 1

    # if (epoch < 101 and epoch % 5 == 0) or (epoch>100 and epoch % 20 == 0):
    #     torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/epoch_{epoch}.pt")

n_samples = 1000 // batch_pos.num_graphs
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
