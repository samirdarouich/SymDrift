import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import write
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import MoleculeDataset, CompositionBatchSampler
from tspath.generative import EquivariantDriftingField
from tspath.model import EGNN, PaiNN, MLP, DiT
from tspath.utils import sample_noise_like, batch_inputs_to_atoms
from tspath.alignment import get_rmsd_batched
from tspath.analysis import get_validity, evaluate_covmat, print_covmat_results, pca_plot
from torch_geometric.data import Data
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import json
import math

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
    batch_sampling.num_graphs = total_samples
    batch_sampling.ptr = torch.arange(0, total_samples * n_atoms + 1, n_atoms, device=device)

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

    # Determine whether to use brute-force permutations
    unique_types = torch.unique(batch.x.view(-1, n_atoms)[0], dim=0, return_counts=True)
    no_permutations = n_samples
    for t, count in zip(*unique_types):
        no_permutations *= math.factorial(count.item())

    if no_permutations < 1e5:
        use_brute_force_permutations = True
    else:
        use_brute_force_permutations = False
    
    rmsd = get_rmsd_batched(
        x_sample, x_target, atomic_numbers=batch_sampling.x.view(-1, n_atoms),
        align=True, permute=True, brute_force_permutations=use_brute_force_permutations
    )
    
    if was_training:
        model.train()

    batch_sampling.pos_generated = x
    atoms_noise = batch_inputs_to_atoms(batch_sampling, "pos")
    atoms_samples = batch_inputs_to_atoms(batch_sampling, "pos_generated")
    
    for i, atom in enumerate(atoms_samples):
        atom.info["rmsd"] = rmsd[i].item()
    return atoms_samples, atoms_noise, rmsd

        
def visualize(model, batch, current_step, n_samples=None, outdir=None):
    step = current_step
    atoms_samples, atoms_noise, rmsd = sample(model, batch, n_samples=n_samples)
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
                "rmsd": {
                    "mean": rmsd.mean().item(),
                    "median": rmsd.median().item()
                },
                **metrics_generated,
                **metrics
            }, f, indent=4
            )
        print(f"Epoch {epoch}: Sample RMSD: {rmsd.mean().item():.4f}; Stable atoms: {metrics_generated['frac_stable_atoms']:.4f}; Stable mol: {metrics_generated['frac_stable_molecules']:.4f}")
    plt.close()
    return atoms_samples


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
# dataset_name = "t1x_eq_CHN3O" #! 6 ATOMS
# dataset_name = "t1x_eq_C3H2N2O2" #! 9 ATOMS
# dataset_name = "t1x_eq_C5H8O" #! 14 ATOMS
dataset_name = "qm9_C3H2N2O2" #! 9 ATOMS
data_folder = "qm9" if dataset_name.startswith("qm9") else "transition1x_eq"

split_identifier = None
# split_identifier = "debug"

augment_with_rotations = False
augment_with_permutations = False

dataset = MoleculeDataset(
    source=dataset_name,
    root=f"/home/samirdarouich/projects/TS_physics/tspath/data/{data_folder}",
    split="train",
    identifier="identifier" if dataset_name.startswith("qm9") else "rxn",
    split_identifier=split_identifier,
    augment_with_rotations=augment_with_rotations,
    augment_with_permutations=augment_with_permutations,
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
}
model = model_dict[model_type]
model.to(device)

only_pos_drift = False
drift_str = "all_drift"
if only_pos_drift:
    drift_str = "pos_drift"
print(f"Dataset: {dataset_name}, Model: {model_type}, Aligned: {aligned}, Permuted: {permuted}, Brute-force permutations: {brute_force_permutations}, Augment with rotations: {augment_with_rotations}, Augment with permutations: {augment_with_permutations}, drift: {drift_str}")

normalize_drift = True
temperatures = [0.05]
temp_str = "_".join([f"{t:.2f}" for t in temperatures])
outdir = f"runs/molecule_wrong_ordering/dataset_{dataset_name}/split_{split_identifier}/{model_type}/temp_{temp_str}/norm_{normalize_drift}/augment_rot_{augment_with_rotations}_augment_perm_{augment_with_permutations}/aligned_{aligned}_permuted_{permuted}_brute_force_{brute_force_permutations}/{drift_str}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

drifting_field = EquivariantDriftingField(
    temperatures=temperatures,
    aligned=aligned,
    permuted=permuted,
    brute_force_permutations=brute_force_permutations,
    normalize_drift=normalize_drift,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

batch_size_pos = min(len(dataset), 64)
batch_size_neg = 64

batch_sampler = CompositionBatchSampler(
    dataset,
    k=1,  # number of chunks per batch
    n=batch_size_pos,  # chunk size (number of samples per chunk)
    shuffle=True,
    drop_last=False,
    resample=False,
    seed=42,
)
dataloader = GeometricDataLoader(
    dataset, batch_sampler=batch_sampler, shuffle=False
)

losses = []
model.train()

n_steps = 1_000 #500_000
n_epochs = n_steps // len(dataloader)

scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
step_count = 0
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y_orig = batch.pos.clone()
        y = batch.pos.clone()

        # Permute atomic numbers to be canonically ordered. This is important, otherwise
        # the brute force algorithm will do incorrect permutations when comapring different
        # x to ys.
        # z_orig = batch.x.reshape(batch.num_graphs, -1).clone()
        # sort_idx = torch.argsort(z_orig, dim=1)
        # gather_idx = sort_idx[..., None].expand(-1, -1, y_orig.shape[1])
        # y = torch.gather(y_orig.view(batch.num_graphs, batch.num_atoms[0], -1), 1, gather_idx).view(-1, y_orig.shape[1])
        # new_z = torch.gather(z_orig, 1, sort_idx).view(-1)
        # batch.x = new_z
        batch.pos_orig = y.clone()

        # Sample noise
        # if less positive samples than required negatve samples, repeat positive 
        # samples until we have enough
        if batch_size_pos < batch_size_neg:
            n_repeat = batch_size_neg // batch.num_graphs
            batch_neg = create_batch_object(batch, n_samples=n_repeat)
        else:
            # Sample bz negative samples
            n_repeat = 1
            batch_neg = batch.clone()
            z = sample_noise_like(batch.pos, batch.batch)
            batch_neg.pos = z

        # Call the model
        x = model(batch_neg)
        
        # # THIS is 0
        # batch_test = batch.clone()
        # batch_test.x = z_orig.view(-1)
        # batch_neg_test = create_batch_object(batch_test, n_samples=n_repeat)
        
        # sort_idx = torch.argsort(batch_neg_test.x.view(-1,9),dim=1)
        # inv_sort_idx = torch.argsort(sort_idx, dim=1)
        # gather_idx = inv_sort_idx[..., None].expand(-1, -1, 3)
        # batch_neg_test.pos = torch.gather(batch_neg.pos.view(-1,9,3),1,gather_idx).view(-1,3)
        # x_test = model(batch_neg_test)
        
        # x_reordered = torch.gather(x.view(-1,9,3),1,gather_idx).view(-1,3)
        # diff = (x_test - x_reordered).norm()

        # Call the drift
        V, drift_pos, drift_neg, *_ = drifting_field(
            x.detach(), # avoid unnecessary gradient tracking
            y,
            x.detach(), # avoid unnecessary gradient tracking
            batch.num_atoms[0],
            atomic_numbers=batch.x,
        )
            
        # In case of many to one, use regression task (with alignment)
        if only_pos_drift:
            x_drifted = (x + drift_pos).detach()
        else:
            x_drifted = (x + V).detach()
        
        # Compute RMSD loss
        loss = get_rmsd_batched(
            x.view(n_repeat*batch_size_pos, -1, 3),
            x_drifted.view(n_repeat*batch_size_pos, -1, 3),
        ).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(
            {
                "step": step_count, 
                "loss": loss.item(),
                "mse(V)": torch.sqrt(torch.mean(V**2)).item(),
                "mse(pos_drift)": torch.sqrt(torch.mean(drift_pos**2)).item(), 
                "mse(neg_drift)": torch.sqrt(torch.mean(drift_neg**2)).item(),
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