import os

import matplotlib.pyplot as plt
from ase.io import write
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import MoleculeDataset, CompositionBatchSampler
from tspath.generative import EquivariantDriftingField
from tspath.model import EGNN, EquiformerV2, PaiNN
from tspath.utils import sample_noise_like, batch_inputs_to_atoms
from torch_geometric.data import Data


def create_batch_object(batch, n_samples):
    
    batch_sampling = Data()
    # Sample prior noise
    n_atoms = batch.num_atoms[0].item()
    d = batch.pos.shape[1]

    batch_ = torch.arange(n_samples, device=device).repeat_interleave(n_atoms)
    dummy = torch.zeros((n_samples * n_atoms, d), dtype=torch.float, device=device)
    x = batch.x[:n_atoms].repeat(n_samples)
    num_atoms = batch.num_atoms[0].repeat(n_samples)

    z = sample_noise_like(dummy, batch_)
    
    batch_sampling.pos = z
    batch_sampling.batch = batch_
    batch_sampling.x = x
    batch_sampling.num_atoms = num_atoms
    return batch_sampling

@torch.no_grad()
def sample(model, batch, n_samples):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()

    # Sample prior noise
    batch_sampling = create_batch_object(batch, n_samples)

    # generate samples
    x = model(batch_sampling)

    if was_training:
        model.train()

    batch_sampling.pos_generated = x
    atoms_samples = batch_inputs_to_atoms(batch_sampling, "pos_generated")
    return atoms_samples

        
def visualize(model, batch, current_step, n_samples=None, outdir=None):
    step = current_step
    atoms_samples = sample(model, batch, n_samples=n_samples)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        if isinstance(step, int):
            step_str = f"{step:04d}"
        else:
            step_str = str(step)
        write(f"{plot_dir}/step_{step_str}.png", atoms_samples[0])
        write(f"{plot_dir}/step_{step_str}.xyz", atoms_samples)
    plt.close()


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "t1x_eq_CHN3O" #! just 6 ATOMS
augment_with_rotations = True
augment_with_permutations = True

dataset = MoleculeDataset(
    source=dataset_name,
    root="/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq",
    split="train",
    split_identifier="debug",
    augment_with_rotations=augment_with_rotations,
    augment_with_permutations=augment_with_permutations,
)

model_type = "egnn"
aligned = True
permuted = True
brute_force_permutations = False
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

normalize_drift = True
temperatures = [0.05]
temp_str = "_".join([f"{t:.2f}" for t in temperatures])
outdir = f"runs/molecule/dataset_{dataset_name}/{model_type}/temp_{temp_str}/norm_{normalize_drift}/augment_rot_{augment_with_rotations}_augment_perm_{augment_with_permutations}/aligned_{aligned}_permuted_{permuted}_brute_force_{brute_force_permutations}"
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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

# In case of brute-force we just want to sample one target to make the alignment cheaper
if brute_force_permutations:
    batch_size = 1
else:
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

n_samples = 1000
pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
step_count = 0
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y = batch.pos.clone()

        # Sample noise
        if batch.num_graphs == 1:
            batch_neg = create_batch_object(batch, n_samples=batch_size)
        else:
            # Sample bz negative samples
            batch_neg = batch.clone()
            z = sample_noise_like(batch.pos, batch.batch)
            batch_neg.pos = z

        # Call the model
        x = model(batch_neg)
        
        # Call the drift
        V, drift_pos, drift_neg, *_ = drifting_field(
            x,
            y,
            x,
            batch.num_atoms[0],
            atomic_numbers=batch.x,
        )

        x_drifted = (x + V).detach()

        loss = torch.nn.functional.mse_loss(x, x_drifted)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(
            {
                "step": step_count, 
                "mse(V)": loss.item(),
                "mse(pos_drift)": torch.sqrt(torch.mean(drift_pos**2)).item(),
                "mse(neg_drift)": torch.sqrt(torch.mean(drift_neg**2)).item(),
            }
        )

        if step_count % (n_steps // 10) == 0 and step_count > 0:
            visualize(
                model, batch, current_step=step_count, n_samples=n_samples, outdir=outdir
            )
        step_count += 1
            
    # if (epoch < 101 and epoch % 5 == 0) or (epoch>100 and epoch % 20 == 0):
    #     torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/epoch_{epoch}.pt")

visualize(model, batch, current_step="final", n_samples=n_samples, outdir=outdir)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")
