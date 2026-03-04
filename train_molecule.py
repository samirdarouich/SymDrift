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

        
def visualize(model, batch, current_epoch, n_samples=None, outdir=None):
    epoch = current_epoch
    atoms_samples = sample(model, batch, n_samples=n_samples)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        if isinstance(epoch, int):
            epoch_str = f"{epoch:04d}"
        else:
            epoch_str = str(epoch)
        write(f"{plot_dir}/epoch_{epoch_str}.png", atoms_samples[0])
        write(f"{plot_dir}/epoch_{epoch_str}.xyz", atoms_samples)
    plt.close()


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "t1x_eq_CHN3O" #! just 6 ATOMS
augment_with_rotations = False
augment_with_permutations = False

dataset = MoleculeDataset(
    source=dataset_name,
    root="/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq",
    split="train",
    split_identifier="debug",
    augment_with_rotations=augment_with_rotations,
    augment_with_permutations=augment_with_permutations,
)

batch_sampler = CompositionBatchSampler(
    dataset,
    k=1,  # number of chunks per batch
    n=1,  # chunk size (number of samples per chunk)
    shuffle=True,
    drop_last=False,
    resample=True,
    seed=42,
)
dataloader = GeometricDataLoader(
    dataset, batch_sampler=batch_sampler, shuffle=False
)

model_type = "egnn"
aligned = True
permuted = False
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

outdir = f"runs/molecule/dataset_{dataset_name}/{model_type}/augment_rot_{augment_with_rotations}_augment_perm_{augment_with_permutations}/aligned_{aligned}_permuted_{permuted}_brute_force_{brute_force_permutations}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

drifting_field = EquivariantDriftingField(
    temperature=0.15,
    aligned=aligned,
    permuted=permuted,
    brute_force_permutations=brute_force_permutations,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

losses = []
model.train()
n_epochs = 500
n_samples = 1000
pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y = batch.pos.clone()

        # Sample noise
        # z = sample_noise_like(batch.pos, batch.batch)
        # batch.pos = z
        
        # Sample bz negative samples
        batch_ = create_batch_object(batch, n_samples=64)

        # Call the model
        x = model(batch_)
        
        # Call the drift
        V, *_ = drifting_field(
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
        pbar.set_postfix({"loss": loss.item()})

        if epoch % (n_epochs // 10) == 0 and batch_idx == 0 and epoch > 0:
            visualize(
                model, batch, current_epoch=epoch, n_samples=n_samples, outdir=outdir
            )
            
    # if (epoch < 101 and epoch % 5 == 0) or (epoch>100 and epoch % 20 == 0):
    #     torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/epoch_{epoch}.pt")

visualize(model, batch, current_epoch="final", n_samples=n_samples, outdir=outdir)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")
