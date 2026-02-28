import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import ToyMoleculeDataset
from tspath.generative import EquivariantDriftingField
from tspath.model import EGNN, EquiformerV2, PaiNN
from tspath.utils import sample_noise_like_2d


@torch.no_grad()
def sample(model, batch, n_samples=None):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()
    batch_sampling = batch.clone()

    # Sample prior noise
    n_atoms = batch.num_atoms[0].item()
    if n_samples is not None:
        batch_ = torch.arange(n_samples, device=device).repeat_interleave(n_atoms)
        dummy = torch.zeros((n_samples * n_atoms, 2), dtype=torch.float, device=device)
        x = batch.x[0].repeat(n_samples * n_atoms)
        num_atoms = batch.num_atoms[0].repeat(n_samples * n_atoms)
        z = sample_noise_like_2d(dummy, batch_)
    else:
        batch_ = batch.batch
        z = sample_noise_like_2d(batch.pos, batch_)
        x = batch.x
        num_atoms = batch.num_atoms
    batch_sampling.pos = z
    batch_sampling.batch = batch_
    batch_sampling.x = x
    batch_sampling.num_atoms = num_atoms

    # generate samples
    x = model(batch_sampling)

    if was_training:
        model.train()

    return x.cpu().numpy()


def evaluate_samples(samples, r0, theta0):
    samples_ = samples.reshape(-1, 3, 3)
    dist01 = r0
    dist12 = r0
    dist02 = r0 * 2.0 * np.sin(np.radians(theta0) / 2)
    target, _ = torch.sort(
        torch.tensor([[dist01, dist12, dist02]], device=samples.device), dim=1
    )

    # Sort distances for permutation invariance
    distances = torch.cdist(samples_, samples_)
    pairwise = distances[
        :, torch.triu_indices(3, 3, offset=1)[0], torch.triu_indices(3, 3, offset=1)[1]
    ]
    pairwise_sorted, _ = torch.sort(pairwise, dim=1)
    mse = ((target - pairwise_sorted) ** 2).mean()
    return mse.item()


def visualize(model, batch, current_epoch, n_samples=None, outdir=None):
    epoch = current_epoch
    x_samples = sample(model, batch, n_samples=n_samples)
    mse = evaluate_samples(torch.tensor(x_samples), r0, theta0)
    plt.scatter(
        x_samples[:, 0], x_samples[:, 1], alpha=0.5, color="red", label="Samples"
    )
    plt.scatter(
        pos_dataset[:, :, 0],
        pos_dataset[:, :, 1],
        alpha=0.75,
        color="gray",
        label="Dataset",
    )
    plt.legend()
    plt.xlim(
        pos_dataset[:, :, 0].min().item() - 0.5, pos_dataset[:, :, 0].max().item() + 0.5
    )
    plt.ylim(
        pos_dataset[:, :, 1].min().item() - 0.5, pos_dataset[:, :, 1].max().item() + 0.5
    )
    plt.title(f"Epoch {epoch}, MSE: {mse:.4f}")
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        if isinstance(epoch, int):
            epoch_str = f"{epoch:04d}"
        else:
            epoch_str = str(epoch)
        plt.savefig(f"{outdir}/epoch_{epoch_str}.png")
    plt.close()


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "ccc"
augment_with_rotations = False
augment_with_permutations = True

r0 = 2.0
theta0 = 120.0
dataset = ToyMoleculeDataset(
    name=dataset_name,
    n_samples=1000,
    T=0,
    seed=42,
    r0=r0,
    theta0=theta0,
    augment_with_rotations=augment_with_rotations,
    augment_with_permutations=augment_with_permutations,
)
pos_dataset = torch.stack([data.pos for data in dataset])
dataloader = GeometricDataLoader(
    dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42)
)

model_type = "painn"
aligned = True
permuted = True
brute_force_permutations = True
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

use_2d_drifting = True

outdir = f"runs/toy_molecule/dataset_{dataset_name}/{model_type}/augment_rot_{augment_with_rotations}_augment_perm_{augment_with_permutations}/aligned_{aligned}_permuted_{permuted}_brute_force_{brute_force_permutations}"

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
n_samples = None  # just sample as many samples as batch size
n_samples = 1000
pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y = batch.pos.clone()

        # Sample noise (2d and add zero z-component)
        z = sample_noise_like_2d(batch.pos, batch.batch)
        batch.pos = z

        x = model(batch)

        if x[..., 2].abs().max() > 1e-4:
            print(
                "Warning: Non-zero z-component in model prediction, which should be zero for 2D data."
            )

        # Get drifting field in 2D as 3D rotations could include reflections in 2D which are not valid
        if use_2d_drifting:
            V, *_ = drifting_field(
                x[..., :2],
                y[..., :2],
                x[..., :2],
                batch.num_atoms[0],
                atomic_numbers=batch.x,
                temperature=None,
                aligned=None,
                permuted=None,
                brute_force_permutations=None,
            )
            V = torch.cat([V, torch.zeros_like(V[..., :1])], dim=-1)
        else:
            V, *_ = drifting_field(
                x,
                y,
                x,
                batch.num_atoms[0],
                atomic_numbers=batch.x,
                temperature=None,
                aligned=None,
                permuted=None,
                brute_force_permutations=None,
            )
            if V[..., 2].abs().max() > 1e-4:
                print(
                    "Warning: Non-zero z-component in drift field, which should be zero for 2D data."
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

visualize(model, batch, current_epoch="final", n_samples=n_samples, outdir=outdir)
torch.save({"state_dict": model.state_dict()}, f"{outdir}/final_model.pt")
