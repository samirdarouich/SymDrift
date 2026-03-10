import os
from functools import partial

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.analysis import evaluate_toy
from tspath.datasets import ToyMoleculeDataset
from tspath.generative import DriftingField
from tspath.model import EGNN, PaiNN
from tspath.utils import sample_noise_like_2d
from torch.optim.lr_scheduler import CosineAnnealingLR


def create_batch_object(n_atoms, n_samples):

    batch_sampling = Data()
    # Sample prior noise
    x = torch.tensor([6] * n_atoms, device=device)  # C3 molecule

    batch_ = torch.arange(n_samples, device=device).repeat_interleave(n_atoms)
    dummy = torch.zeros((n_samples * n_atoms), dtype=torch.float, device=device)
    x = x.repeat(n_samples)
    num_atoms = torch.tensor(n_atoms, device=device).repeat(n_samples)
    z = sample_noise_like_2d(dummy, batch_)

    batch_sampling.pos = z
    batch_sampling.batch = batch_
    batch_sampling.x = x
    batch_sampling.num_atoms = num_atoms
    return batch_sampling


@torch.no_grad()
def sample(model, n_atoms, n_samples):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()

    # sample prior noise
    batch_sampling = create_batch_object(n_atoms, n_samples)

    # generate samples
    x = model(batch_sampling)

    if was_training:
        model.train()

    return x.cpu().numpy()


def visualize(model, n_atoms, current_step, n_samples=None, outdir=None):
    step = current_step
    x_samples = sample(model, n_atoms=n_atoms, n_samples=n_samples)
    mse = evaluation_function(torch.tensor(x_samples))
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
        pos_dataset[:, :, 0].min().item() - 1.0, pos_dataset[:, :, 0].max().item() + 1.0
    )
    plt.ylim(
        pos_dataset[:, :, 1].min().item() - 1.0, pos_dataset[:, :, 1].max().item() + 1.0
    )
    plt.title(f"Step {step}, MSE: {mse:.4f}")
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        if isinstance(step, int):
            step_str = f"{step:04d}"
        else:
            step_str = str(step)
        plt.savefig(f"{plot_dir}/step_{step_str}.png")
    plt.close()

def invariant_distance_embedder(xs, ys, atomic_numbers):
    """
    xs, ys: (B, N, 3)
    atomic_numbers: (B, N)

    returns:
        scalar loss
    """

    B, N, _ = xs.shape
    i, j = torch.triu_indices(N, N, offset=1, device=device)
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

            # flatten pair distances (This only works for homogenous systems, otherwise contsructt indices for N_pairs of each type)
            dx = Dx[pair_mask].view(B,N,N)[:,i,j]
            dy = Dy[pair_mask].view(B,N,N)[:,i,j]

            dx = torch.sort(dx, dim=1)[0]
            dy = torch.sort(dy, dim=1)[0]

            dxs.append(dx)
            dys.append(dy)

    dxs = torch.cat(dxs, dim=1)
    dys = torch.cat(dys, dim=1)

    return dxs, dys


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "carbon_chain"

n_atoms = 8
r0 = 2.0
theta0 = 120.0
factor = 1.25

n_samples = 1000
dataset = ToyMoleculeDataset(
    name=dataset_name,
    n_samples=n_samples,
    T=0,
    seed=42,
    r0=r0,
    theta0=theta0,
    factor=factor,
    n_atoms=n_atoms,
)
evaluation_function = partial(
    evaluate_toy,
    dataset_name=dataset_name,
    n_atoms=n_atoms,
    r0=r0,
    theta0=theta0,
    factor=factor,
)

if dataset_name == "carbon_chain":
    dataset_name += f"_n_atoms_{n_atoms}"

pos_dataset = torch.stack([data.pos for data in dataset])
n_atoms = pos_dataset.shape[1]

model_type = "painn"
model_dict = {
    "painn": PaiNN(sphere_channels=256,num_layers=9,),
    "egnn": EGNN(num_distance_basis=0),
}
model = model_dict[model_type]
model.to(device)


normalize_drift = True
temperatures = [0.05]
temp_str = "_".join([f"{t:.2f}" for t in temperatures])
outdir = f"runs/toy_molecule/dataset_{dataset_name}/{model_type}/temp_{temp_str}/norm_{normalize_drift}/embedder_distance"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

drifting_field = DriftingField(
    temperatures=temperatures,
    normalize_drift=normalize_drift,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

losses = []
model.train()

n_steps = 9000 * 2
batch_size = 64
dataloader = GeometricDataLoader(
    dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42)
)

# if just one y is used, then we overall train less steps, as steps = n_epochs * batch size
# in the augmented case we have more samples and thus more steps, use less epochs
n_epochs = n_steps // len(dataloader)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
n_samples = 1000
pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
step_count = 0
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y = batch.pos.clone()

        if batch.num_graphs == 1:
            # sample more negative than positive (save comp. effort for alignment and permutation)
            batch_neg = create_batch_object(n_atoms=n_atoms, n_samples=batch_size)
        else:
            batch_neg = batch.clone()
            # Sample noise (2d and add zero z-component)
            z = sample_noise_like_2d(batch.pos, batch.batch)
            batch_neg.pos = z

        x = model(batch_neg)

        if x[..., 2].abs().max() > 1e-4:
            print(
                "Warning: Non-zero z-component in model prediction, which should be zero for 2D data."
            )

        # Encode samples and targets
        x_embedded, y_embedded = invariant_distance_embedder(
            x.view(batch.num_graphs, -1, 3), 
            y.view(batch.num_graphs, -1, 3), 
            batch.x.view(batch.num_graphs, -1)
        )

        # Call the drift
        V, drift_pos, drift_neg, *_ = drifting_field(
            x_embedded.detach(),
            y_embedded,
            x_embedded.detach(),
        )

        x_drifted = (x_embedded + V).detach()

        loss = torch.nn.functional.mse_loss(x_embedded, x_drifted)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(
            {
                "step": step_count, 
                "mse(V)": torch.sqrt(torch.mean(V**2)).item(),
                "mse(pos_drift)": torch.sum(torch.sqrt(torch.mean(drift_pos**2, dim=(1, 2)))).item(),
                "mse(neg_drift)": torch.sum(torch.sqrt(torch.mean(drift_neg**2, dim=(1, 2)))).item(),
            }
        )

        if step_count % (n_steps // 10) == 0 and step_count > 0:
            visualize(
                model,
                current_step=step_count,
                n_atoms=n_atoms,
                n_samples=n_samples,
                outdir=outdir,
            )

        step_count += 1
        
    
    scheduler.step()
visualize(
    model, current_step="final", n_atoms=n_atoms, n_samples=n_samples, outdir=outdir
)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")
