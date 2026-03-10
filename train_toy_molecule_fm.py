import os
from functools import partial

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.alignment import (
    brute_force_and_kabch_batched,
    get_rmsd_batched,
    hungarian_and_kabch_batched,
)
from tspath.analysis import evaluate_toy
from tspath.datasets import ToyMoleculeDataset
from tspath.generative import CondOTScheduler
from tspath.model import EGNN, PaiNN
from tspath.utils import sample_noise_like_2d


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
    x_reshaped = batch.x.view(B, n_atoms)  # (B, n_atoms)
    x_expanded = x_reshaped.unsqueeze(1).repeat(
        1, n_samples, 1
    )  # (B, n_samples, n_atoms)
    x_expanded = x_expanded.reshape(B * n_samples * n_atoms)

    z = sample_noise_like_2d(dummy, batch_)

    batch_sampling.pos = z
    batch_sampling.batch = batch_
    batch_sampling.x = x_expanded
    batch_sampling.num_atoms = num_atoms
    return batch_sampling


@torch.no_grad()
def sample(model, scheduler, num_steps, batch, n_samples):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()

    # Sample prior noise n_samples * B
    batch_sampling = create_batch_object(batch, n_samples)

    # generate samples
    x, _ = scheduler.sample(
        batch_sampling.pos, num_steps=num_steps, model=model, batch=batch_sampling
    )

    if was_training:
        model.train()

    return x.cpu().numpy()


def visualize(
    model, scheduler, num_steps, batch, current_step, n_samples=None, outdir=None
):
    step = current_step
    x_samples = sample(model, scheduler, num_steps, batch, n_samples=n_samples)
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
        plt.savefig(f"{outdir}/step_{step_str}.png")
    plt.close()


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
aligned = True
permuted = True
brute_force_permutations = False
model_dict = {
    "painn": PaiNN(
        use_noise_schedule_sigma_encoding=True, sphere_channels=256, num_layers=9
    ),
    "egnn": EGNN(use_noise_schedule_sigma_encoding=True),
}
model = model_dict[model_type]
model.to(device)


outdir = f"runs/toy_molecule/dataset_{dataset_name}/{model_type}/fm_baseline/aligned_{aligned}_permuted_{permuted}_brute_force_{brute_force_permutations}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

flow_scheduler = CondOTScheduler(sigma=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)


batch_size = 64
dataloader = GeometricDataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
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

        # Target is the clean data
        x1 = batch.pos.clone()
        batch.pos_orig = x1.clone()

        # Sample noise for each sample
        batch_neg = batch.clone()
        x0 = sample_noise_like_2d(batch.pos, batch.batch)

        # Get initial RMSD before alignment and permutation
        x1_ = x1.view(batch.num_graphs, -1, x1.shape[-1])[..., :2]  # 2d problem
        x0_ = x0.view(batch.num_graphs, -1, x0.shape[-1])[..., :2]  # 2d problem
        rmsd_orig = get_rmsd_batched(x0_, x1_)

        # Align and permute if specified
        if aligned and permuted:
            atomic_numbers = batch.x.view(batch.num_graphs, -1)
            if brute_force_permutations:
                x1_aligned, _ = brute_force_and_kabch_batched(x0_, x1_, atomic_numbers)
            else:
                x1_aligned, _ = hungarian_and_kabch_batched(x0_, x1_, atomic_numbers)
            rmsd_aligned = get_rmsd_batched(x0_, x1_aligned)
            x1_aligned = x1_aligned.view(-1, x1_.shape[-1])
            x1_aligned = torch.cat(
                [x1_aligned, torch.zeros_like(x1_aligned[..., :1])], dim=-1
            )  # add zero z-component
        else:
            x1_aligned = x1
            rmsd_aligned = rmsd_orig

        xt, t, v_target = flow_scheduler.sample_time_and_conditional_path(
            x0, x1_aligned, batch=batch.batch
        )
        rmsd_xt = get_rmsd_batched(
            x1_aligned.view(batch.num_graphs, -1, x1.shape[-1]),
            xt.view(batch.num_graphs, -1, xt.shape[-1]),
        )

        batch.pos = xt
        batch.t = t

        # Call the model
        v_pred = model(batch)

        x1_pred = xt + v_pred * (1 - t)
        rmsd_x1_pred = get_rmsd_batched(
            x1_aligned.view(batch.num_graphs, -1, x1.shape[-1]),
            x1_pred.view(batch.num_graphs, -1, x1_pred.shape[-1]),
        )

        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(
            {
                "step": step_count,
                "loss": loss.item(),
                "rmsd_orig": rmsd_orig.mean().item(),
                "rmsd_aligned": rmsd_aligned.mean().item(),
                "rmsd_xt": rmsd_xt.mean().item(),
                "rmsd_x1_pred": rmsd_x1_pred.mean().item(),
            }
        )

        if step_count % (n_steps // 10) == 0 and step_count > 0:
            n_samples = 1000 // batch.num_graphs
            visualize(
                model,
                flow_scheduler,
                num_steps=1,
                batch=batch,
                current_step=step_count,
                n_samples=n_samples,
                outdir=outdir,
            )
        step_count += 1

    scheduler.step()
    # if (epoch < 101 and epoch % 5 == 0) or (epoch>100 and epoch % 20 == 0):
    #     torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/epoch_{epoch}.pt")

visualize(
    model,
    flow_scheduler,
    num_steps=1,
    batch=batch,
    current_step="final",
    n_samples=n_samples,
    outdir=outdir,
)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")
