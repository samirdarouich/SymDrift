import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tspath.datasets import CompositionBatchSampler, MoleculeDataset
from tspath.generative import EquivariantDriftingField
from tspath.utils import batch_inputs_to_atoms, sample_noise_like


def plot_drifting_field(x, V_pos, V_neg, atom_idx=None):
    coords = torch.stack(
        [x[..., :2], (x + V_pos)[..., :2], (x + V_neg)[..., :2]], dim=0
    )
    x_min = coords[..., 0].min().item()
    x_max = coords[..., 0].max().item()
    y_min = coords[..., 1].min().item()
    y_max = coords[..., 1].max().item()
    V = V_pos - V_neg
    if atom_idx is None:
        atom_idx = slice(None)
    n_per_row = 3
    rows = int(np.ceil(x.shape[0] / n_per_row))
    fig, ax = plt.subplots(rows, n_per_row, figsize=(5 * n_per_row, 5 * rows))
    for i in range(x.shape[0]):
        ax_ = ax[i // n_per_row, i % n_per_row]
        ax_.scatter(
            x[i, atom_idx, 0].cpu().numpy(),
            x[i, atom_idx, 1].cpu().numpy(),
            color="black",
        )
        ax_.quiver(
            x[i, atom_idx, 0].cpu().numpy(),
            x[i, atom_idx, 1].cpu().numpy(),
            V_pos[i, atom_idx, 0].cpu().numpy(),
            V_pos[i, atom_idx, 1].cpu().numpy(),
            color="red",
            scale=1,
            scale_units="xy",
        )
        ax_.quiver(
            x[i, atom_idx, 0].cpu().numpy(),
            x[i, atom_idx, 1].cpu().numpy(),
            V_neg[i, atom_idx, 0].cpu().numpy(),
            V_neg[i, atom_idx, 1].cpu().numpy(),
            color="blue",
            scale=1,
            scale_units="xy",
        )
        ax_.quiver(
            x[i, atom_idx, 0].cpu().numpy(),
            x[i, atom_idx, 1].cpu().numpy(),
            V[i, atom_idx, 0].cpu().numpy(),
            V[i, atom_idx, 1].cpu().numpy(),
            color="green",
            scale=1,
            scale_units="xy",
        )
        ax_.set_xlim(x_min - 0.5, x_max + 0.5)
        ax_.set_ylim(y_min - 0.5, y_max + 0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_aligned_distribution(x, diff_pos, diff_neg, plot="both", atom_idx=None):
    x_ref = x[:, None, :, :]  # (N, 1, n_atoms, 3)
    coords = [x_ref]
    if plot in ("both", "pos"):
        coords.append(x_ref + diff_pos)
    if plot in ("both", "neg"):
        coords.append(x_ref + diff_neg)
    coords = torch.cat(coords, dim=1)  # (N, *, n_atoms, 3)
    x_min = coords[..., 0].amin().item()
    x_max = coords[..., 0].amax().item()
    y_min = coords[..., 1].amin().item()
    y_max = coords[..., 1].amax().item()

    if atom_idx is None:
        atom_idx = slice(None)
    n_per_row = 3
    rows = int(np.ceil(x.shape[0] / n_per_row))
    fig, ax = plt.subplots(rows, n_per_row, figsize=(5 * n_per_row, 5 * rows))
    for i in range(x.shape[0]):
        ax_ = ax[i // n_per_row, i % n_per_row]
        ax_.scatter(
            x[i, atom_idx, 0].cpu().numpy(),
            x[i, atom_idx, 1].cpu().numpy(),
            color="black",
        )

        for j in range(diff_pos.shape[1]):
            if plot == "both" or plot == "pos":
                ax_.quiver(
                    x[i, atom_idx, 0].cpu().numpy(),
                    x[i, atom_idx, 1].cpu().numpy(),
                    diff_pos[i, j, atom_idx, 0].cpu().numpy(),
                    diff_pos[i, j, atom_idx, 1].cpu().numpy(),
                    color="red",
                    alpha=0.5,
                    scale=1,
                    scale_units="xy",
                )
            if plot == "both" or plot == "neg":
                ax_.quiver(
                    x[i, atom_idx, 0].cpu().numpy(),
                    x[i, atom_idx, 1].cpu().numpy(),
                    diff_neg[i, j, atom_idx, 0].cpu().numpy(),
                    diff_neg[i, j, atom_idx, 1].cpu().numpy(),
                    color="blue",
                    alpha=0.5,
                    scale=1,
                    scale_units="xy",
                )
        ax_.set_xlim(x_min - 0.5, x_max + 0.5)
        ax_.set_ylim(y_min - 0.5, y_max + 0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


dataset = MoleculeDataset(
    # source="t1x_eq_all_structures",
    # source="t1x_eq_C5H8O",
    source="t1x_eq_CHN3O",
    root="/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq",
    split="train",
    # split_identifier="debug",
)

sampler = CompositionBatchSampler(
    dataset, k=1, n=10, shuffle=True, drop_last=False, resample=False, seed=42
)
dataloader = GeometricDataLoader(
    dataset, batch_sampler=sampler, num_workers=4, persistent_workers=True
)
driting_field_equivariant = EquivariantDriftingField(
    temperature=1.0, mask_self=True, normalize_over_x=False, aligned=True
)

driting_field_naive = EquivariantDriftingField(
    temperature=1.0, mask_self=True, normalize_over_x=False, aligned=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
for batch in dataloader:
    batch = batch.to(device)
    y = batch.pos.clone()
    z = sample_noise_like(batch.pos, batch.batch)

    y_reshaped = y.view(-1, batch.num_atoms[0], 3)
    z_reshaped = z.view(-1, batch.num_atoms[0], 3)

    V, V_pos, V_neg, diff_pos, diff_neg = driting_field_equivariant(
        z,
        y,
        z,
        temperature=1.0,
        n_atoms=batch.num_atoms[0],
    )

    (
        V_unaligned,
        V_pos_unaligned,
        V_neg_unaligned,
        diff_pos_unaligned,
        diff_neg_unaligned,
    ) = driting_field_naive(
        z,
        y,
        z,
        temperature=1.0,
        n_atoms=batch.num_atoms[0],
    )

    atoms = batch_inputs_to_atoms(batch)
    break
