import matplotlib.pyplot as plt
import torch
from ase import Atoms
from tspath.datasets import ToyMoleculeDataset
from tspath.generative import EquivariantDriftingField


def rotation_matrix_2d(angle_deg):
    theta = torch.deg2rad(torch.tensor(angle_deg, device=device))
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]
    )

def plot_one_molecule(positions, color, idx, label, ax, alpha=1.0, plot_bonds=True):
    # Plot the original and rotated positions
    x0, y0 = positions[idx, :, 0], positions[idx, :, 1]
    ax.scatter( x0, y0, alpha=alpha, color=color, label=label,)
    if plot_bonds:
        n_atoms = x0.shape[0]
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = torch.sqrt((x0[i] - x0[j]) ** 2 + (y0[i] - y0[j]) ** 2)
                if d <= 2.1:
                    ax.plot(
                        [x0[i].item(), x0[j].item()],
                        [y0[i].item(), y0[j].item()],
                        linestyle="--",
                        linewidth=1.2,
                        color=color,
                        alpha=alpha * 0.9,
                        zorder=0,
                    )
    for i in range(x0.shape[0]):
        ax.text(x0[i] + 0.03, y0[i] + 0.03, f"{i + 1}", color="black", fontsize=15)

def plot_drift_field(positions, drift_field, color, idx, label, ax, drift_color=None, target_label=None, alpha=1.0):
    x0, y0 = positions[idx, :, 0], positions[idx, :, 1]
    if drift_color is None:
        drift_color = color
    ax.quiver(
        x0,
        y0,
        drift_field[idx, :, 0],
        drift_field[idx, :, 1],
        color=drift_color,
        label=label,
        alpha=alpha*0.75,
        angles="xy",
        scale_units="xy",
        scale=1.0,
    )
    target = positions + drift_field
    
    plot_one_molecule(target, color, idx, target_label, ax, alpha=alpha)

device = "cpu"
dataset = ToyMoleculeDataset(
    name="ccc",
    n_samples=1,
    T=0,
    seed=42,
    r0=2.0,
    theta0=120.0,
    augment_with_rotations=False,
    augment_with_permutations=False,
)
# Only keep 2D
pos_dataset = torch.stack([data.pos[:, :2] for data in dataset]).to(device)

drift = EquivariantDriftingField(
    temperature=0.15, aligned=True, permuted=True, brute_force_permutations=True
)

atoms = Atoms(
    symbols=["C", "C", "C"],
    positions=dataset[0].pos,
)
atoms.write("original_positions.xyz")

# Set the angle here
angle_deg = 90.0
rotation = rotation_matrix_2d(angle_deg)

# Rotate xy-coordinates of the molecule positions
pos_xy_rot = pos_dataset @ rotation.T

P = [1, 2, 0]  # Example permutation (1->2, 2->0, 0->1)
pos_xy_rot = pos_xy_rot[:, P, :]

V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(
    pos_dataset[0], pos_xy_rot[0], pos_dataset[0], n_atoms=3,
    aligned=False, permuted=False, brute_force_permutations=False
)

fig, ax = plt.subplots(1,3,figsize=(18, 6))

plot_one_molecule(pos_dataset, "black", 0, "$x$", ax[0], alpha=0.5)
plot_one_molecule(pos_dataset, "black", 0, "$x$", ax[1], alpha=0.5)
plot_one_molecule(pos_dataset, "black", 0, "$x$", ax[2], alpha=0.5)
plot_drift_field(pos_dataset, drift_pos, "red", 0, "$V=w(x,y)(y-x)$", ax[0], drift_color="black", target_label="$y$", alpha=0.5)

V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(
    pos_dataset[0], pos_xy_rot[0], pos_dataset[0], n_atoms=3,
    aligned=True, permuted=False, brute_force_permutations=False
)

# alignment aware
plot_drift_field(
    pos_dataset, drift_pos, "red", 0, 
    r"$V=w(x,\tilde{y})(\tilde{y}-x), \quad \tilde{y} = R y $"
    , ax[1], drift_color="black", target_label=r"$\tilde{y}$", alpha=0.25
)

# permutation and alignment aware
V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(
    pos_dataset[0], pos_xy_rot[0], pos_dataset[0], n_atoms=3,
    aligned=True, permuted=True, brute_force_permutations=True
)
plot_drift_field(
    pos_dataset, drift_pos, "red", 0, 
    r"$V=w(x,\tilde{y})(\tilde{y}-x), \quad \tilde{y} = R P y $"
    , ax[2], drift_color="black", target_label=r"$\tilde{y}$", alpha=0.25
)

ax[0].legend(loc="upper left", fontsize=15)
ax[1].legend(loc="upper left", fontsize=15)
ax[2].legend(loc="upper left", fontsize=15)
for ax_ in ax:
    ax_.set_xlabel("")
    ax_.set_ylabel("")
    ax_.set_xticks([])
    ax_.set_yticks([])
    ax_.set_xlim(-3.0, 3.0)
    ax_.set_ylim(-3.0, 3.0)
    ax_.set_aspect("equal", adjustable="box")
plt.tight_layout()

plt.savefig("equivariant_drift_field.png")
plt.show()