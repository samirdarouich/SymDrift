import torch
from tspath.datasets import ToyMoleculeDataset
from tspath.generative import EquivariantDriftingField
import matplotlib.pyplot as plt

def plot_one_molecule(positions, color, idx, label, ax, alpha=1.0, plot_bonds=True, noise=False):
    # Plot the original and rotated positions
    x0, y0 = positions[idx, :, 0], positions[idx, :, 1]
    ax.scatter( x0, y0, alpha=alpha, color=color, label=label,)
    if plot_bonds:
        
        if noise:
            for i,j in [(0,1), (1,2)]:
                ax.plot(
                        [x0[i].item(), x0[j].item()],
                        [y0[i].item(), y0[j].item()],
                        linestyle="--",
                        linewidth=1.2,
                        color=color,
                        alpha=alpha * 0.9,
                        zorder=0,
                    )
        else:
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

def plot_drift_field(positions, drift_field, color, idx, label, ax, drift_color=None, target_label=None, alpha=1.0, plot_target=True):
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
    
    if plot_target:
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

n_noise_samples = 10

torch.manual_seed(42)
z = torch.randn(n_noise_samples, 3, 2)
z -= z.mean(dim=1, keepdim=True)

pos_dataset_ = pos_dataset.view(-1, 2)
z_ = z.view(-1, 2)


drift = EquivariantDriftingField(
    temperature=0.15, aligned=True, permuted=True, brute_force_permutations=True
)

V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(
    z_, pos_dataset_, z_, n_atoms=3,
    aligned=False, permuted=False, brute_force_permutations=False
)

fig, ax = plt.subplots(figsize=(18, 6))
plot_one_molecule(z, "black", 3, "$x_0$", ax[0], alpha=0.5, noise=True)
plot_drift_field(
    # z, V.view(-1, 3, 2), "red", 3, 
    z, drift_pos, "red", 3, 
    "$V=w(x,y)(y-x)$", ax[0], drift_color="black",
    target_label="$y$", alpha=0.5, plot_target=False
)
ax[0].legend(loc="upper left", fontsize=15)
ax[0].set_xlim(-3,3)
ax[0].set_ylim(-3,3)
ax[0].set_xlabel("")
ax[0].set_ylabel("")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_aspect("equal", adjustable="box")


drift = EquivariantDriftingField(
    temperature=0.15, aligned=True, permuted=True, brute_force_permutations=True
)

V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(
    z_, pos_dataset[0], z_, n_atoms=3,
    aligned=True, permuted=True, brute_force_permutations=True
)

plot_one_molecule(z, "black", 3, "$x_0$", ax[1], alpha=0.5, noise=True)
plot_drift_field(
    z, V.view(-1, 3, 2), "red", 3, 
    # z, drift_neg, "red", 3, 
    r"$V=w(x,\tilde{y})(\tilde{y}-x), \quad \tilde{y} = R P y $", 
    ax[1], drift_color="black",
    target_label=r"$\tilde{y}$", alpha=0.5, plot_target=False
)
ax[1].legend(loc="upper left", fontsize=15)
ax[1].set_xlim(-3,3)
ax[1].set_ylim(-3,3)
ax[1].set_xlabel("")
ax[1].set_ylabel("")
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_aspect("equal", adjustable="box")

for i in range(z.shape[0]):
    if i == 0:
        label = r"$x_i \in \mathbf{x}$"
    else:
        label = ""
    plot_one_molecule(z, None, i, label, ax[2], alpha=0.25, noise=True)
plot_one_molecule(pos_dataset, "black", 0, "$y^{+}$", ax[2], alpha=1.0)
ax[2].legend(loc="upper left", fontsize=15)
ax[2].set_xlim(-3,3)
ax[2].set_ylim(-3,3)
ax[2].set_xlabel("")
ax[2].set_ylabel("")
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_aspect("equal", adjustable="box")


plt.tight_layout()
plt.savefig("aware_drift_samples.png")