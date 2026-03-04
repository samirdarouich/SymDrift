
import torch
from tspath.model import EGNN
from tspath.datasets import ToyMoleculeDataset
from tspath.generative import EquivariantDriftingField
from tspath.utils import sample_noise_like_2d
from torch_geometric.data import Data
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_one_molecule(positions, color, idx, label, ax, alpha=1.0, plot_bonds=True, noise=False, annotate=False):
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
    if annotate:
        for i in range(x0.shape[0]):
            ax.text(x0[i] + 0.03, y0[i] + 0.03, f"{i + 1}", color="black", fontsize=15)
 
@torch.no_grad()
def sample(model, n_samples):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    model.eval()
    device = next(model.parameters()).device

    # Sample prior noise and prepare batch data
    atomic_numbers = torch.tensor([6, 6, 6], device=device)  # C3 molecule
    n_atoms = atomic_numbers.shape[0]

    batch_ = torch.arange(n_samples, device=device).repeat_interleave(n_atoms)
    dummy = torch.zeros((n_samples * n_atoms, 2), dtype=torch.float, device=device)
    x = atomic_numbers.repeat(n_samples)
    num_atoms = torch.tensor([n_atoms] * n_samples, device=device)
    z = sample_noise_like_2d(dummy, batch_)

    batch_sampling = Data()
    batch_sampling.pos = z
    batch_sampling.batch = batch_
    batch_sampling.x = x
    batch_sampling.num_atoms = num_atoms

    # generate samples
    x = model(batch_sampling)

    return x[:, :2] # Only keep 2D coordinates

dataset = ToyMoleculeDataset(
    name="ccc",
    n_samples=500,
    T=0,
    seed=42,
    r0=2.0,
    theta0=120.0,
    augment_with_rotations=True,
    augment_with_permutations=False,
)
# Only keep 2D
dataset_samples = torch.stack([data.pos[:, :2] for data in dataset]).to(device)

torch.manual_seed(42)
model = EGNN()
model.to(device)

# ckpt = "toy_molecule/dataset_ccc/egnn/augment_rot_False_augment_perm_False/aligned_False_permuted_False_brute_force_False/final_model.pt"
# ckpt = "toy_molecule/dataset_ccc/egnn/augment_rot_False_augment_perm_False/aligned_True_permuted_True_brute_force_False/final_model.pt"
# ckpt = "toy_molecule/dataset_ccc/egnn/augment_rot_True_augment_perm_False/aligned_True_permuted_True_brute_force_False/final_model.pt"
ckpt = "toy_molecule/dataset_ccc/egnn/augment_rot_False_augment_perm_False/aligned_True_permuted_True_brute_force_True/final_model.pt"
ckpt_ = torch.load(ckpt, weights_only=False)
model.load_state_dict({k.replace("model.", ""): v for k, v in ckpt_["state_dict"].items()})
    
sample_ = sample(model, n_samples=500).view(-1, 3, 2)

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    dataset_samples[:, :, 0].cpu(), dataset_samples[:, :, 1].cpu(), alpha=0.75, color="tab:blue", label="Dataset", s=10
)

ax.scatter(
    sample_[:, :, 0].cpu(), sample_[:, :, 1].cpu(), alpha=0.5, color="tab:red", label="Sample", s=10
)


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
dataset_samples = torch.stack([data.pos[:, :2] for data in dataset]).to(device)
plot_one_molecule(
    dataset_samples.cpu(), color="tab:blue", idx=0, label="", ax=ax, alpha=0.75, plot_bonds=True, noise=False, annotate=True
)

ax.legend(loc="upper right", fontsize=15)
ax.set_aspect("equal", adjustable="box")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(ckpt), "samples.png"))
plt.close()