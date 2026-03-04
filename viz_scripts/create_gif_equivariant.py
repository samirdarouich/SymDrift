
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

naive = False
if naive:
    folder = "aligned_False_permuted_False_brute_force_False"
else:
    folder = "aligned_True_permuted_True_brute_force_True"
ckpt_dir = f"runs/toy_molecule/dataset_ccc/egnn/augment_rot_False_augment_perm_False/{folder}/checkpoints"
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and "epoch" in f], key=lambda x: int(x.split("epoch_")[1].split(".")[0]))

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
dataset_samples = torch.stack([data.pos[:, :2] for data in dataset]).to(device).view(-1, 2)
drift = EquivariantDriftingField(temperature=0.15)

torch.manual_seed(42)
model = EGNN()
model.to(device)

sample_ = sample(model, n_samples=500)
V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(sample_, dataset_samples, sample_, n_atoms=3)

samples = [sample_.view(-1, 3, 2)]
drifts = [(V.view(-1, 3, 2), drift_pos, drift_neg, w_pos, w_neg)]
training_iteration = [0]

for ckpt in ckpts:
    ckpt_ = torch.load(os.path.join(ckpt_dir, ckpt), weights_only=False)
    model.load_state_dict({k.replace("model.", ""): v for k, v in ckpt_["state_dict"].items()})
    sample_ = sample(model, n_samples=500)
    
    V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(
        sample_, dataset_samples, sample_, n_atoms=3,
        aligned=not naive, permuted=not naive, brute_force_permutations=not naive
    )

    samples.append(sample_.view(-1, 3, 2))
    drifts.append((V.view(-1, 3, 2), drift_pos, drift_neg, w_pos, w_neg))
    training_iteration.append(int(ckpt.split("epoch_")[1].split(".")[0])+1)

dataset_samples = dataset_samples.cpu()
traj = torch.stack(samples).cpu()
Vs = torch.stack([d[0] for d in drifts]).cpu()
drift_pos = torch.stack([d[1] for d in drifts]).cpu()
drift_neg = torch.stack([d[2] for d in drifts]).cpu()
w_pos = torch.stack([d[3] for d in drifts]).cpu()
w_neg = torch.stack([d[4] for d in drifts]).cpu()

save_dir = ckpt_dir.replace("checkpoints", "trajectory_plots")
os.makedirs(save_dir, exist_ok=True)

for i in range(traj.shape[0]):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    idx_sample = 5
    
    ### Sample evolution plot
    
    # plot one current sample
    plot_one_molecule(traj[i], color="black", idx=idx_sample, label="$x$", ax=ax)
    
    # plot the positive sample
    plot_one_molecule(dataset_samples.unsqueeze(0), color="tab:blue", idx=0, label="$\mathbf{y}^{+}$", ax=ax)
    
    # plot several negative samples (except the current sample)
    traj_neg = torch.cat([traj[i, :idx_sample], traj[i, (idx_sample+1):]])

    for j in range(10):
        label = "$\mathbf{y}^{-}$" if j == 0 else None  # Only label the first negative sample for the legend
        plot_one_molecule(traj_neg, color="tab:red", idx=j, label=label, alpha=0.25, ax=ax)
    
    # plot positive drift field for this sample
    plot_drift_field(
        traj[i], drift_pos[i], color="tab:blue", idx=idx_sample, 
        label=r"$V_p^+ (\mathbf{x})$", ax=ax, alpha=0.8, plot_target=False
    )
    
    # plot negative drift field for this sample
    plot_drift_field(
        traj[i], drift_neg[i], color="tab:orange", idx=idx_sample, 
        label=r"$V_p^- (\mathbf{x})$", ax=ax, alpha=0.8, plot_target=False
    )
    
    # plot total drift field for this sample
    plot_drift_field(
        traj[i], Vs[i], color="black", idx=idx_sample, 
        label=r"$V_{p,q} (\mathbf{x})$", ax=ax, alpha=0.8, plot_target=False
    )
    
    # highlight the 10 most important samples
    w_neg_i_sample = w_neg[i][idx_sample]
    topk_neg = torch.topk(w_neg_i_sample, k=3).indices
    for j in topk_neg:
        plot_one_molecule(traj_neg, color="tab:red", idx=j, label="", alpha=0.75, ax=ax)
    
    
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    plt.title(f"Training Iteration {training_iteration[i]}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"trajectory_drift_{i:02d}.png"))
    plt.close()
    