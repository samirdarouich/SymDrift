
import torch
from tspath.model import MLP
from tspath.datasets import ToyDataset
from tspath.generative import DriftingField
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@torch.no_grad()
def sample(model, n_samples):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    model.eval()
    device = next(model.parameters()).device

    z = torch.randn(n_samples, 2, device=device)

    # generate samples
    x = model(z)

    return x

ckpt_dir = "../runs/toy_spiral/normed_training/checkpoints"
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and "epoch" in f], key=lambda x: int(x.split("epoch=")[1].split("-")[0]))

dataset = ToyDataset(n_samples=500, seed=42, n_arms=2, noise=0.001)
dataset_samples = torch.stack([dataset[i] for i in range(len(dataset))]).to(device)
drift = DriftingField(temperatures=torch.tensor([0.15]))

torch.manual_seed(42)
model = MLP()
model.to(device)

sample_ = sample(model, n_samples=500)
V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(sample_, dataset_samples, sample_)
samples = [sample_]
drifts = [(V, drift_pos[0], drift_neg[0], w_pos[0], w_neg[0])]
training_iteration = [0]

for ckpt in ckpts:
    ckpt_ = torch.load(os.path.join(ckpt_dir, ckpt), weights_only=False)
    model.load_state_dict({k.replace("model.", ""): v for k, v in ckpt_["state_dict"].items()})
    sample_ = sample(model, n_samples=500)
    
    V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(sample_, dataset_samples, sample_)

    samples.append(sample_)
    # just keep the first temperature (everything except of V has the T dimension)
    drifts.append((V, drift_pos[0], drift_neg[0], w_pos[0], w_neg[0]))
    training_iteration.append(ckpt_["epoch"]+1)

dataset_samples = dataset_samples.cpu()
traj = torch.stack(samples).cpu()
Vs = torch.stack([d[0] for d in drifts]).cpu()
drift_pos = torch.stack([d[1] for d in drifts]).cpu()
drift_neg = torch.stack([d[2] for d in drifts]).cpu()
w_pos = torch.stack([d[3] for d in drifts]).cpu()
w_neg = torch.stack([d[4] for d in drifts]).cpu()

save_dir = ckpt_dir.replace("checkpoints", "trajectory_plots_normed")
save_dir = ckpt_dir.replace("checkpoints", "trajectory_plots")
os.makedirs(save_dir, exist_ok=True)
for i in range(traj.shape[0]):
    
    idx_sample = 5
    x0 = traj[i, idx_sample, 0].item()
    y0 = traj[i, idx_sample, 1].item()
    
    ### Sample evolution plot
    
    # plot one current sample
    plt.scatter(x0, y0, s=25, color="black", label="$x$")
    plt.scatter(traj[i, :, 0], traj[i, :, 1], alpha=0.25, color="tab:orange", label="$\mathbf{y}^{-}$", s=10)
    plt.scatter(dataset_samples[:, 0], dataset_samples[:, 1], alpha=0.25, color="tab:blue", label="$\mathbf{y}^{+}$", s=7)
    #plt.plot(traj[:i,:no_samples,0], traj[:i,:no_samples,1], alpha=0.5, color="tab:orange", linewidth=1)
    plt.title(f"Training Iteration {training_iteration[i]}", fontweight="bold")
    plt.legend(loc="upper right")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"trajectory_sample_{i:02d}.png"))
    plt.show()
    plt.close()
    
    ### Drift field plot
    
    # plot one current sample
    plt.scatter(x0, y0, s=25, color="black", label="$x$")
    
    # plot all positives samples
    plt.scatter(dataset_samples[:, 0], dataset_samples[:, 1], alpha=0.25, s=10, color="tab:blue", label="$\mathbf{y}^{+}$")

    # plot all negative samples (except the current sample)
    traj_neg = torch.cat([traj[i, :idx_sample, 0], traj[i, (idx_sample+1):, 0]])
    traj_neg_y = torch.cat([traj[i, :idx_sample, 1], traj[i, (idx_sample+1):, 1]])
    plt.scatter(traj_neg, traj_neg_y, alpha=0.25, color="tab:orange", label="$\mathbf{y}^{-}$", s=10)
    
    # highlight the 10 most important samples (for the first temperature)
    w_pos_i_sample = w_pos[i][idx_sample]
    w_neg_i_sample = w_neg[i][idx_sample]
    topk_pos = torch.topk(w_pos_i_sample, k=10).indices
    topk_neg = torch.topk(w_neg_i_sample, k=10).indices
    
    plt.scatter(dataset_samples[topk_pos, 0], dataset_samples[topk_pos, 1], alpha=0.75, s=10, color="tab:blue", edgecolor="black")
    plt.scatter(traj[i, topk_neg, 0], traj[i, topk_neg, 1], alpha=0.75, s=10, color="tab:orange", edgecolor="black")
    
    # Plot positive drift vector
    plt.quiver(
        x0, y0,
        drift_pos[i, idx_sample, 0], drift_pos[i, idx_sample, 1],
        angles='xy', scale_units='xy', scale=1, color='tab:blue', alpha=0.8, 
        # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
        label=r"$V_p^+ (\mathbf{x})$",
        width=0.007,          # thinner shaft
        headwidth=3,          # smaller arrow head width
    )
    
    # Plot negative drift vector
    plt.quiver(
        x0, y0,
        drift_neg[i, idx_sample, 0], drift_neg[i, idx_sample, 1],
        angles='xy', scale_units='xy', scale=1, color='tab:orange', alpha=0.8, 
        # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
        label=r"$V_p^- (\mathbf{x})$",
        width=0.007,          # thinner shaft
        headwidth=3,          # smaller arrow head width
    )

    # Plot total drift vector
    plt.quiver(
        x0, y0,
        Vs[i, idx_sample, 0], Vs[i, idx_sample, 1],
        angles='xy', scale_units='xy', scale=1, color='black', alpha=0.8, 
        # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
        label=r"$V_{p,q} (\mathbf{x})$",
        width=0.007,          # thinner shaft
        headwidth=3,          # smaller arrow head width
    )
    
    
    plt.title(f"Training Iteration {training_iteration[i]}", fontweight="bold")
    plt.legend(loc="upper right")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    
    # safe overall plot
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"trajectory_drift_{i:02d}.png"))
    

    # safe zoomed version
    delta = 0.5  # zoom window size
    plt.xlim(x0 - delta, x0 + delta)
    plt.ylim(y0 - delta, y0 + delta)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"trajectory_drift_zoom_{i:02d}.png"))

    plt.show()
    plt.close()
    