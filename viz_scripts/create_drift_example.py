import matplotlib.pyplot as plt
import torch
from tspath.datasets import ToyDataset
from tspath.generative import DriftingField

torch.manual_seed(10)
dataset = ToyDataset(name="spiral", n_samples=1000, noise=0.001, n_arms=2, seed=42)
y = torch.from_numpy(dataset.data)
idx = torch.randperm(y.shape[0])[:5]
y = y[idx]

z = torch.randn_like(y)
drift = DriftingField(temperature=0.15)

V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = drift(z, y, z)

# plot one current sample
plt.scatter(z[0, 0], z[0, 1], s=25, color="black", label="$x$")

# plot all positives samples
plt.scatter(y[:, 0], y[:, 1], s=10, color="tab:blue", label="$\mathbf{y}^{+}$")

# plot all negative samples (except the current sample)
plt.scatter(z[1:, 0], z[1:, 1], s=10, color="tab:orange", label="$\mathbf{y}^{-}$")

# plot drift to each positive samples
plt.quiver(
    z[0,0].repeat(diff_pos.shape[1]), z[0,1].repeat(diff_pos.shape[1]),
    diff_pos[0,:,0], diff_pos[0,:,1],
    angles='xy', scale_units='xy', scale=1, color='tab:blue', alpha=0.25,
    width=0.005,
    headwidth=3,
)

plt.quiver(
    z[0,0], z[0,1],
    drift_pos[0,0], drift_pos[0,1],
    angles='xy', scale_units='xy', scale=1, color='tab:blue', alpha=0.8, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=r"$V_p^+ (\mathbf{x})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
    headlength=3,
)

# plot drift to each negative samples
plt.quiver(
    z[0,0].repeat(diff_neg.shape[1]), z[0,1].repeat(diff_neg.shape[1]),
    diff_neg[0,:,0], diff_neg[0,:,1],
    angles='xy', scale_units='xy', scale=1, color='tab:orange', alpha=0.25,
    width=0.005,
    headwidth=3,
)

plt.quiver(
    z[0,0], z[0,1],
    drift_neg[0,0], drift_neg[0,1],
    angles='xy', scale_units='xy', scale=1, color='tab:orange', alpha=0.8, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=r"$V_p^- (\mathbf{x})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
)

plt.quiver(
    z[0,0], z[0,1],
    V[0,0], V[0,1],
    angles='xy', scale_units='xy', scale=1, color='black', alpha=0.8, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=r"$V_{p,q} (\mathbf{x})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
)

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.savefig("drift_pos_example.png", dpi=300)
# plt.gca().set_aspect("equal", adjustable="box")