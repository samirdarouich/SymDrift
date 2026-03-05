import matplotlib.pyplot as plt
import torch
from tspath.datasets import ToyDataset
from tspath.generative import DriftingField

torch.manual_seed(10)
dataset = ToyDataset(name="spiral", n_samples=1000, noise=0.001, n_arms=2, seed=42)
y = torch.from_numpy(dataset.data)
idx = torch.randperm(y.shape[0])[:5]
idx = [0,30, 60, 90, 120]
y = y[idx]

z = torch.randn_like(y)
drift = DriftingField(temperatures=torch.tensor([0.05, 0.25]))

plt.figure(figsize=(8, 8))
# circle around z[0], with raidus such that exp(-dist/tau) ~0
circle = plt.Circle(
    (z[0, 0].item(), z[0, 1].item()),
    float(drift.temperatures[0] * torch.log(torch.tensor(1e-3))),
    fill=False,
    linestyle="--",
    linewidth=1.5,
    color="gray",
    alpha=0.8,
    label=rf"k(x,y,$\tau={drift.temperatures[0]:.2f}$)~0",
)
plt.gca().add_patch(circle)

circle = plt.Circle(
    (z[0, 0].item(), z[0, 1].item()),
    float(drift.temperatures[1] * torch.log(torch.tensor(1e-3))),
    fill=False,
    linestyle="--",
    linewidth=1.5,
    color="gray",
    alpha=0.8,
    label=rf"k(x,y,$\tau={drift.temperatures[1]:.2f}$)~0",
)
plt.gca().add_patch(circle)

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
    drift_pos[0,0,0], drift_pos[0,0,1],
    angles='xy', scale_units='xy', scale=1, color='tab:blue', alpha=0.8, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=rf"$V_p^+ (\mathbf{{x}},\tau={drift.temperatures[0]:.2f})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
    headlength=3,
)

plt.quiver(
    z[0,0], z[0,1],
    drift_pos[1,0,0], drift_pos[1,0,1],
    angles='xy', scale_units='xy', scale=1, color='tab:blue', alpha=0.5, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=rf"$V_p^+ (\mathbf{{x}},\tau={drift.temperatures[1]:.2f})$",
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
    drift_neg[0,0,0], drift_neg[0,0,1],
    angles='xy', scale_units='xy', scale=1, color='tab:orange', alpha=0.8, 
    # label=rf"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=rf"$V_p^- (\mathbf{{x}},\tau={drift.temperatures[0]:.2f})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
)

plt.quiver(
    z[0,0], z[0,1],
    drift_neg[1,0,0], drift_neg[1,0,1],
    angles='xy', scale_units='xy', scale=1, color='tab:orange', alpha=0.5, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=rf"$V_p^- (\mathbf{{x}},\tau={drift.temperatures[1]:.2f})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
)

V_tau_015 = drift_pos[0] - drift_neg[0]
V_tau_025 = drift_pos[1] - drift_neg[1]

plt.quiver(
    z[0,0], z[0,1],
    V_tau_015[0,0], V_tau_015[0,1],
    angles='xy', scale_units='xy', scale=1, color='black', alpha=0.8, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=rf"$V_{{p,q}} (\mathbf{{x}},\tau={drift.temperatures[0]:.2f})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
)

plt.quiver(
    z[0,0], z[0,1],
    V_tau_025[0,0], V_tau_025[0,1],
    angles='xy', scale_units='xy', scale=1, color='black', alpha=0.5, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=rf"$V_{{p,q}} (\mathbf{{x}},\tau={drift.temperatures[1]:.2f})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
)

V_combined = V_tau_015 + V_tau_025

V_tau_015_norm = torch.sqrt(torch.mean(V_tau_015**2))
V_tau_025_norm = torch.sqrt(torch.mean(V_tau_025**2))
V_tau_015_ = V_tau_015 / (V_tau_015_norm + 1e-8)
V_tau_025_ = V_tau_025 / (V_tau_025_norm + 1e-8)
V_combined_normed = V_tau_015_ + V_tau_025_

plt.quiver(
    z[0,0], z[0,1],
    V_combined[0,0], V_combined[0,1],
    angles='xy', scale_units='xy', scale=1, color='green', alpha=0.8, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=r"$\sum_{\tau} V_{p,q} (\mathbf{x})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
)

plt.quiver(
    z[0,0], z[0,1],
    V_combined_normed[0,0], V_combined_normed[0,1],
    angles='xy', scale_units='xy', scale=1, color='green', alpha=0.5, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=r"$\sum_{\tau} V_{p,q}/||V_{p,q}|| (\mathbf{x})$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
)

plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 3.5)
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])
plt.legend(ncol=3, loc="upper center")
plt.tight_layout()
plt.savefig("drift_pos_example_two_temperatures.png", dpi=300)
# plt.gca().set_aspect("equal", adjustable="box")