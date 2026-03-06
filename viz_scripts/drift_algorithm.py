from tspath.model import MLP
from tspath.datasets import ToyDataset
from tspath.generative import DriftingField
import matplotlib.pyplot as plt
import torch

torch.manual_seed(42)

dataset = ToyDataset(n_samples=250, n_arms=2, noise=0.001, seed=42, )
z = torch.randn_like(torch.from_numpy(dataset.data))

model = MLP(input_dim=2, hidden_dim=64, output_dim=2)
model.load_state_dict(
    {k.replace("model.",""): v for k, v in torch.load("/home/samirdarouich/projects/TS_physics/tspath/runs/toy_spiral/normed_training/checkpoints/epoch_epoch=99-step_step=400.ckpt", weights_only=False)["state_dict"].items()}
    )
model.eval()

with torch.no_grad():
    x = model(z)
    y_neg = x.clone()
    y_pos = torch.from_numpy(dataset.data)

drift_field = DriftingField(0.15, normalize_drift=False)

V, *_ = drift_field(x, y_pos, y_neg)

plt.figure(figsize=(6,6))
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])
plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 3.5)

plt.scatter(z[:, 0], z[:, 1], s=5, label="$\mathbf{\epsilon}$", alpha=0.25, color="gray")
plt.legend(loc="upper center", fontsize=15, ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.15))
plt.savefig("algorithm_part_1.png",  dpi=300)

plt.scatter(y_pos[:, 0], y_pos[:, 1], s=10, label="$\mathbf{y}^+$", color="tab:blue")
plt.scatter(y_neg[:, 0], y_neg[:, 1], s=7, label="$\mathbf{y}^-=f(\mathbf{\epsilon})$", color="tab:orange", alpha=0.75)
plt.legend(loc="upper center", fontsize=15, ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.15))
plt.savefig("algorithm_part_2.png",  dpi=300)

plt.quiver(
    x[:,0], x[:,1],
    V[:,0], V[:,1],
    angles='xy', scale_units='xy', scale=1, color='black', alpha=0.5, 
    # label=r"$\frac{1}{Z_p} \mathbb{E}_{p} [ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) ]$",
    label=r"$V_{p,q} (f(\mathbf{\epsilon}))$",
    width=0.007,          # thinner shaft
    headwidth=3,          # smaller arrow head width
)
plt.legend(loc="upper center", fontsize=15, ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.15))
plt.savefig("algorithm_part_3.png",  dpi=300)