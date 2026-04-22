
import torch
from tspath.model import EquiformerV2, MLP
from tspath.datasets import MoleculeDataset, CompositionBatchSampler, ToyDataset
from tspath.utils import (
    sample_noise_like,
    batch_inputs_to_atoms,
    get_shortest_path_fast_batched_x_1,
    get_rmsd_batch_aligned,
    get_rmsd_batch,
)
from tspath.analysis import check_validity, pymatgen_match, visualize_atoms_list
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import DataLoader
from ase.visualize import view
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(
    "/home/vinhtong/tspath/runs/tspath_run_2026_02_23_09_41_14/checkpoints/last.ckpt",  # scratch training with pipleline and no detach manuall optimizer
    weights_only=False,
)
torch.manual_seed(42)

model = MLP()
model.load_state_dict(
    {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    # ckpt_true
)
model.eval()
model.to(device)

dataset = ToyDataset(n_samples=1000, seed=42)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

ys = []
predicts = []
for y in dataloader:
    y = y.to(device)
    ys.append(y)

    z = torch.randn_like(y)
    with torch.no_grad():
        output = model(z)
    predicts.append(output)
ys = torch.cat(ys, dim=0)
predicts = torch.cat(predicts, dim=0)

plt.scatter(ys[:, 0].cpu(), ys[:, 1].cpu(), label="Ground Truth")
plt.scatter(predicts[:, 0].cpu(), predicts[:, 1].cpu(), label="Predicted")
plt.legend()
plt.savefig("just_python.png")