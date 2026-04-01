import os
from functools import partial

import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.analysis import evaluate_toy
from tspath.datasets import ToyMoleculeDataset
from tspath.generative import EquivariantDriftingField
from tspath.model import EGNN, MLP, DiT, PaiNN, TorchMDDynamics
from tspath.utils import sample_noise_like_2d


def create_batch_object(batch, n_samples):

    # Repeat each graph in the batch n_samples times to create a new batch for sampling
    data_list = batch.to_data_list()
    repeated_list = [data for data in data_list for _ in range(n_samples)]
    batch_negative = Batch.from_data_list(repeated_list)

    z = sample_noise_like_2d(batch_negative.pos, batch_negative.batch)
    batch_negative.pos = z

    return batch_negative


@torch.no_grad()
def sample(model, batch, n_samples):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()

    # sample prior noise
    batch_sampling = create_batch_object(batch, n_samples)

    # generate samples
    x = model(batch_sampling)

    if was_training:
        model.train()

    return x.cpu()


def visualize(model, batch, current_step, n_samples=None, outdir=None):
    step = current_step
    x_samples = sample(model, batch=batch, n_samples=n_samples)
    mse = evaluation_function(x_samples)
    plt.scatter(
        x_samples[:, 0], x_samples[:, 1], alpha=0.5, color="red", label="Samples"
    )
    plt.scatter(
        pos_dataset[:, :, 0],
        pos_dataset[:, :, 1],
        alpha=0.75,
        color="gray",
        label="Dataset",
    )
    plt.legend()
    plt.xlim(
        pos_dataset[:, :, 0].min().item() - 1.0, pos_dataset[:, :, 0].max().item() + 1.0
    )
    plt.ylim(
        pos_dataset[:, :, 1].min().item() - 1.0, pos_dataset[:, :, 1].max().item() + 1.0
    )
    plt.title(f"Step {step}, MSE: {mse:.4f}")
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        if isinstance(step, int):
            step_str = f"{step:04d}"
        else:
            step_str = str(step)
        plt.savefig(f"{plot_dir}/step_{step_str}.png")
    plt.close()


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "carbon_chain"
augment_with_rotations = False
augment_with_permutations = False

n_atoms = 3
r0 = 2.0
theta0 = 120.0
factor = 1.25

# If no augmentation than its just one molecule
if augment_with_rotations and augment_with_permutations:
    n_samples = 1000
else:
    n_samples = 1

dataset = ToyMoleculeDataset(
    name=dataset_name,
    n_samples=n_samples,
    T=0,
    seed=42,
    r0=r0,
    theta0=theta0,
    factor=factor,
    n_atoms=n_atoms,
    augment_with_rotations=augment_with_rotations,
    augment_with_permutations=augment_with_rotations,
)
evaluation_function = partial(
    evaluate_toy,
    dataset_name=dataset_name,
    n_atoms=n_atoms,
    r0=r0,
    theta0=theta0,
    factor=factor,
)

if dataset_name == "carbon_chain":
    dataset_name += f"_n_atoms_{n_atoms}"

pos_dataset = torch.stack([data.pos for data in dataset])
n_atoms = pos_dataset.shape[1]

model_type = "dit_naive"
aligned = True
permuted = True
brute_force_permutations = True
model_dict = {
    "painn": PaiNN(
        sphere_channels=256,
        num_layers=9,
    ),
    "egnn": EGNN(num_distance_basis=0),
    "mlp": MLP(input_dim=n_atoms * 3 , hidden_dim=256, num_layers=9, output_dim=n_atoms * 2), #2d and atomic number as input
    "dit_perm_eq": DiT(mgn_num_node_features=1, positional_encoding_bool=False, relative_positional_embedding_bool=False, max_radius=0), #positional_encoding_bool=True breaks permutation equivariance, 
    "dit_naive": DiT(
        mgn_num_node_features=1, 
        positional_encoding_bool=True, # break permutation equivariance
        relative_positional_embedding_bool=True, # break rotation equivariance
        absolute_positional_embedding_bool=True, # break translation and rotation equivariance
    ),
    "torchmd": TorchMDDynamics(sphere_channels=160, num_layers=9, max_radius=11.0, node_attr_dim=1),
}
model = model_dict[model_type]
model.to(device)

only_pos_drift = True
drift_str = "all_drift"
if only_pos_drift:
    drift_str = "pos_drift"

normalize_drift = True
temperatures = [0.15]
temp_str = "_".join([f"{t:.2f}" for t in temperatures])
outdir = f"runs/toy_molecule/dataset_{dataset_name}/{model_type}/temp_{temp_str}/norm_{normalize_drift}/augment_rot_{augment_with_rotations}_augment_perm_{augment_with_permutations}/aligned_{aligned}_permuted_{permuted}_brute_force_{brute_force_permutations}/{drift_str}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

drifting_field = EquivariantDriftingField(
    temperatures=temperatures,
    normalize_drift=normalize_drift,
    aligned=aligned,
    permuted=permuted,
    brute_force_permutations=brute_force_permutations,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

losses = []
model.train()

n_steps = 3000
batch_size_pos = min(16, len(dataset))
dataloader = GeometricDataLoader(
    dataset,
    batch_size=batch_size_pos,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
)
n_neg_per_pos = 64

# if just one y is used, then we overall train less steps, as steps = n_epochs * batch size
# in the augmented case we have more samples and thus more steps, use less epochs
n_epochs = n_steps // len(dataloader)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
n_samples = 1000
pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
step_count = 0
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y = batch.pos.clone()

        # sample negative samples for each positive sample in the batch
        batch_neg = create_batch_object(batch=batch, n_samples=n_neg_per_pos)

        x = model(batch_neg)
        
        if epoch == 20 and batch_idx == 0:
            # test permutation equivariance
            with torch.no_grad():
                N = batch_neg.x.size(0)
                noise = torch.rand(N).to(device)
                sort_key = batch_neg.batch.float() + noise
                idx = torch.argsort(sort_key)
                inv_idx = torch.empty_like(idx)
                inv_idx[idx] = torch.arange(N).to(device)

                batch_neg_perm = batch_neg.clone()
                batch_neg_perm.pos = batch_neg.pos[idx]
                batch_neg_perm.bonded_edge_index = inv_idx[batch_neg.bonded_edge_index]
                
                x_perm = model(batch_neg_perm)
                diff = x[idx] - x_perm
                mse_perm = torch.mean(diff**2)
                if mse_perm > 1e-6:
                    print(f"Warning: Permutation equivariance test failed with MSE {mse_perm.item():.2e}")
                
                # test rotation equivariance
                rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device=device) # 90 degree rotation
                batch_neg_rot = batch_neg.clone()
                batch_neg_rot.pos = torch.cat([batch_neg.pos[:,:2] @ rot.T, torch.zeros_like(batch_neg.pos[:,2:])], dim=-1)
                x_rot = model(batch_neg_rot)
                x_rot_expected = x[:, :2] @ rot.T
                diff = x_rot[:, :2] - x_rot_expected
                mse_rot = torch.mean(diff**2)
                if mse_rot > 1e-6:
                    print(f"Warning: Rotation equivariance test failed with MSE {mse_rot.item():.2e}")
        
        # if x[..., 2].abs().max() > 1e-4:
        #     print(
        #         "Warning: Non-zero z-component in model prediction, which should be zero for 2D data."
        #     )
        
        # Get drifting field in 2D as 3D rotations could include reflections in 2D which are not valid
        V, V_pos, V_neg, *_ = drifting_field(
            x[..., :2],
            y[..., :2],
            x[..., :2],
            batch.num_atoms[0],
            # atomic_numbers=batch.x, # dont use as we assume always all molecules are the same in this toy example
            temperatures=None,
            aligned=None,
            permuted=None,
            brute_force_permutations=None,
        )

        if only_pos_drift:
            V_pos_ = torch.cat(
                [V_pos, torch.zeros_like(V_pos[..., :1])], dim=-1
            )
            x_drifted = (x + V_pos_).detach()
        else:
            V_ = torch.cat([V, torch.zeros_like(V[..., :1])], dim=-1)
            x_drifted = (x + V_).detach()

        loss = torch.nn.functional.mse_loss(x, x_drifted)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(
            {
                "step": step_count,
                "mse(V)": torch.sqrt(torch.mean(V**2)).item(),
                "mse(pos_drift)": torch.sqrt(torch.mean(V_pos**2)).item(),
                "mse(neg_drift)": torch.sqrt(torch.mean(V_neg**2)).item(),
            }
        )

        if step_count % (n_steps // 10) == 0 and step_count > 0:
            n_samples = 1000 // batch.num_graphs
            visualize(
                model,
                batch=batch,
                current_step=step_count,
                n_samples=n_samples,
                outdir=outdir,
            )

        step_count += 1

    scheduler.step()

n_samples = 1000 // batch.num_graphs
visualize(model, current_step="final", batch=batch, n_samples=n_samples, outdir=outdir)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")
