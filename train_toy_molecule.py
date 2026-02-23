import torch
import os
from ase.io import write
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import ToyMoleculeDataset
from tspath.generative import EquivariantDriftingField
from tspath.model import EquiformerV2
from tspath.utils import sample_noise_like, batch_inputs_to_atoms

@torch.no_grad()
def sample(model, batch):
    """Generate samples by integrating the learned flow field."""

    was_training = model.training
    model.eval()

    # Sample prior noise
    z = sample_noise_like(batch.pos, batch.batch)
    batch.pos = z

    # generate samples
    x = model(batch)

    # Convert to ASE Atoms
    batch.pos = x
    atoms_pred = batch_inputs_to_atoms(batch, pos_key="pos")

    if was_training:
        model.train()

    return atoms_pred

def visualize( model, batch, step, current_epoch, outdir=None):
    epoch = current_epoch
    atoms_pred = sample(model, batch)
    if outdir is not None:
        os.makedirs(f"{outdir}/{step}", exist_ok=True)
    write(f"{outdir}/{step}/sample_db.xyz", atoms_pred)
    
    # Write atoms as png image using ASE's built-in visualization
    for i, atoms in enumerate(atoms_pred):
        sample_path = f"{outdir}/{step}/epoch_{epoch:05d}_sample_{i}.png"
        write(sample_path, atoms, scale=100)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ToyMoleculeDataset(num_samples=1000, T=800, seed=42)
dataloader = GeometricDataLoader(dataset, batch_size=256, shuffle=True)
drifting_field = EquivariantDriftingField(temperature=0.15, aligned=True)
model = EquiformerV2(
    max_radius=11.0,
    num_distance_basis=64,
    num_layers=3,
    lmax_list=[2],
    use_noise_schedule_sigma_encoding=False,
)
model.to(device)

outdir = "drifting_samples"

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

losses = []
model.train()
n_epochs = 6000
pbar = tqdm(n_epochs, total=n_epochs, desc="Training")
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y = batch.pos.clone()

        z = sample_noise_like(batch.pos, batch.batch)
        batch.pos = z

        x = model(batch)

        # Get drifting field
        V = drifting_field(x, y, x, batch.num_atoms[0], temperature=None, aligned=None)

        x_drifted = (x + V).detach()

        loss = torch.nn.functional.mse_loss(x, x_drifted)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        
        if epoch % 500 == 0 and batch_idx == 0 and epoch > 0:
            visualize(model, batch, step=epoch, current_epoch=epoch, outdir=outdir)

