import os

import matplotlib.pyplot as plt
from ase.io import write
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.datasets import MoleculeDataset, CompositionBatchSampler
from tspath.generative import CondOTScheduler
from tspath.model import EGNN, EquiformerV2, PaiNN
from tspath.utils import sample_noise_like, batch_inputs_to_atoms
from torch_geometric.data import Data
from tspath.alignment import hungarian_and_kabch_batched, brute_force_and_kabch_batched, get_rmsd_batched

def create_batch_object(batch, n_samples):
    
    batch_sampling = Data()
    # Sample prior noise
    n_atoms = batch.num_atoms[0].item()
    d = batch.pos.shape[1]

    batch_ = torch.arange(n_samples, device=device).repeat_interleave(n_atoms)
    dummy = torch.zeros((n_samples * n_atoms, d), dtype=torch.float, device=device)
    x = batch.x[:n_atoms].repeat(n_samples)
    num_atoms = batch.num_atoms[0].repeat(n_samples)

    z = sample_noise_like(dummy, batch_)
    
    batch_sampling.pos = z
    batch_sampling.batch = batch_
    batch_sampling.x = x
    batch_sampling.num_atoms = num_atoms
    return batch_sampling

@torch.no_grad()
def sample(model, scheduler, num_steps, batch, n_samples):
    """Generate samples by integrating the learned flow field."""
    torch.manual_seed(42)
    was_training = model.training
    model.eval()

    # Sample prior noise
    batch_sampling = create_batch_object(batch, n_samples)

    # generate samples
    x, _ = scheduler.sample(
        batch_sampling.pos, num_steps=num_steps, model=model, batch=batch_sampling
    )
    
    if was_training:
        model.train()

    batch_sampling.pos_generated = x
    atoms_samples = batch_inputs_to_atoms(batch_sampling, "pos_generated")
    return atoms_samples

        
def visualize(model, scheduler, num_steps, batch, current_step, n_samples=None, outdir=None):
    step = current_step
    atoms_samples = sample(model, scheduler, num_steps, batch, n_samples=n_samples)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        if isinstance(step, int):
            step_str = f"{step:04d}"
        else:
            step_str = str(step)
        write(f"{plot_dir}/step_{step_str}.png", atoms_samples[0])
        write(f"{plot_dir}/step_{step_str}.xyz", atoms_samples)
    plt.close()


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
dataset_name = "t1x_eq_CHN3O" #! just 6 ATOMS
augment_with_rotations = False
augment_with_permutations = False

dataset = MoleculeDataset(
    source=dataset_name,
    root="/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq",
    split="train",
    split_identifier="debug",
    augment_with_rotations=augment_with_rotations,
    augment_with_permutations=augment_with_permutations,
)

model_type = "egnn"
aligned = True
permuted = True
brute_force_permutations = False
model_dict = {
    "painn": PaiNN(use_noise_schedule_sigma_encoding=True),
    "egnn": EGNN(use_noise_schedule_sigma_encoding=True),
    "equiformerv2": EquiformerV2(
        max_radius=11.0,
        num_distance_basis=64,
        num_layers=3,
        lmax_list=[2],
        use_noise_schedule_sigma_encoding=True,
    ),
}
model = model_dict[model_type]
model.to(device)


outdir = f"runs/molecule/dataset_{dataset_name}/{model_type}/fm_baseline/aligned_{aligned}_permuted_{permuted}_brute_force_{brute_force_permutations}"
ckpt_dir = f"{outdir}/checkpoints"
plot_dir = f"{outdir}/plots"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

flow_scheduler = CondOTScheduler(sigma=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)


batch_size = 64
batch_sampler = CompositionBatchSampler(
    dataset,
    k=1,  # number of chunks per batch
    n=batch_size,  # chunk size (number of samples per chunk)
    shuffle=True,
    drop_last=False,
    resample=True,
    seed=42,
)
dataloader = GeometricDataLoader(
    dataset, batch_sampler=batch_sampler, shuffle=False
)

losses = []
model.train()

n_steps = 9000
n_epochs = n_steps // len(dataloader)

n_samples = 1000
pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
step_count = 0
for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        
        # Target is the clean data
        x1 = batch.pos.clone()

        # Sample noise for each sample
        batch_neg = batch.clone()
        x0 = sample_noise_like(batch.pos, batch.batch)
        
        # Align and permute if specified
        if aligned and permuted:
            x1_ = x1.view(batch.num_graphs, -1, x1.shape[-1])
            x0_ = x0.view(batch.num_graphs, -1, x0.shape[-1])
            rmsd_orig = get_rmsd_batched(x1_, x0_)
            atomic_numbers = batch.x.view(batch.num_graphs, -1).long()

            if brute_force_permutations:
                x0_aligned, _ = brute_force_and_kabch_batched(x1_, x0_, atomic_numbers)
            else:
                x0_aligned = hungarian_and_kabch_batched(x1_, x0_, atomic_numbers)
            rmsd_aligned = get_rmsd_batched(x1_, x0_aligned)
            x0_aligned = x0_aligned.view(-1, x0.shape[-1])
        
        xt, t, v_target = flow_scheduler.sample_time_and_conditional_path(
            x0_aligned, x1, batch=batch.batch
        )
        rmsd_xt = get_rmsd_batched(
            x1.view(batch.num_graphs, -1, x1.shape[-1]), 
            xt.view(batch.num_graphs, -1, xt.shape[-1])
        )
        
        batch.pos = xt
        batch.t = t

        # Call the model
        v_pred = model(batch)
        
        x1_pred = xt + v_pred * (1-t)
        rmsd_x1_pred = get_rmsd_batched(
            x1.view(batch.num_graphs, -1, x1.shape[-1]), 
            x1_pred.view(batch.num_graphs, -1, x1_pred.shape[-1])
        )
    
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)

        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(
            {
                "step": step_count, 
                "loss": loss.item(),
                "rmsd_orig": rmsd_orig.mean().item(),
                "rmsd_aligned": rmsd_aligned.mean().item(),
                "rmsd_xt": rmsd_xt.mean().item(),
                "rmsd_x1_pred": rmsd_x1_pred.mean().item(),
            }
        )

        if step_count % (n_steps // 10) == 0 and step_count > 0:
            visualize(
                model, flow_scheduler, num_steps=1, batch=batch, current_step=step_count, n_samples=n_samples, outdir=outdir
            )
        step_count += 1
            
    # if (epoch < 101 and epoch % 5 == 0) or (epoch>100 and epoch % 20 == 0):
    #     torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/epoch_{epoch}.pt")

visualize(model, flow_scheduler, num_steps=1, batch=batch, current_step="final", n_samples=n_samples, outdir=outdir)
torch.save({"state_dict": model.state_dict()}, f"{ckpt_dir}/final_model.pt")
