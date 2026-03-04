from tspath.utils import hungarian_and_kabch_batched,brute_force_and_kabch_batched
from tspath.datasets import ToyMoleculeDataset
import torch
import matplotlib.pyplot as plt
import numpy as np

def rotation_matrix_2d(angle_deg):
    theta = torch.deg2rad(torch.tensor(angle_deg, device=device))
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]
    )

device = "cpu"
dataset = ToyMoleculeDataset(
    name="cccccc",
    n_samples=1,
    r0=2.0,
    augment_with_rotations=False,
    augment_with_permutations=False,
)

# Only keep 2D
pos_dataset = dataset[0].pos[:,:2].to(device)

# Set the angle here
angle_deg = 90.0
rotation = rotation_matrix_2d(angle_deg)

# Rotate xy-coordinates of the molecule positions
pos_xy_rot = pos_dataset @ rotation.T

P = [3,2,5,0,4,1]  # Permutation to reorder the atoms 'CCCCCC'
# P = [0,2,1]  # Permutation to reorder the atoms 'CCC'
pos_xy_rot = pos_xy_rot[P, :]

def get_rmsd(x):
    return ((x[:,:2] - pos_xy_rot)**2).sum(-1).mean()**0.5

angles = list(range(0, 90))
rmsd_dict = []
rmsd_brute_force = []
for angle in angles:
    # randomly rotate input positions
    rotation = rotation_matrix_2d(angle)
    rotated_positions = pos_dataset @ rotation.T
    y_aligned, *_ = hungarian_and_kabch_batched(pos_xy_rot[None], rotated_positions[None])
    rmsd_dict.append(get_rmsd(y_aligned[0]))
    
    y_aligned_brute_force, *_ = brute_force_and_kabch_batched(pos_xy_rot[None], rotated_positions[None])
    rmsd_brute_force.append(get_rmsd(y_aligned_brute_force[0]))

golden_ratio = (1 + 5**0.5) / 2
width = 6
height = width / golden_ratio
angles = np.abs(np.array(angles) - 90) #(initial rotation was 90 degrees, so we shift the x-axis accordingly)
plt.figure(figsize=(width, height))
plt.scatter(angles, rmsd_dict, marker='o', label='Hungarian + Kabsch')
plt.scatter(angles, rmsd_brute_force, marker='x', label='Brute Force + Kabsch')
plt.xlabel(r"Rotation Angle $\theta(x,Py)$ / °", fontsize=15)
plt.ylabel(r"|$x-RPy$| / $\mathrm{\AA}$", fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig("convergence_iterative_hung_kabsch.png", dpi=300)