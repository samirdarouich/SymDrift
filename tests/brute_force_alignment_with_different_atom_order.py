import torch
from ase.io import read
from tspath.alignment import get_rmsd_batched
from tspath.analysis import pymatgen_match

dataset = read(
    "/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq/raw/t1x_eq_CHN3O.xyz",
    ":",
)
x = torch.stack([torch.tensor(atom.get_positions()) for atom in dataset]).float().cuda()
atomic_number = (
    torch.stack([torch.tensor(atom.get_atomic_numbers()) for atom in dataset])
    .float()
    .cuda()
)


x_test = x[0].clone().unsqueeze(0)
y_test = x[1].clone().unsqueeze(0)
atomic_numbers_x_test = atomic_number[0].clone().unsqueeze(0)
atomic_numbers_y_test = atomic_number[1].clone().unsqueeze(0)

print("Assuming x and y have the same atomic number ordering --> but they actually dont have, hence the algorithm will do incorrect permutations between different atom types")
rmsd = get_rmsd_batched(
    x_test,
    y_test,
    atomic_numbers=atomic_numbers_x_test,
    align=True,
    permute=True,
    brute_force_permutations=True,
)
print(rmsd.item())

sort_idx = torch.argsort(atomic_numbers_x_test, dim=-1)
x_test = torch.gather(x_test, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))
atomic_numbers_x_test = torch.gather(
    atomic_numbers_x_test, 1, sort_idx
)

sort_idx = torch.argsort(atomic_numbers_y_test, dim=-1)
y_test = torch.gather(y_test, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))
atomic_numbers_y_test = torch.gather(
    atomic_numbers_y_test, 1, sort_idx
)
print("Matching the atom order of x and y to be the same --> now it should be correct")
rmsd = get_rmsd_batched(
    x_test,
    y_test,
    atomic_numbers=atomic_numbers_x_test,
    align=True,
    permute=True,
    brute_force_permutations=True,
)
print(rmsd.item())

print("Now using pymatgen's implementation to see if we get the same result")
rmsd, _ = pymatgen_match(dataset[1], dataset[0])
print(rmsd)
