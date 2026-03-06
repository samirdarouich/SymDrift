import numpy as np
import torch
from ase.io import read
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher
from scipy.spatial.transform import Rotation
from tspath.alignment import brute_force_and_kabch_batched


def random_permute_within_atom_types(atomic_numbers, seed=None):
    n_atoms = atomic_numbers.numel()

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    perm = list(range(n_atoms))
    for z in torch.unique(atomic_numbers):
        idx = (atomic_numbers == z).nonzero(as_tuple=True)[0]
        shuffled = idx[torch.randperm(len(idx), generator=g)]
        for i, j in zip(idx.tolist(), shuffled.tolist()):
            perm[i] = j

    return perm


def get_pymatgen_molecule_from_ase_atoms(atoms):
    from pymatgen.core import Molecule

    symbols = atoms.get_chemical_symbols()
    coords = atoms.get_positions()
    return Molecule(symbols, coords)


def get_ase_atoms_from_pymatgen_molecule(molecule):
    from ase import Atoms

    atomic_numbers = molecule.atomic_numbers
    coords = molecule.cart_coords
    return Atoms(atomic_numbers, positions=coords)


class NaiveBruteForceOrderMatcher(BruteForceOrderMatcher):
    def __init__(self, target):
        target = get_pymatgen_molecule_from_ase_atoms(target)
        super().__init__(target)

    def fit(self, p):
        p_ = get_pymatgen_molecule_from_ase_atoms(p)
        p_prime, rmsd = super().fit(p_)
        p_prime_ase = get_ase_atoms_from_pymatgen_molecule(p_prime)
        return p_prime_ase, rmsd


def rotate_atom(atom, theta, phi, tau):
    rotated_atom = atom.copy()
    r = Rotation.from_euler("zyx", [theta, phi, tau], degrees=True).as_matrix()
    rotated_atom.set_positions(atom.get_positions() @ r.T)
    return rotated_atom


def permute_atom(atom, seed=None):
    perm = random_permute_within_atom_types(
        torch.tensor(atom.get_atomic_numbers()), seed=seed
    )
    print(f"Permutation: {perm}")
    permuted_atom = atom.copy()[perm]
    return permuted_atom


def pytorch_rotate_align_parallel(target, source):
    x = torch.tensor(target.get_positions()).unsqueeze(0)
    y = torch.tensor(source.get_positions()).unsqueeze(0)
    atomic_numbers = torch.tensor(target.get_atomic_numbers()).unsqueeze(0)

    y_permuted_aligned, _ = brute_force_and_kabch_batched(x, y, atomic_numbers)
    rmse = torch.sqrt(torch.mean(torch.square(y_permuted_aligned - x), dim=[-1, -2]))[0]

    return y_permuted_aligned.squeeze(0).numpy(), rmse.item()


# get original atom
atom = read(
    "/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq/raw/t1x_eq_CHN3O.xyz",
    0,
)

# create a rotated and permuted version of the original atom
rotated_atom = rotate_atom(atom, theta=70, phi=30, tau=20)
rotated_atom = permute_atom(rotated_atom, seed=42)

matcher = NaiveBruteForceOrderMatcher(rotated_atom)


# Align and rotate original atom to the rotated and permuted version
theta_values = [0, 45, 90, 135, 180]
phi_values = [0, 45, 90, 135, 180]
tau_values = [0, 45, 90, 135, 180]

bz = len(theta_values) * len(phi_values) * len(tau_values)
atomic_numbers = (
    torch.tensor(rotated_atom.get_atomic_numbers()).unsqueeze(0).repeat(bz, 1)
)
x = torch.tensor(rotated_atom.get_positions()).unsqueeze(0).repeat(bz, 1, 1)
ys = []
aligned = []
rmsds = []
rmsds_torch = []
for theta in theta_values:
    for phi in phi_values:
        for tau in tau_values:
            test_atom = rotate_atom(atom, theta, phi, tau)
            ys.append(torch.tensor(test_atom.get_positions()).unsqueeze(0))
            rotated_aligned, rmsd = matcher.fit(test_atom)
            aligned.append(torch.tensor(rotated_aligned.positions).unsqueeze(0))
            rmsds.append(rmsd)
            rotated_aligend_torch, rmsd_torch = pytorch_rotate_align_parallel(
                rotated_atom, test_atom
            )
            rmsds_torch.append(rmsd_torch)
            assert (
                np.abs(rotated_aligned.positions - rotated_aligend_torch).max() < 1e-8
            ), "Something wrong with own implementation"

assert max(rmsds) < 1e-5, (
    "Brute force matcher from pymatgen failed to find correct alignment"
)
assert max(rmsds_torch) < 1e-5, (
    "Brute force matching from package failed to find correct alignment"
)

# Check batched versions as well
aligned = torch.cat(aligned, dim=0)
ys = torch.cat(ys, dim=0)
y_permuted_aligned, _ = brute_force_and_kabch_batched(x, ys, atomic_numbers)

assert torch.allclose(y_permuted_aligned, aligned), (
    "Aligned structures differ between implementations"
)

print("All tests passed successfully!")
