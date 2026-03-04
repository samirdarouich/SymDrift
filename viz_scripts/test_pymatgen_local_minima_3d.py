from pymatgen.analysis.molecule_matcher import HungarianOrderMatcher, KabschMatcher
from ase.io import read
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

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


class NaiveKabschMatcher(KabschMatcher):
    
    def __init__(self, target):
        target = get_pymatgen_molecule_from_ase_atoms(target)
        super().__init__(target)
    
    def fit(self, p):
        p_ = get_pymatgen_molecule_from_ase_atoms(p)
        p_prime, rmsd =  super().fit(p_)
        p_prime_ase = get_ase_atoms_from_pymatgen_molecule(p_prime)
        return p_prime_ase, rmsd

class NaiveHungarianOrderMatcher(HungarianOrderMatcher):
    
    def __init__(self, target):
        target = get_pymatgen_molecule_from_ase_atoms(target)
        super().__init__(target)
    
    def fit(self, p):
        p_ = get_pymatgen_molecule_from_ase_atoms(p)
        p_prime, rmsd =  super().fit(p_)
        p_prime_ase = get_ase_atoms_from_pymatgen_molecule(p_prime)
        return p_prime_ase, rmsd
    
    def permutations(self, p_atoms, p_centroid, p_weights, q_atoms, q_centroid, q_weights):
        """Generate two possible permutations of atom order. This method uses the principle component
        of the inertia tensor to pre-align the molecules and hungarian method to determine the order.
        There are always two possible permutation depending on the way to pre-aligning the molecules.

        Args:
            p_atoms: atom numbers
            p_centroid: array of atom positions
            p_weights: array of atom weights
            q_atoms: atom numbers
            q_centroid: array of atom positions
            q_weights: array of atom weights

        Yield:
            perm_inds: array of atoms' order
        """
        # generate full view from q shape to fill in atom view on the fly
        perm_inds = np.zeros(len(p_atoms), dtype=np.int64)

        # Find unique atoms
        species = np.unique(p_atoms)

        for specie in species:
            p_atom_inds = np.where(p_atoms == specie)[0]
            q_atom_inds = np.where(q_atoms == specie)[0]
            A = q_centroid[q_atom_inds]
            B = p_centroid[p_atom_inds]

            # Perform Hungarian analysis on distance matrix between atoms of 1st
            # structure and trial structure
            distances = cdist(A, B, "euclidean")
            _a_inds, b_inds = linear_sum_assignment(distances)

            perm_inds[q_atom_inds] = p_atom_inds[b_inds]

        yield perm_inds

def rotate_atom(atom, theta, phi, tau):
    rotated_atom = atom.copy()
    r = Rotation.from_euler('zyx', [theta, phi, tau], degrees=True).as_matrix()
    rotated_atom.set_positions(atom.get_positions() @ r.T)
    return rotated_atom

def permute_atom(atom, seed=None):
    perm = random_permute_within_atom_types(torch.tensor(atom.get_atomic_numbers()), seed=seed)
    permuted_atom = atom.copy()[perm]
    return permuted_atom

# get original atom
atom = read("/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq/raw/t1x_eq_C5H8O.xyz",0)

# ! As first test, just rotate the same atom and see if we can recover the original one, without applying any permutation.
matcher_kabsch = NaiveKabschMatcher(atom)
matcher_hungarian = NaiveHungarianOrderMatcher(atom)

# ! If we apply hungarian + kabsch, even though the atom indices are not permuted, we 
# ! most of the time stuck in some local minima

# test different intial rotations of each others
rmsds_kabsch = []
rmsds_hungarian_kabsch = []
angles = []
theta_values = [0, 45, 90, 135, 180]
phi_values = [0, 45, 90, 135, 180]
tau_values = [0, 45, 90, 135, 180]
for theta in theta_values:
    for phi in phi_values:
        for tau in tau_values:
            test_atom = rotate_atom(atom, theta=theta, phi=phi, tau=tau)
            rotated_aligned, rmsd = matcher_kabsch.fit(test_atom)
            rmsds_kabsch.append(rmsd)
            angles.append((theta, phi, tau))

            rotated_aligned_hungarian, rmsd_hungarian = matcher_hungarian.fit(test_atom)
            rmsds_hungarian_kabsch.append(rmsd_hungarian)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(rmsds_kabsch)), rmsds_kabsch, marker='o', label='Kabsch')
plt.xlabel('Angle Index')
plt.ylabel('RMSD')
plt.title('Kabsch Matcher RMSD')
plt.grid(True)
plt.ylim(0, 1.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(rmsds_hungarian_kabsch)), rmsds_hungarian_kabsch, marker='o', label='Hungarian', color='orange')
plt.xlabel('Angle Index')
plt.ylabel('RMSD')
plt.title('Hungarian Matcher RMSD')
plt.grid(True)
plt.ylim(0, 1.3)
plt.legend()

plt.tight_layout()
plt.savefig("matcher_comparison_rotations.png")
plt.close()
            

#! Now randomly rotate and permute the atom
rotated_atom = rotate_atom(atom, theta=30, phi=20, tau=80)
rotated_atom = permute_atom(rotated_atom, seed=42)

# ! As first test, just rotate the same atom and see if we can recover the original one, without applying any permutation.
matcher_kabsch = NaiveKabschMatcher(rotated_atom)
matcher_hungarian = NaiveHungarianOrderMatcher(rotated_atom)

# ! If we apply hungarian + kabsch, even though the atom indices are not permuted, we 
# ! most of the time stuck in some local minima

# test different intial rotations of each others
rmsds_kabsch = []
rmsds_hungarian_kabsch = []
angles = []
theta_values = [0, 45, 90, 135, 180]
phi_values = [0, 45, 90, 135, 180]
tau_values = [0, 45, 90, 135, 180]
for theta in theta_values:
    for phi in phi_values:
        for tau in tau_values:
            test_atom = rotate_atom(atom, theta=theta, phi=phi, tau=tau)
            rotated_aligned, rmsd = matcher_kabsch.fit(test_atom)
            rmsds_kabsch.append(rmsd)
            angles.append((theta, phi, tau))

            rotated_aligned_hungarian, rmsd_hungarian = matcher_hungarian.fit(test_atom)
            rmsds_hungarian_kabsch.append(rmsd_hungarian)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(rmsds_kabsch)), rmsds_kabsch, marker='o', label='Kabsch')
plt.xlabel('Angle Index')
plt.ylabel('RMSD')
plt.title('Kabsch Matcher RMSD')
plt.grid(True)
plt.ylim(0, 1.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(rmsds_hungarian_kabsch)), rmsds_hungarian_kabsch, marker='o', label='Hungarian', color='orange')
plt.xlabel('Angle Index')
plt.ylabel('RMSD')
plt.title('Hungarian Matcher RMSD')
plt.grid(True)
plt.ylim(0, 1.3)
plt.legend()

plt.tight_layout()
plt.savefig("matcher_comparison_rotations_and_perm.png")
plt.close()