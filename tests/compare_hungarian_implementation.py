import numpy as np
import torch
from ase.io import read
from pymatgen.analysis.molecule_matcher import HungarianOrderMatcher
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from torch_linear_assignment import batch_linear_assignment
from tspath.utils import kabsch_batched, hungarian_and_kabch_batched


def permute_by_atom_type(x, y, atom_types):
    """
    x, y: (B, N, d)
    atom_types: (B, N)  integer labels
    """
    B, N, d = x.shape
    device = x.device

    y_permuted = torch.zeros_like(y)

    for b in range(B):
        types = atom_types[b]
        unique_types = torch.unique(types)

        perm_indices = torch.empty(N, dtype=torch.long, device=device)

        for t in unique_types:
            mask = types == t

            idx = mask.nonzero(as_tuple=False).squeeze(-1)

            x_t = x[b, idx]  # (n_t, d)
            y_t = y[b, idx]  # (n_t, d)

            cost = torch.cdist(x_t, y_t)
            assignment = batch_linear_assignment(cost.unsqueeze(0)).squeeze(
                0
            )  # Hungarian (n_t,)

            # _a_inds, b_inds = linear_sum_assignment(cost.numpy())

            # assert all(b_inds==assignment.numpy()), "difference"

            perm_indices[idx] = idx[assignment]

        y_permuted[b] = y[b, perm_indices]

    y_permuted_aligned, R = kabsch_batched(x, y_permuted)
    return y_permuted_aligned, R


def permute_by_atom_type_parallel(x, y, atom_types):
    B, N, d = x.shape
    device = x.device

    y_permuted = torch.zeros_like(y)

    unique_types = torch.unique(atom_types)

    for t in unique_types:
        # mask: (B, N)
        mask = atom_types == t

        # number of atoms of this type (assume constant per batch)
        n_t = mask.sum(dim=1)

        # if variable per batch → more complex handling required
        assert torch.all(n_t == n_t[0]), (
            "Different counts per batch not supported in this simple version"
        )
        n_t = n_t[0].item()

        # Gather atoms of type t
        idx = mask.nonzero(as_tuple=False)
        # idx: (B*n_t, 2) → (batch_idx, atom_idx)

        x_t = torch.zeros(B, n_t, d, device=device, dtype=x.dtype)
        y_t = torch.zeros(B, n_t, d, device=device, dtype=y.dtype)

        for b in range(B):
            atom_idx = idx[idx[:, 0] == b][:, 1]
            x_t[b] = x[b, atom_idx]
            y_t[b] = y[b, atom_idx]

        # Compute batched cost
        cost = torch.cdist(x_t, y_t)  # (B, n_t, n_t)

        # Solve Hungarian in batch
        assignment = batch_linear_assignment(cost)  # (B, n_t)

        # Scatter back
        for b in range(B):
            atom_idx = idx[idx[:, 0] == b][:, 1]
            y_permuted[b, atom_idx] = y_t[b, assignment[b]]

    y_permuted_aligned, R = kabsch_batched(x, y_permuted)
    return y_permuted_aligned, R


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


class NaiveHungarianOrderMatcher(HungarianOrderMatcher):
    def __init__(self, target):
        target = get_pymatgen_molecule_from_ase_atoms(target)
        super().__init__(target)

    def fit(self, p):
        p_ = get_pymatgen_molecule_from_ase_atoms(p)
        p_prime, rmsd = super().fit(p_)
        p_prime_ase = get_ase_atoms_from_pymatgen_molecule(p_prime)
        return p_prime_ase, rmsd

    def match(self, p):
        """Similar as `KabschMatcher.match` but this method also finds the order of
        atoms which belongs to the best match.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            inds: The indices of atoms
            U: 3x3 rotation matrix
            V: Translation vector
            rmsd: Root mean squared deviation between P and Q
        """
        if sorted(p.atomic_numbers) != sorted(self.target.atomic_numbers):
            raise ValueError("The number of the same species aren't matching!")

        p_coord, q_coord = p.cart_coords, self.target.cart_coords
        p_atoms, q_atoms = (
            np.array(p.atomic_numbers),
            np.array(self.target.atomic_numbers),
        )

        p_weights = np.array([site.species.weight for site in p])
        q_weights = np.array([site.species.weight for site in self.target])

        # Both sets of coordinates must be translated first, so that
        # their center of mass with the origin of the coordinate system.
        p_trans, q_trans = p_coord.mean(axis=0), q_coord.mean(axis=0)
        p_centroid, q_centroid = p_coord - p_trans, q_coord - q_trans

        # Initializing return values
        rmsd = np.inf

        # Generate all permutation grouped/sorted by the elements
        inds = []
        U = np.empty(0)
        for p_inds_test in self.permutations(
            p_atoms, p_centroid, p_weights, q_atoms, q_centroid, q_weights
        ):
            p_centroid_test = p_centroid[p_inds_test]
            U_test = self.kabsch(p_centroid_test, q_centroid)

            p_centroid_prime_test = np.dot(p_centroid_test, U_test)
            rmsd_test = np.sqrt(np.mean(np.square(p_centroid_prime_test - q_centroid)))

            if rmsd_test < rmsd:
                inds, U, rmsd = p_inds_test, U_test, rmsd_test

        # Rotate and translate matrix P unto matrix Q using Kabsch algorithm.
        # P' = P * U + V
        V = q_trans - np.dot(p_trans, U)

        return inds, U, V, rmsd

    @staticmethod
    def permutations(p_atoms, p_centroid, p_weights, q_atoms, q_centroid, q_weights):
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
    r = Rotation.from_euler("zyx", [theta, phi, tau], degrees=True).as_matrix()
    rotated_atom.set_positions(atom.get_positions() @ r.T)
    return rotated_atom


def permute_atom(atom, seed=None):
    perm = random_permute_within_atom_types(
        torch.tensor(atom.get_atomic_numbers()), seed=seed
    )
    permuted_atom = atom.copy()[perm]
    return permuted_atom


def pytorch_rotate_align(target, source):
    x = torch.tensor(target.get_positions()).unsqueeze(0)
    y = torch.tensor(source.get_positions()).unsqueeze(0)
    atomic_numbers = torch.tensor(target.get_atomic_numbers()).unsqueeze(0)

    # pymatgen does this
    y_permuted_aligned, R = permute_by_atom_type(x, y, atomic_numbers)
    rmse = torch.sqrt(torch.mean(torch.square(y_permuted_aligned - x), dim=[-1, -2]))[0]

    return y_permuted_aligned.squeeze(0).numpy(), rmse.item()


def pytorch_rotate_align_parallel(target, source):
    x = torch.tensor(target.get_positions()).unsqueeze(0)
    y = torch.tensor(source.get_positions()).unsqueeze(0)
    atomic_numbers = torch.tensor(target.get_atomic_numbers()).unsqueeze(0)

    y_permuted_aligned, R = permute_by_atom_type_parallel(x, y, atomic_numbers)
    rmse = torch.sqrt(torch.mean(torch.square(y_permuted_aligned - x), dim=[-1, -2]))[0]

    return y_permuted_aligned.squeeze(0).numpy(), rmse.item()


# get original atom
atom = read(
    "/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq/raw/t1x_eq_C5H8O.xyz",
    0,
)

# create a rotated and permuted version of the original atom
rotated_atom = rotate_atom(atom, theta=45, phi=70, tau=0)
rotated_atom = permute_atom(rotated_atom, seed=42)

matcher = NaiveHungarianOrderMatcher(rotated_atom)

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
for theta in theta_values:
    for phi in phi_values:
        for tau in tau_values:
            test_atom = rotate_atom(atom, theta, phi, tau)
            ys.append(torch.tensor(test_atom.get_positions()).unsqueeze(0))
            rotated_aligned, rmsd = matcher.fit(test_atom)
            aligned.append(torch.tensor(rotated_aligned.positions).unsqueeze(0))
            rotated_aligend_torch, rmsd_torch = pytorch_rotate_align(
                rotated_atom, test_atom
            )
            rotated_aligend_torch_parallel, rmsd_torch_parallel = (
                pytorch_rotate_align_parallel(rotated_atom, test_atom)
            )

            assert (
                np.abs(rotated_aligned.positions - rotated_aligend_torch).max() < 1e-8
            ), "Something wrong with own implementation"
            assert (
                np.abs(rotated_aligned.positions - rotated_aligend_torch_parallel).max()
                < 1e-8
            ), "Something wrong with own parallel implementation"

# Check batched versions as well
aligned = torch.cat(aligned, dim=0)
ys = torch.cat(ys, dim=0)
y_permuted_aligned, R = permute_by_atom_type_parallel(x, ys, atomic_numbers)
y_permuted_aligned_parallel, R_parallel = permute_by_atom_type_parallel(
    x, ys, atomic_numbers
)
y_permuted_aligend_codebase, *_ = hungarian_and_kabch_batched(
    x, ys, atomic_numbers, max_iter=1
)

assert torch.allclose(y_permuted_aligned, y_permuted_aligned_parallel), (
    "Parallel and non-parallel implementations differ"
)
assert torch.allclose(y_permuted_aligned, aligned), (
    "Aligned structures differ between implementations"
)
assert torch.allclose(y_permuted_aligned_parallel, aligned), (
    "Aligned structures differ between implementations"
)
assert torch.allclose(y_permuted_aligend_codebase, aligned), (
    "Aligned structures differ between implementations"
)
