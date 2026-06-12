from collections import defaultdict
from typing import Callable, Tuple

import datamol as dm
import numpy as np
import torch
from datamol.types import Mol
from rdkit import Chem, RDLogger
from torch_geometric.transforms import BaseTransform

from symdrift.alignment import kabsch_batched_scatter
from symdrift.datasets.utils import (
    allowable_features,
    atom_to_feature_vector,
    build_conformer,
    compute_edge_index,
    get_automorphisms,
    compute_orbit_id_matrix,
    get_chiral_tensors,
)

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

__all__ = [
    "RemoveCOMReaction",
    "RemoveCOMConformer",
    "RemoveCOM",
    "RandomRotate",
    "RandomPermute",
    "ConformerAugment",
    "AlignReaction",
    "TargetReaction",
    "BoltzmannWeightingConformers",
    "GraphAutomorphism",
    "FeaturizeMolecule",
    "FeaturizeReaction",
]


def cache_decorator(func: Callable):
    """Decorator to handle caching logic."""

    def wrapper(self, smiles: str, *args, **kwargs):
        cache_key = func.__name__
        if smiles in self.cache and cache_key in self.cache[smiles]:
            return self.cache[smiles][cache_key]
        result = func(self, smiles, *args, **kwargs)
        self.cache[smiles][cache_key] = result
        return result

    return wrapper


class RemoveCOMReaction(BaseTransform):
    def forward(self, data):
        for pos_key in ["pos_ts", "pos_r", "pos_p"]:
            pos = data[pos_key]
            com = pos.mean(dim=0, keepdim=True)
            data[pos_key] = pos - com
        return data


class RemoveCOMConformer(BaseTransform):
    def forward(self, data):
        pos = data.pos
        num_atoms = data.x.shape[0]
        num_conformers = data.num_conformers
        pos = pos.view(num_conformers, num_atoms, 3)
        com = pos.mean(dim=1, keepdim=True)
        data.pos = (pos - com).view(-1, 3)
        return data


class RemoveCOM(BaseTransform):
    def forward(self, data):
        for pos_key in ["pos"]:
            pos = data[pos_key]
            com = pos.mean(dim=0, keepdim=True)
            data[pos_key] = pos - com
        return data


class RandomRotate(BaseTransform):
    def forward(self, data):

        # Sample random unit quaternion
        q = torch.randn(4)
        q = q / q.norm()

        w, x, y, z = q

        # Convert quaternion to rotation matrix
        R = torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

        data.pos = data.pos @ R.T
        return data


class RandomPermute(BaseTransform):
    def forward(self, data):
        types = data.x
        perm = torch.arange(len(types))

        # Permute within each atom type
        for t in types.unique():
            idx = (types == t).nonzero(as_tuple=True)[0]
            perm[idx] = idx[torch.randperm(len(idx))]

        # Apply permutation
        data.pos = data.pos[perm]
        data.x = data.x[perm]

        return data


class ConformerAugment(BaseTransform):
    def __init__(
        self,
        num_augs=1,
        rotate=True,
        permute=True,
        ignore_hs=False,
        use_atom_features=True,
    ):
        self.num_augs = num_augs
        self.rotate = rotate
        self.permute = permute
        self.ignore_hs = ignore_hs
        self.use_atom_features = use_atom_features
        self.cache = defaultdict(dict)

    def get_random_rotation(self):
        # Sample random unit quaternion
        q = torch.randn(4)
        q = q / q.norm()

        w, x, y, z = q

        # Convert quaternion to rotation matrix
        R = torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return R

    @cache_decorator
    def get_automorphisms(self, smiles):
        """Find random permutations of atoms that preserve atom types and bonding structure."""
        mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
        automorphisms = get_automorphisms(
            mol=mol, ignore_hs=self.ignore_hs, use_atom_features=self.use_atom_features
        )
        return automorphisms

    def get_random_permutation(self, smiles, x):
        all_perms = self.get_automorphisms(smiles)
        num_perms = all_perms.size(0)
        if num_perms < self.num_augs:
            # fill the rest by subsampling (with replacement) from found permutations
            needed = self.num_augs - num_perms
            idx = torch.randint(0, num_perms, (needed,), device=all_perms.device)
            sampled = all_perms[idx]
            all_perms = torch.cat([all_perms, sampled], dim=0)
        elif num_perms > self.num_augs:
            # subsample (without replacement) from found permutations
            idx = torch.randperm(num_perms, device=all_perms.device)[: self.num_augs]
            all_perms = all_perms[idx]

        # Verify that the permutations preserve atom types
        x_repeated = x.unsqueeze(0).expand(self.num_augs, -1)
        perm_indices = torch.arange(self.num_augs)[:, None]
        assert (x_repeated[perm_indices, all_perms] == x_repeated).all(), (
            "Permutation should preserve atom types"
        )
        return all_perms

    def forward(self, data):
        pos_list = []
        x_list = []
        conf_idx_list = []

        smiles = data.smiles
        pos = data.pos
        x = data.x_conf
        conf_idx = data.conformer_index

        # Get permutations that preserve the bonding structure and atom types, if needed
        if self.permute:
            permutations = self.get_random_permutation(smiles, data.x)

        unique_confs = conf_idx.unique()
        new_conf_counter = 0

        for conf in unique_confs:
            mask = conf_idx == conf
            pos_c = pos[mask]
            x_c = x[mask]
            for i in range(self.num_augs):
                pos_aug = pos_c.clone()
                x_aug = x_c.clone()

                # --- Rotation per conformer ---
                if self.rotate:
                    R = self.get_random_rotation()
                    pos_aug = pos_aug @ R.T

                # --- Permutation within conformer ---
                if self.permute:
                    perm = permutations[i]
                    pos_aug = pos_aug[perm]
                    x_aug = x_aug[perm]

                # --- Store ---
                pos_list.append(pos_aug)
                x_list.append(x_aug)
                conf_idx_list.append(
                    torch.full((len(pos_aug),), new_conf_counter, dtype=conf_idx.dtype)
                )

                new_conf_counter += 1

        # --- Concatenate everything ---
        data.pos = torch.cat(pos_list, dim=0)
        data.x_conf = torch.cat(x_list, dim=0)
        data.conformer_index = torch.cat(conf_idx_list, dim=0)
        data.num_conformers *= self.num_augs

        return data


class AlignReaction(BaseTransform):
    def forward(self, data):
        pos_r = data.pos_r
        pos_p = data.pos_p

        # align product to reactant
        pos_p_aligned = kabsch_batched_scatter(
            pos_r, pos_p, torch.zeros(pos_r.shape[0], dtype=torch.long)
        )

        data.pos_p = pos_p_aligned
        return data


class TargetReaction(BaseTransform):
    def __init__(self, target_str="pos_ts"):
        super().__init__()
        self.target_str = target_str

    def forward(self, data):
        data.pos = getattr(data, self.target_str).clone()
        return data


class BoltzmannWeightingConformers(BaseTransform):
    def __init__(self, keep_top_n=30):
        super().__init__()
        self.keep_top_n = keep_top_n

    def forward(self, data):
        boltzmann_weights = data.boltzmann_weights  # [num_conformers, 1]
        positions = data.pos  # [num_conformers*num_atoms, 3]
        num_atoms = data.x.shape[0]
        num_conformers = data.num_conformers
        positions = positions.view(
            num_conformers, num_atoms, 3
        )  # Reshape to [num_conformers, num_atoms, 3]
        # Sort after boltzmann weighting
        sorted_indices = torch.argsort(boltzmann_weights.view(-1), descending=True)[
            : self.keep_top_n
        ]
        data.x_conf = data.x_conf[: self.keep_top_n * num_atoms]
        data.pos = positions[sorted_indices].view(
            -1, 3
        )  # Reshape back to [num_conformers*num_atoms, 3]
        data.boltzmann_weights = boltzmann_weights[sorted_indices]
        data.energy = data.energy[sorted_indices]
        data.num_conformers = data.num_conformers.clamp(max=self.keep_top_n)
        data.conformer_index = data.conformer_index[: self.keep_top_n * num_atoms]
        return data


class GraphAutomorphism(BaseTransform):
    def __init__(
        self,
        smiles_key="smiles",
        use_atom_features=True,
        ignore_hs=False,
        perm_chunk_size: int = 512,
    ):
        # smiles based cache
        self.cache = defaultdict(dict)
        self.smiles_key = smiles_key
        self.use_atom_features = use_atom_features
        self.ignore_hs = ignore_hs
        self.perm_chunk_size = perm_chunk_size

    def forward(self, data):
        if hasattr(data, self.smiles_key):
            smiles = getattr(data, self.smiles_key)
            automorphisms = self.get_automorphisms(smiles)
            data.automorphisms = automorphisms.view(-1)
            data.num_automorphisms = torch.tensor(
                len(automorphisms), dtype=torch.long
            )
            data.orbit_ids = self.get_orbit_ids(smiles)  # [n_pairs]
            
        return data

    def get_mol(self, smiles: str) -> Mol:
        return dm.to_mol(smiles, remove_hs=False, ordered=True)

    @cache_decorator
    def get_automorphisms(self, smiles: str):
        mol = self.get_mol(smiles)
        automorphisms = get_automorphisms(
            mol, use_atom_features=self.use_atom_features, ignore_hs=self.ignore_hs
        )
        return automorphisms

    @cache_decorator
    def get_orbit_ids(self, smiles: str) -> torch.Tensor:
        """Computed the [n_atoms, n_atoms] symmetric orbit-ID matrix for a molecular graph
        using the automorphisms. The (i,j) entry of the matrix is the ID of the orbit that 
        the pair (i,j) belongs to. Only take upper triangular part of the matrix 
        (excluding diagonal) since it's symmetric and diagonal is trivial.
        
        Every interaction (i,j) belongs to an orbit defined by the set of automorphisms
        that map i to some k and j to some l. This means these interactions are 
        equivalent under graph symmetries and can be treated interchangably.
        """
        automorphisms = self.get_automorphisms(smiles)  # [n_perms, n_atoms]
        orbit_ids = compute_orbit_id_matrix(automorphisms)  # [n_atoms, n_atoms]
        n_atoms = orbit_ids.shape[0]
        triu_r, triu_c = torch.triu_indices(n_atoms, n_atoms, offset=1)
        return orbit_ids[triu_r, triu_c]


class FeaturizeMolecule(BaseTransform):
    def __init__(self, smiles_key="smiles"):
        # smiles based cache
        self.cache = defaultdict(dict)
        self.smiles_key = smiles_key

    def forward(self, data):
        if hasattr(data, self.smiles_key):
            smiles = getattr(data, self.smiles_key)
            node_attr = self.get_atom_features(smiles, use_ogb_feat=True)
            chiral_index, chiral_nbr_index, chiral_tag = self.get_chiral_centers(smiles)
            bonded_edge_index, edge_attr = self.get_edge_index(
                smiles, use_edge_feat=True
            )

            data.node_attr = node_attr
            data.chiral_index = chiral_index
            data.chiral_nbr_index = chiral_nbr_index
            data.chiral_tag = chiral_tag
            data.bonded_edge_index = bonded_edge_index
            data.edge_attr = edge_attr
        return data

    def get_mol(self, smiles: str) -> Mol:
        return dm.to_mol(smiles, remove_hs=False, ordered=True)

    def get_atomic_numbers_from_mol(self, mol: Mol) -> torch.Tensor:
        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=torch.int32,
        )
        return atomic_numbers

    def get_atom_features_from_mol(
        self, mol: Mol, use_ogb_feat: bool = True
    ) -> torch.Tensor:
        if use_ogb_feat:
            atom_features = torch.tensor(
                [atom_to_feature_vector(atom) for atom in mol.GetAtoms()],
                dtype=torch.float32,
            )
        else:
            atom_features = torch.tensor(
                [atom.GetFormalCharge() for atom in mol.GetAtoms()],
                dtype=torch.float32,
            ).view(-1, 1)
        return atom_features

    def get_edge_index_from_mol(
        self, mol: Mol, use_edge_feat: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes and shortest_hops for a given mol object."""
        edge_index, edge_attr = compute_edge_index(mol, with_edge_attr=use_edge_feat)
        return edge_index, edge_attr

    @cache_decorator
    def get_chiral_centers(self, smiles: str) -> torch.Tensor:
        # compute chiral centers
        mol = self.get_mol(smiles)
        chiral_index, chiral_nbr_index, chiral_tag = self.get_chiral_centers_from_mol(
            mol
        )

        self.cache[smiles]["chiral_centers"] = (
            chiral_index,
            chiral_nbr_index,
            chiral_tag,
        )
        return chiral_index, chiral_nbr_index, chiral_tag

    def get_chiral_centers_from_mol(self, mol: Mol) -> torch.Tensor:
        chiral_index, chiral_nbr_index, chiral_tag = get_chiral_tensors(mol)
        return chiral_index, chiral_nbr_index, chiral_tag

    @cache_decorator
    def get_mol_with_conformer(self, smiles: str, positions: torch.Tensor) -> Mol:
        mol = self.get_mol(smiles)
        mol.AddConformer(build_conformer(positions))
        return mol

    @cache_decorator
    def get_edge_index(
        self,
        smiles: str,
        use_edge_feat: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes for a given smiles."""
        # compute edge index
        mol = self.get_mol(smiles)
        edge_index, edge_attr = self.get_edge_index_from_mol(
            mol, use_edge_feat=use_edge_feat
        )

        self.cache[smiles]["edge_index"] = edge_index
        self.cache[smiles]["edge_attr"] = edge_attr
        return edge_index, edge_attr

    @cache_decorator
    def get_atom_features(self, smiles: str, use_ogb_feat: bool = True) -> torch.Tensor:
        # compute atom features
        mol = self.get_mol(smiles)
        atom_features = self.get_atom_features_from_mol(mol, use_ogb_feat=use_ogb_feat)
        return atom_features

    @cache_decorator
    def get_atomic_numbers(self, smiles: str) -> torch.Tensor:
        # compute atomic numbers
        mol = self.get_mol(smiles)
        atomic_numbers = self.get_atomic_numbers_from_mol(mol)
        return atomic_numbers


class FeaturizeReaction(BaseTransform):
    def __init__(self, r_smiles_key="r_smiles", p_smiles_key="p_smiles"):
        super().__init__()
        self.r_smiles_key = r_smiles_key
        self.p_smiles_key = p_smiles_key

    def get_mol(self, smiles: str) -> Chem.Mol:
        params = Chem.SmilesParserParams()
        params.removeHs = False
        return Chem.MolFromSmiles(smiles, params)

    def get_canonicalized_permutation(self, mol: Chem.Mol) -> np.ndarray:
        # This uses the reactant and product SMILES mapping.
        # 2. Calc Permutations (MapNum -> Index mappings)
        # perm: map_num for atom at index i
        perm = np.array([a.GetAtomMapNum() for a in mol.GetAtoms()]) - 1
        # perm_inv: index of atom with map_num i (Canonical ordering)
        perm_inv = np.argsort(perm)
        return perm, perm_inv

    def get_atom_features_from_mol(
        self, mol: Mol, use_ogb_feat: bool = True
    ) -> torch.Tensor:

        perm, perm_inv = self.get_canonicalized_permutation(mol)
        canonical_atoms = np.array(mol.GetAtoms())[perm_inv]
        z = torch.tensor(
            [atom.GetAtomicNum() for atom in canonical_atoms], dtype=torch.long
        )

        if use_ogb_feat:
            atom_features = torch.tensor(
                [atom_to_feature_vector(atom) for atom in canonical_atoms],
                dtype=torch.float32,
            )
        else:
            atom_features = torch.tensor(
                [atom.GetFormalCharge() for atom in canonical_atoms],
                dtype=torch.float32,
            ).view(-1, 1)
        return atom_features, z

    def get_edge_index(
        self,
        mol_r: Chem.Mol,
        mol_p: Chem.Mol,
        use_edge_feat: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes and shortest_hops for a given smiles."""
        # compute edge index
        N = mol_r.GetNumAtoms()

        _, perm_inv_r = self.get_canonicalized_permutation(mol_r)
        _, perm_inv_p = self.get_canonicalized_permutation(mol_p)

        adj_r = Chem.rdmolops.GetAdjacencyMatrix(mol_r)
        adj_p = Chem.rdmolops.GetAdjacencyMatrix(mol_p)
        adj_r_perm = adj_r[perm_inv_r, :][:, perm_inv_r]
        adj_p_perm = adj_p[perm_inv_p, :][:, perm_inv_p]

        # 2. Combine Graphs (Union of Edges)
        adj = adj_r_perm + adj_p_perm
        row, col = adj.nonzero()
        row = torch.from_numpy(row).long()
        col = torch.from_numpy(col).long()

        # get edge attributes using the original indices
        row_r, col_r = (
            torch.from_numpy(perm_inv_r[row]).long(),
            torch.from_numpy(perm_inv_r[col]).long(),
        )
        _, edge_attr_r, _ = compute_edge_index(
            mol_r,
            with_edge_attr=use_edge_feat,
            with_shortest_hops=False,
            edge_index=torch.stack([row_r, col_r]),
        )

        row_p, col_p = (
            torch.from_numpy(perm_inv_p[row]).long(),
            torch.from_numpy(perm_inv_p[col]).long(),
        )
        _, edge_attr_p, _ = compute_edge_index(
            mol_p,
            with_edge_attr=use_edge_feat,
            with_shortest_hops=False,
            edge_index=torch.stack([row_p, col_p]),
        )

        # Sort edges (PyG convention: row-major sort)
        edge_index = torch.stack([row, col])
        perm = (edge_index[0] * N + edge_index[1]).argsort()

        no_of_bonds = len(allowable_features["possible_bond_type_list"])

        # adapt edge attribute such that no bond equals 0
        edge_attr_r = edge_attr_r + 1
        edge_attr_p = edge_attr_p + 1
        edge_attr_r[edge_attr_r == no_of_bonds] = 0
        edge_attr_p[edge_attr_p == no_of_bonds] = 0

        edge_index = edge_index[:, perm]
        edge_attr = (edge_attr_r[perm] * no_of_bonds) + edge_attr_p[perm]

        return edge_index, edge_attr

    def forward(self, data):
        if hasattr(data, self.r_smiles_key) and hasattr(data, self.p_smiles_key):
            smiles_r = getattr(data, self.r_smiles_key)
            smiles_p = getattr(data, self.p_smiles_key)

            mol_r = self.get_mol(smiles_r)
            mol_p = self.get_mol(smiles_p)

            r_node_attr, r_canonical_atoms = self.get_atom_features_from_mol(
                mol_r, use_ogb_feat=True
            )
            p_node_attr, p_canonical_atoms = self.get_atom_features_from_mol(
                mol_p, use_ogb_feat=True
            )

            assert (r_canonical_atoms == p_canonical_atoms).all(), (
                "Canonical atom order should be the same for reactant and product"
            )
            assert (r_canonical_atoms == data.x).all(), (
                "Canonical atom order should be the same as in the xyz file"
            )

            bonded_edge_index, edge_attr = self.get_edge_index(
                mol_r, mol_p, use_edge_feat=True
            )

            data.r_node_attr = r_node_attr
            data.p_node_attr = p_node_attr
            data.bonded_edge_index = bonded_edge_index
            data.edge_attr = edge_attr

        return data
