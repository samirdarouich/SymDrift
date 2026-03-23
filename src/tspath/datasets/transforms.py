import torch
from torch_geometric.transforms import BaseTransform
from rdkit import Chem
from rdkit import RDLogger
from tspath.alignment import kabsch_batched_scatter
from tspath.datasets.utils import atom_to_feature_vector, get_chiral_tensors, build_conformer, compute_edge_index
from collections import defaultdict
from datamol.types import Mol
from typing import Callable, Tuple
import datamol as dm

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

__all__ = [
    "RemoveCOMReaction",
    "RemoveCOM",
    "RandomRotate",
    "RandomPermute",
    "AlignReaction",
    "FeaturizeMolecule",
    "BoltzmannWeightingConformers",
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


class RemoveCOM(BaseTransform):
    def forward(self, data):
        for pos_key in ["pos"]:
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
        perm = torch.randperm(data.num_nodes)
        data.pos = data.pos[perm]
        data.x = data.x[perm]
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

class FeaturizeMolecule(BaseTransform):
    
    def __init__(self):
        # smiles based cache
        self.cache = defaultdict(dict)

    def forward(self, data):
        if hasattr(data, "smiles"):
            smiles = data.smiles
            node_attr = self.get_atom_features(smiles)
            chiral_index, chiral_nbr_index, chiral_tag = self.get_chiral_centers(
                smiles
            )
            bonded_edge_index, edge_features, shortest_hops = self.get_edge_index(smiles, True, True)

            data.node_attr = node_attr
            data.chiral_index = chiral_index
            data.chiral_nbr_index = chiral_nbr_index
            data.chiral_tag = chiral_tag
            data.shortest_hops = shortest_hops
            data.bonded_edge_index = bonded_edge_index
            data.edge_features = edge_features
        else:
            data.bonded_edge_index = torch.empty((2, 0)).long()
            data.edge_features = None
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
        self, mol: Mol, use_edge_feat: bool = False, use_shortest_hops: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes and shortest_hops for a given mol object."""
        edge_index, edge_attr, shortest_hops  = compute_edge_index(
            mol, with_edge_attr=use_edge_feat, with_shortest_hops=use_shortest_hops
        )
        return edge_index, edge_attr, shortest_hops

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
        self, smiles: str, use_edge_feat: bool, use_shortest_hops: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes and shortest_hops for a given smiles."""
        # compute edge index
        mol = self.get_mol(smiles)
        edge_index, edge_attr, shortest_hops = self.get_edge_index_from_mol(
            mol, use_edge_feat=use_edge_feat, use_shortest_hops=use_shortest_hops
        )

        self.cache[smiles]["edge_index"] = edge_index
        self.cache[smiles]["edge_attr"] = edge_attr
        self.cache[smiles]["shortest_hops"] = shortest_hops
        return edge_index, edge_attr, shortest_hops
    
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
    
class BoltzmannWeightingConformers(BaseTransform):
    def __init__(self, keep_top_n=30):
        super().__init__()
        self.keep_top_n = keep_top_n

    def forward(self, data):
        boltzmann_weights = data.boltzmann_weights  # [num_conformers, 1]
        positions = data.pos # [num_conformers*num_atoms, 3]
        num_atoms = data.x.shape[0]
        num_conformers = data.num_conformers
        positions = positions.view(num_conformers, num_atoms, 3)  # Reshape to [num_conformers, num_atoms, 3]
        # Sort after boltzmann weighting
        sorted_indices = torch.argsort(boltzmann_weights.view(-1), descending=True )[: self.keep_top_n]
        data.pos = positions[sorted_indices].view(-1, 3)  # Reshape back to [num_conformers*num_atoms, 3]
        data.boltzmann_weights = boltzmann_weights[sorted_indices]
        data.energy = data.energy[sorted_indices]
        data.num_conformers = data.num_conformers.clamp(max=self.keep_top_n)
        return data