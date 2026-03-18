from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
from ase.io import read
from torch_geometric.transforms import Compose
import logging
from collections import defaultdict
from typing import Optional
from tspath.datasets.transforms import RandomPermute, RandomRotate, RemoveCOM, RemoveCOMConformer, FeaturizeMolecule, BoltzmannWeightingConformers
from tspath.utils import inputs_to_atoms
import datamol as dm
import os
import pickle
import glob
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)

def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def check_disconnected_components(mol):
    """Check for disconnected components using Union-Find algorithm."""
    # Initialize parent array for union-find
    n_nodes = mol.GetNumAtoms()
    parent = list(range(n_nodes))

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Add reverse edge for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Process all edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        union(src, dst)

    # Count unique components
    components = {}
    for node in range(n_nodes):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)

    return list(components.values())

class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        source,
        root,
        identifier: str = "rxn",
        split=None,              # 'train' | 'val' | 'test'
        split_identifier: Optional[str] = None,
        transform=None,
        pre_transform=Compose([RemoveCOM(), FeaturizeMolecule()]),
        pre_filter=None,
        augment_with_rotations=False,
        augment_with_permutations=False,
    ):
        self.identifier = identifier
        self.source = source
        
        # Add transforms for data augmentation
        transforms = Compose([])
        if transform is not None:
            transforms.transforms.extend(transform.transforms)
        if augment_with_rotations:
            transforms.transforms.append(RandomRotate())
        if augment_with_permutations:
            transforms.transforms.append(RandomPermute())
        
        super().__init__(root, transforms, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        logger.info(f"Loaded dataset from {self.processed_paths[0]} with {self.len()} samples.")
        
        # load split if specified
        if split is not None:
            self._apply_split(split, split_identifier)
            
        self.comp_to_indices = defaultdict(list)
        for idx in range(len(self)):
            self.comp_to_indices[self.get(idx).formula.item()].append(idx)
        self.compositions = sorted(list(self.comp_to_indices.keys()))

    def _apply_split(self, split, split_identifier=None):
        if split_identifier is not None:
            filename = f"split_{self.source}_{split_identifier}.npz"
        else:
            filename = f"split_{self.source}.npz"
        split_dict = np.load(osp.join(self.raw_dir, filename))

        assert split in split_dict, f"Split '{split}' not in split file"

        indices = split_dict[split]
        indices = torch.as_tensor(indices, dtype=torch.long)
        
        logger.info(f"Applying split '{split}' with {len(indices)} samples.")

        # re-slices data & slices correctly
        self.data, self.slices = self.collate([self.get(i) for i in indices])
        
    @property
    def raw_file_names(self):
        return [f'{self.source}.xyz']
    
    @property
    def processed_file_names(self):
        return [f'{self.source}.pt']

    def process(self):
        data_path = osp.join(self.raw_dir, self.raw_file_names[0])
        molecules = read(data_path, index=':')
        
        data_list = []
        unqiue_conformer_formulas = sorted(list(set(mol.get_chemical_formula() for mol in molecules)))
        for i in tqdm(range(0, len(molecules)), desc="Processing molecules"):
            mol = molecules[i]
            formula = mol.get_chemical_formula()
            data = Data(
                x = torch.tensor(mol.numbers, dtype=torch.float),
                num_atoms = torch.tensor(len(mol), dtype=torch.long),
                pos = torch.tensor(mol.positions, dtype=torch.float),
                identifier = torch.tensor(mol.info[self.identifier], dtype=torch.long),
                formula = torch.tensor(unqiue_conformer_formulas.index(formula), dtype=torch.long),
            )

            if mol.info.get("smiles") is not None:
                data.smiles = mol.info["smiles"]

            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)
        
        # Sort data_list by identifier key
        data_list.sort(key=lambda data: data.identifier.item())
        torch.save(self.collate(data_list), self.processed_paths[0])
        
    def get_ase_atoms(self, idx):
        data = self.get(idx)
        atoms = inputs_to_atoms(data)
        return atoms
    
    def get_dataset_as_atoms(self):
        atoms_list = []
        for idx in range(len(self)):
            atoms = self.get_ase_atoms(idx)
            atoms_list.append(atoms)
        return atoms_list
    
    
class ConformerDataset(InMemoryDataset):
    def __init__(
        self,
        source,
        root,
        split=None,              # 'train' | 'val' | 'test'
        split_identifier: Optional[str] = None,
        transform=None,
        pre_transform=Compose([RemoveCOMConformer(), FeaturizeMolecule()]),
        pre_filter=None,
        sort_by_boltzmann_weight=True,
        keep_top_n=30,
    ):  
        self.source = source
        
        if sort_by_boltzmann_weight and split in ["train", "val"]:
            if transform is None:
                transform = Compose([BoltzmannWeightingConformers(keep_top_n=keep_top_n)])
            else:
                transform.transforms.append(BoltzmannWeightingConformers(keep_top_n=keep_top_n))
                
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        logger.info(f"Loaded dataset from {self.processed_paths[0]} with {self.len()} samples.")
        self._cache_indices()
        
        # load split if specified
        if split is not None:
            self._apply_split(split, split_identifier)
            
    def _cache_indices(self):
        self.comp_to_indices = defaultdict(list)
        self.file_identifier_to_indices = {}
        for idx in range(len(self)):
            data = self.get(idx)
            self.comp_to_indices[data.formula].append(idx)
            self.file_identifier_to_indices[data.file_identifier] = idx
        self.compositions = sorted(list(self.comp_to_indices.keys()))
        self.file_identifiers = sorted(list(self.file_identifier_to_indices.keys()))
        
    @property
    def raw_file_names(self):
        return glob.glob(osp.join(self.raw_dir, "*.pickle"))
    
    @property
    def processed_file_names(self):
        return [f'{self.source}.pt']
    
    def _apply_split(self, split, split_identifier=None):
        if split_identifier is not None:
            filename = f"split_{self.source}_{split_identifier}.npz"
        else:
            filename = f"split_{self.source}.npz"
        split_dict = np.load(osp.join(self.raw_dir, filename))

        assert split in split_dict, f"Split '{split}' not in split file"

        indices = split_dict[split]

        logger.info(f"Applying split '{split}' with {len(indices)} samples.")

        # re-slices data & slices correctly
        if isinstance(indices[0], str):
            self.data, self.slices = self.collate(
                [self.get(self.file_identifier_to_indices[i]) for i in indices]
            )
        else:
            self.data, self.slices = self.collate(
                [self.get(i) for i in indices]
            )
        
    def get_ase_atoms(self, idx):
        data = self.get(idx)
        # [num_conformers, num_atoms, 3]
        positions = data.pos.view(
            -1, len(data.x), 3
        ) 
        atoms_list = []
        for conformer_id in range(positions.shape[0]):
            conformer_pos = positions[conformer_id]
            data_conformer = data.clone()
            data_conformer.pos = conformer_pos
            atoms = inputs_to_atoms(data_conformer, info_keys=["smiles"])
            atoms.info["conformer_id"] = conformer_id
            atoms_list.append(atoms)
        return atoms_list
    
    def get_dataset_as_atoms(self):
        atoms_list = []
        for idx in range(len(self)):
            atoms = self.get_ase_atoms(idx)
            atoms_list.extend(atoms)
        return atoms_list
    
    def process(self):
        data_list = []
        for pkl_path in tqdm(self.raw_paths, desc="Processing conformer pickles"):
            data = self.process_mol(pkl_path)
            if data is not None:
                data.file_identifier = os.path.basename(pkl_path).split(".pickle")[0]
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
        
    def process_mol(self, pkl_path):
        
        # Load molecule data
        mol_dict = load_pkl(pkl_path)
        confs = mol_dict["conformers"]
        mols = [conf["rd_mol"] for conf in confs]
        
        # Get SMILES with atom indices (use first mol as they're all same)
        mol = mols[0]
        smiles = dm.to_smiles(
            mol,
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )
        formula = rdMolDescriptors.CalcMolFormula(mol)
        
        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
        )
        atomic_charges = torch.tensor(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.long
        )

        # Collect conformer positions and energies
        positions = []
        energies = []
        boltzmann_weights = []

        for conf in confs:
            mol = conf["rd_mol"]
            pos = torch.from_numpy(mol.GetConformer().GetPositions()).float()
            energy = torch.tensor([conf["totalenergy"]]).float()
            weight = torch.tensor([conf["boltzmannweight"]]).float()

            positions.append(pos)
            energies.append(energy)
            boltzmann_weights.append(weight)

        # Stack conformer data
        positions = torch.stack(positions)  # [num_conformers, num_atoms, 3]
        energies = torch.stack(energies)  # [num_conformers, 1]
        boltzmann_weights = torch.stack(boltzmann_weights)  # [num_conformers, 1]

        # Check for disconnected components
        components = check_disconnected_components(mol)
        
        if len(components) > 1:
            logger.warning(f"Skipping {smiles} due to disconnected components")
            return None
        
        data = Data(
            x=atomic_numbers,  # [num_atoms]
            num_atoms = torch.tensor(len(atomic_numbers), dtype=torch.long),
            charges=atomic_charges,  # [num_atoms]
            pos=positions.view(-1, 3),  # [num_conformers*num_atoms, 3]
            energy=energies,  # [num_conformers, 1]
            boltzmann_weights=boltzmann_weights,  # [num_conformers, 1]
            smiles=smiles,
            formula=formula,
            num_conformers=positions.shape[0],
        )
        
        return data