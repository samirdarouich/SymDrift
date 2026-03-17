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
from tspath.datasets.transforms import RandomPermute, RandomRotate, RemoveCOM, FeaturizeMolecule
from tspath.utils import inputs_to_atoms

logger = logging.getLogger(__name__)

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