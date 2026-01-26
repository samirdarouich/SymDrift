from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
from ase.io import read
from torch_geometric.transforms import BaseTransform, Compose
import logging

logger = logging.getLogger(__name__)
    
class RemoveCOM(BaseTransform):
    def forward(self, data):
        for pos_key in ['pos']:
            pos = data[pos_key]
            com = pos.mean(dim=0, keepdim=True)
            data[pos_key] = pos - com
        return data

class ReactionDataset(InMemoryDataset):
    def __init__(
        self,
        source,
        root,
        identifier: str = "rxn",
        split=None,              # 'train' | 'val' | 'test'
        transform=None,
        pre_transform=Compose([RemoveCOM()]),
        pre_filter=None,
    ):
        self.identifier = identifier
        self.source = source
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        logger.info(f"Loaded dataset from {self.processed_paths[0]} with {self.len()} samples.")
        
        # load split if specified
        if split is not None:
            self._apply_split(split)

    def _apply_split(self, split):
        split_dict = np.load(osp.join(self.raw_dir, f"split_{self.source}.npz"))

        assert split in split_dict, f"Split '{split}' not in split file"

        indices = split_dict[split]
        indices = torch.as_tensor(indices, dtype=torch.long)
        
        logger.info(f"Applying split '{split}' with {len(indices)} samples.")

        # re-slices data & slices correctly
        self.data, self.slices = self.collate(
            [self.get(i) for i in indices]
        )
        
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
        for i in tqdm(range(0, len(molecules)), desc="Processing molecules"):
            mol = molecules[i]
            data = Data(
                x = torch.tensor(mol.numbers, dtype=torch.float),
                num_atoms = torch.tensor(len(mol), dtype=torch.long),
                pos = torch.tensor(mol.positions, dtype=torch.float),
                rxn = torch.tensor(mol.info[self.identifier], dtype=torch.long),
            )
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)
        
        # Assert no duplicate rxn keys
        rxn_keys = [data.rxn.item() for data in data_list]
        assert len(rxn_keys) == len(set(rxn_keys)), "Duplicate rxn keys found"
        
        # Sort data_list by rxn key
        data_list.sort(key=lambda data: data.rxn.item())
        torch.save(self.collate(data_list), self.processed_paths[0])