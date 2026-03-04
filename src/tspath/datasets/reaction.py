from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
from ase.io import read
from torch_geometric.transforms import Compose
from tspath.datasets.transforms import RemoveCOMReaction, AlignReaction
import logging

logger = logging.getLogger(__name__)

class ReactionDataset(InMemoryDataset):
    def __init__(
        self,
        source,
        root,
        identifier: str = "rxn",
        split=None,              # 'train' | 'val' | 'test'
        transform=None,
        pre_transform=Compose([RemoveCOMReaction(), AlignReaction()]),
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
        assert len(molecules) % 3 == 0, "Data size should be multiple of 3 (R, TS, P)"
        
        data_list = []
        for i in tqdm(range(0, len(molecules), 3), desc="Processing molecules"):
            mol_r = molecules[i]
            mol_ts = molecules[i + 1]
            mol_p = molecules[i + 2]
            
            data = Data(
                x = torch.tensor(mol_ts.numbers, dtype=torch.float),
                num_atoms = torch.tensor(len(mol_ts), dtype=torch.long),
                pos_r = torch.tensor(mol_r.positions, dtype=torch.float),
                pos = torch.tensor(mol_ts.positions, dtype=torch.float),
                pos_p = torch.tensor(mol_p.positions, dtype=torch.float),
                rxn = torch.tensor(mol_ts.info[self.identifier], dtype=torch.long),
            )
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)
        
        # Assert no duplicate rxn keys
        rxn_keys = [data.rxn for data in data_list]
        assert len(rxn_keys) == len(set(rxn_keys)), "Duplicate rxn keys found"
        
        # Sort data_list by rxn key
        data_list.sort(key=lambda data: data.rxn)
        torch.save(self.collate(data_list), self.processed_paths[0])