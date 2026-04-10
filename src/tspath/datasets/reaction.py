import os.path as osp

import torch
from ase.io import read
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import Compose
from tqdm import tqdm

from tspath.datasets.transforms import (
    AlignReaction,
    FeaturizeReaction,
    RemoveCOMReaction,
    TargetReaction,
)
from tspath.datasets.utils import ConformerData
from tspath.utils import RankedLogger, inputs_to_atoms

logger = RankedLogger(__name__, rank_zero_only=True)


class ReactionDataset(InMemoryDataset):
    def __init__(
        self,
        source,
        root,
        identifier: str = "rxn",
        transform=None,
        pre_transform=Compose(
            [
                RemoveCOMReaction(),
                AlignReaction(),
                TargetReaction(),
                FeaturizeReaction(),
            ]
        ),
        pre_filter=None,
    ):
        self.identifier = identifier
        self.source = source
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self._cache_indices()  # Cache rxn identifiers for quick access
        logger.info(
            f"Loaded dataset from <{self.processed_paths[0]}> with {self.len()} samples."
        )

    def _cache_indices(self):
        self.split_identifier_to_index = dict(
            zip(self.rxn.tolist(), range(len(self.rxn)))
        )
        self.split_identifiers = sorted(self.split_identifier_to_index.keys())

    @property
    def raw_file_names(self):
        return [f"{self.source}.xyz"]

    @property
    def processed_file_names(self):
        return [f"{self.source}.pt"]

    def get_ase_atoms(self, idx, pos_key="pos_ts"):
        data = self[idx]
        # [num_conformers, num_atoms, 3]
        positions = getattr(data, pos_key).view(-1, len(data.x), 3)
        atoms_list = []
        for conformer_id in range(positions.shape[0]):
            conformer_pos = positions[conformer_id]
            data_conformer = data.clone()
            data_conformer.pos = conformer_pos
            atoms = inputs_to_atoms(data_conformer, info_keys=["rxn"])
            atoms.info["conformer_id"] = conformer_id
            atoms_list.append(atoms)
        return atoms_list

    def get_dataset_as_atoms(self, pos_key="pos_ts"):
        atoms_list = []
        for idx in range(len(self)):
            atoms = self.get_ase_atoms(idx, pos_key=pos_key)
            atoms_list.extend(atoms)
        return atoms_list
    
    def process(self):
        data_path = osp.join(self.raw_dir, self.raw_file_names[0])
        molecules = read(data_path, index=":")
        assert len(molecules) % 3 == 0, "Data size should be multiple of 3 (R, TS, P)"

        data_list = []
        for i in tqdm(range(0, len(molecules), 3), desc="Processing molecules"):
            mol_r = molecules[i]
            mol_ts = molecules[i + 1]
            mol_p = molecules[i + 2]

            r_smiles = mol_r.info.get("smiles", None)
            p_smiles = mol_p.info.get("smiles", None)
            smiles = None
            if r_smiles is not None and p_smiles is not None:
                smiles = f"{r_smiles}>>{p_smiles}"
            # make the dataobject in a "conformer" friendly way
            num_atoms = len(mol_ts)
            num_conformers = 1
            data = ConformerData(
                x=torch.tensor(mol_ts.numbers, dtype=torch.float),
                x_conf=torch.tensor(
                    mol_ts.numbers, dtype=torch.float
                ),  # (num_conformers*n_atoms)
                num_atoms=torch.tensor(num_atoms, dtype=torch.long),
                pos_r=torch.tensor(mol_r.positions, dtype=torch.float),
                pos_ts=torch.tensor(mol_ts.positions, dtype=torch.float),
                pos_p=torch.tensor(mol_p.positions, dtype=torch.float),
                rxn=torch.tensor(mol_ts.info[self.identifier], dtype=torch.long),
                conformer_index=torch.arange(num_conformers).repeat_interleave(
                    num_atoms
                ),  # [num_conformers*num_atoms]
                num_conformers=num_conformers,
                r_smiles=mol_r.info.get("smiles", None),
                p_smiles=mol_p.info.get("smiles", None),
                smiles=smiles,
                formula=mol_ts.get_chemical_formula(),
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
