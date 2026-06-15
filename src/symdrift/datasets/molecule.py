import glob
import os
import os.path as osp
from collections import OrderedDict, defaultdict

import datamol as dm
import torch
from ase.io import read
from rdkit.Chem import rdMolDescriptors
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.transforms import Compose
from tqdm import tqdm

from symdrift.datasets.transforms import (
    FeaturizeMolecule,
    RandomPermute,
    RandomRotate,
    RemoveCOM,
    RemoveCOMConformer,
)
from symdrift.datasets.utils import (
    ConformerData,
    check_disconnected_components,
    filter_mols,
    load_pkl,
)
from symdrift.utils import RankedLogger
from symdrift.analysis import inputs_to_atoms

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = ["MoleculeDataset", "ConformerDatasetInMemory", "ConformerDatasetTest", "ConformerDatasetDisk", "ConformerDatasetFromSMILES"]


class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        source,
        root,
        identifier: str = "rxn",
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

        logger.info(
            f"Loaded dataset from <{self.processed_paths[0]}> with {self.len()} samples."
        )

        self.comp_to_indices = defaultdict(list)
        for idx in range(len(self)):
            self.comp_to_indices[self.formula[idx].item()].append(idx)
        self.compositions = sorted(list(self.comp_to_indices.keys()))

    @property
    def raw_file_names(self):
        return [f"{self.source}.xyz"]

    @property
    def processed_file_names(self):
        return [f"{self.source}.pt"]

    def process(self):
        data_path = osp.join(self.raw_dir, self.raw_file_names[0])
        molecules = read(data_path, index=":")

        data_list = []
        unqiue_conformer_formulas = sorted(
            list(set(mol.get_chemical_formula() for mol in molecules))
        )
        for i in tqdm(range(0, len(molecules)), desc="Processing molecules"):
            mol = molecules[i]
            formula = mol.get_chemical_formula()
            data = Data(
                x=torch.tensor(mol.numbers, dtype=torch.float),
                num_atoms=torch.tensor(len(mol), dtype=torch.long),
                pos=torch.tensor(mol.positions, dtype=torch.float),
                identifier=torch.tensor(mol.info[self.identifier], dtype=torch.long),
                formula=torch.tensor(
                    unqiue_conformer_formulas.index(formula), dtype=torch.long
                ),
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
        data = self[idx]
        atoms = inputs_to_atoms(data)
        return atoms

    def get_dataset_as_atoms(self):
        atoms_list = []
        for idx in range(len(self)):
            atoms = self.get_ase_atoms(idx)
            atoms_list.append(atoms)
        return atoms_list


class ConformerShared:
    """Containing shared logic for both InMemory and Disk-based Conformer Datasets."""

    def get_ase_atoms(self, data):
        # [num_conformers, num_atoms, 3]
        positions = data.pos.view(-1, len(data.x), 3)
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
            atoms = self.get_ase_atoms(self[idx])
            atoms_list.extend(atoms)
        return atoms_list

    def process_mol(self, pkl_path):

        # Load molecule data
        mol_dict = load_pkl(pkl_path)

        # Filter out invalid molecules and conformers (this return also if conformer is an edge case)
        filtered_confs = filter_mols(mol_dict)

        if len(filtered_confs) == 0:
            smiles = mol_dict["smiles"]
            logger.warning(f"Smiles '{smiles}' did not pass the filters. Skipping.")
            return None

        # Get SMILES with atom indices (use first mol as they're all same)
        mol = filtered_confs[0]["rd_mol"]

        # Check for disconnected components
        components = check_disconnected_components(mol)

        if len(components) > 1:
            smiles = mol_dict["smiles"]
            logger.warning(f"Skipping {smiles} due to disconnected components")
            return None

        smiles = dm.to_smiles(
            mol,
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )

        # Check that we can convert back to mol from the smiles (sanity check)
        mol_reverse = dm.to_mol(smiles, remove_hs=False, ordered=True)
        if mol_reverse is None:
            logger.warning(
                f"Could not convert SMILES back to mol for {smiles}. Skipping."
            )
            return None

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
        edge_case_conformers = []

        for conf in filtered_confs:
            mol = conf["rd_mol"]
            pos = torch.from_numpy(mol.GetConformer().GetPositions()).float()
            energy = torch.tensor([conf["totalenergy"]]).float()
            weight = torch.tensor([conf["boltzmannweight"]]).float()

            positions.append(pos)
            energies.append(energy)
            boltzmann_weights.append(weight)
            edge_case_conformers.append(conf["edge_case"])

        # Stack conformer data
        positions = torch.stack(positions)  # [num_conformers, num_atoms, 3]
        energies = torch.stack(energies)  # [num_conformers, 1]
        boltzmann_weights = torch.stack(boltzmann_weights)  # [num_conformers, 1]
        edge_case_conformers = torch.tensor(
            edge_case_conformers, dtype=torch.bool
        )  # [num_conformers]

        num_conformers = positions.shape[0]
        num_atoms = positions.shape[1]
        data = ConformerData(
            x=atomic_numbers,  # [num_atoms]
            x_conf=atomic_numbers.repeat(num_conformers),  # [num_conformers*num_atoms]
            charges=atomic_charges,  # [num_atoms]
            pos=positions.view(-1, 3),  # [num_conformers*num_atoms, 3]
            energy=energies,  # [num_conformers, 1]
            boltzmann_weights=boltzmann_weights,  # [num_conformers, 1]
            conformer_index=torch.arange(num_conformers).repeat_interleave(
                num_atoms
            ),  # [num_conformers*num_atoms]
            num_atoms=torch.tensor(num_atoms, dtype=torch.long),
            edge_case_conformers=edge_case_conformers,
            num_conformers=torch.tensor(num_conformers, dtype=torch.long),
            smiles=smiles,
            formula=formula,
        )

        return data

    def process_test_mol(self, mols: list[dm.Mol]):
        """Process a single test molecule into a relevant PyG data object."""
        try:
            mol = mols[0]
            smiles = dm.to_smiles(
                mol,
                canonical=False,
                explicit_hs=True,
                with_atom_indices=True,
                isomeric=True,
            )

            atomic_numbers = torch.tensor(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
            )
            atomic_charges = torch.tensor(
                [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.long
            )

            # Check that we can convert back to mol from the smiles (sanity check)
            mol_reverse = dm.to_mol(smiles, remove_hs=False, ordered=True)
            if mol_reverse is None:
                logger.warning(
                    f"Could not convert SMILES back to mol for {smiles}. Skipping."
                )
                return None

            formula = rdMolDescriptors.CalcMolFormula(mol)

            # Collect conformer positions if provided
            positions = []
            for mol in mols:
                if mol.GetNumConformers() == 0:
                    continue
                pos = torch.from_numpy(mol.GetConformer().GetPositions()).float()
                positions.append(pos)

            # Stack conformer data
            if len(positions) == 0:
                logger.debug(f"No conformer positions found for {smiles}.")
                positions = torch.zeros((1, len(atomic_numbers), 3), dtype=torch.float)
            else:
                positions = torch.stack(positions)  # [num_conformers, num_atoms, 3]

            num_conformers = positions.shape[0]
            num_atoms = positions.shape[1]
            data = ConformerData(
                x=atomic_numbers,  # [num_atoms]
                x_conf=atomic_numbers.repeat(
                    num_conformers
                ),  # [num_conformers*num_atoms]
                charges=atomic_charges,  # [num_atoms]
                pos=positions.view(-1, 3),  # [num_conformers*num_atoms, 3]
                conformer_index=torch.arange(num_conformers).repeat_interleave(
                    num_atoms
                ),  # [num_conformers*num_atoms]
                num_atoms=torch.tensor(num_atoms, dtype=torch.long),
                num_conformers=torch.tensor(num_conformers, dtype=torch.long),
                smiles=smiles,
                formula=formula,
            )

            return data
        except Exception as e:
            logger.warning(f"Skipping: {smiles} due to {e}")
            return None


class ConformerDatasetInMemory(ConformerShared, InMemoryDataset):
    def __init__(
        self,
        source,
        root,
        transform=None,
        pre_transform=Compose([RemoveCOMConformer(), FeaturizeMolecule()]),
        pre_filter=None,
        **kwargs,
    ):
        self.source = source
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        logger.info(
            f"Loaded dataset from <{self.processed_paths[0]}> with {len(self)} samples."
        )
        self._cache_indices()

    def _cache_indices(self):
        self.split_identifier_to_index = dict(
            zip(self.file_identifier, range(len(self.file_identifier)))
        )
        self.split_identifiers = sorted(self.split_identifier_to_index.keys())

    @property
    def raw_file_names(self):
        files = glob.glob(osp.join(self.raw_dir, "*.pickle"))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No pickle files found in raw directory: {self.raw_dir}"
            )
        return files

    @property
    def processed_file_names(self):
        return [f"{self.source}.pt"]

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


class ConformerDatasetTest(ConformerDatasetInMemory):
    @property
    def raw_file_names(self):
        return [osp.join(self.raw_dir, "test_mols.pkl")]

    @property
    def processed_file_names(self):
        return [f"{self.source}_test.pt"]

    def process(self):
        data_list = []
        test_mols = load_pkl(self.raw_paths[0])
        for test_mol_id, test_mol in tqdm(
            test_mols.items(), desc="Processing test conformers"
        ):
            data = self.process_test_mol(test_mol)
            if data is not None:
                data.file_identifier = test_mol_id
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class ConformerDatasetDisk(ConformerShared, Dataset):
    def __init__(
        self,
        source,
        root,
        transform=None,
        pre_transform=Compose([RemoveCOMConformer(), FeaturizeMolecule()]),
        pre_filter=None,
        shard_size: int = 512,
        shard_cache_size: int = 2,
        **kwargs,
    ):
        self.source = source
        self.shard_size = max(1, int(shard_size))
        self.shard_cache_size = max(0, int(shard_cache_size))
        self._shard_cache = OrderedDict()
        super().__init__(root, transform, pre_transform, pre_filter)

        # After initialization/processing, load the metadata to get dataset length and identifiers
        meta_path = osp.join(self.processed_dir, "meta.pt")
        if osp.exists(meta_path):
            self.meta_dict = torch.load(meta_path, weights_only=False)
            self.file_identifier = self.meta_dict["file_identifier"]
            self.index_to_shard = self.meta_dict["index_to_shard"]
            self.shard_files = self.meta_dict["shard_files"]
            self._cache_indices()
            logger.info(
                f"Loaded dataset from <{self.processed_dir}> with {len(self)} samples."
            )
        else:
            raise FileNotFoundError(
                "Metadata file not found. Processing might have failed."
            )

    def _cache_indices(self):
        self.split_identifier_to_index = dict(
            zip(self.file_identifier, range(len(self.file_identifier)))
        )
        self.split_identifiers = sorted(self.split_identifier_to_index.keys())

    @property
    def raw_file_names(self):
        files = sorted(glob.glob(osp.join(self.raw_dir, "*.pickle")))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No pickle files found in raw directory: {self.raw_dir}"
            )
        return [os.path.basename(f) for f in files]

    @property
    def processed_file_names(self):
        # We use a single metadata file as the completion marker for PyG
        return ["meta.pt"]

    def len(self):
        return self.meta_dict["length"]

    def get(self, idx):
        shard_name, local_idx = self.index_to_shard[idx]
        shard_data = self._load_shard(shard_name)
        data = shard_data[local_idx]
        # Protect cached shard objects from in-place transform mutations.
        if hasattr(data, "clone"):
            return data.clone()
        return data

    def _load_shard(self, shard_name):
        if shard_name in self._shard_cache:
            self._shard_cache.move_to_end(shard_name)
            return self._shard_cache[shard_name]

        shard_path = osp.join(self.processed_dir, shard_name)
        payload = torch.load(shard_path, weights_only=False)
        if isinstance(payload, dict):
            data_list = payload.get("data_list", [])
        elif isinstance(payload, list):
            data_list = payload
        else:
            data_list = [payload]

        if self.shard_cache_size > 0:
            self._shard_cache[shard_name] = data_list
            while len(self._shard_cache) > self.shard_cache_size:
                self._shard_cache.popitem(last=False)

        return data_list

    def process(self):
        shard_idx = 0
        current_shard_data = []
        current_shard_identifiers = []

        shard_files = []
        index_to_shard = []
        file_identifiers = []

        def flush_shard():
            nonlocal shard_idx, current_shard_data, current_shard_identifiers
            if len(current_shard_data) == 0:
                return

            shard_name = f"shard_{shard_idx:06d}.pt"
            shard_payload = {
                "data_list": current_shard_data,
                "file_identifier": current_shard_identifiers,
            }
            torch.save(shard_payload, osp.join(self.processed_dir, shard_name))
            shard_files.append(shard_name)

            for local_idx in range(len(current_shard_data)):
                index_to_shard.append((shard_name, local_idx))

            shard_idx += 1
            current_shard_data = []
            current_shard_identifiers = []

        for pkl_path in tqdm(self.raw_paths, desc="Processing conformer pickles"):
            data = self.process_mol(pkl_path)
            if data is not None:
                identifier = os.path.basename(pkl_path).split(".pickle")[0]
                data.file_identifier = identifier
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                current_shard_data.append(data)
                current_shard_identifiers.append(identifier)
                file_identifiers.append(identifier)

                if len(current_shard_data) >= self.shard_size:
                    flush_shard()

        flush_shard()

        # Save metadata to track total length and file identifiers
        meta_dict = {
            "length": len(file_identifiers),
            "file_identifier": file_identifiers,
            "index_to_shard": index_to_shard,
            "shard_files": shard_files,
            "shard_size": self.shard_size,
        }
        torch.save(meta_dict, osp.join(self.processed_dir, "meta.pt"))


class ConformerDatasetFromSMILES(ConformerShared):
    """Pure in-memory dataset built from a CSV of SMILES strings.

    For each SMILES processes it via ``process_test_mol`` and applies ``pre_transform`` 
    (default: RemoveCOMConformer + FeaturizeMolecule).
    """

    def __init__(
        self,
        csv_path: str,
        smiles_col: str = "smiles",
        transform=None,
        pre_transform=None,
    ):
        import pandas as pd

        self.transform = transform

        if pre_transform is None:
            pre_transform = Compose([RemoveCOMConformer(), FeaturizeMolecule()])

        df = pd.read_csv(csv_path)
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in {csv_path}. Available: {list(df.columns)}")

        self.data_list = []
        for smiles in tqdm(df[smiles_col].tolist(), desc="Processing SMILES"):
            mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
            if mol is None:
                logger.warning(f"Could not parse SMILES: {smiles}. Skipping.")
                continue
            data = self.process_test_mol([mol])
            if data is None:
                continue
            if pre_transform is not None:
                data = pre_transform(data)
            self.data_list.append(data)

        logger.info(f"Built in-memory dataset with {len(self.data_list)} molecules from '{csv_path}'.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
