import glob
import os
import os.path as osp
from collections import defaultdict

import datamol as dm
import torch
from ase.io import read
from rdkit.Chem import rdMolDescriptors
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose
from tqdm import tqdm

from tspath.datasets.transforms import (
    FeaturizeMolecule,
    RandomPermute,
    RandomRotate,
    RemoveCOM,
    RemoveCOMConformer,
)
from tspath.datasets.utils import (
    ConformerData,
    check_disconnected_components,
    filter_mols,
    load_pkl,
)
from tspath.utils import RankedLogger, inputs_to_atoms

logger = RankedLogger(__name__, rank_zero_only=True)


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
            f"Loaded dataset from {self.processed_paths[0]} with {self.len()} samples."
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
        transform=None,
        pre_transform=Compose([RemoveCOMConformer(), FeaturizeMolecule()]),
        pre_filter=None,
        **kwargs,
    ):
        self.source = source
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        logger.info(
            f"Loaded dataset from {self.processed_paths[0]} with {len(self)} samples."
        )
        self._cache_indices()

    def _cache_indices(self):
        self.split_identifier_to_index = dict(
            zip(self.file_identifier, range(len(self.file_identifier)))
        )
        self.split_identifiers = sorted(self.split_identifier_to_index.keys())

    @property
    def raw_file_names(self):
        return glob.glob(osp.join(self.raw_dir, "*.pickle"))

    @property
    def processed_file_names(self):
        return [f"{self.source}.pt"]

    def get_ase_atoms(self, idx):
        data = self.get(idx)
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
            num_conformers=num_conformers,
            smiles=smiles,
            formula=formula,
        )

        return data
