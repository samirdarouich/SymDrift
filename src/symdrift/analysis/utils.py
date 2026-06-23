import datamol as dm
import numpy as np
import torch
from ase import Atoms
from rdkit import Chem
from rdkit.Chem.rdchem import Conformer
from rdkit.Geometry import Point3D

__all__ = [
    "inputs_to_atoms",
    "batch_inputs_to_atoms",
    "get_mol_with_conformer",
    "build_conformer",
    "add_predictions",
]

def inputs_to_atoms(inputs, atom_key="x", pos_key="pos", info_keys=[]):
    """
    Converts a single input to an ASE Atoms object.

    Args:
        inputs: The input in PyTorch geometric format.
        pos_key (str): The key in the input that contains the positions.
    Returns:
        Atoms: The ASE Atoms object.
    """
    R = inputs[pos_key].detach().cpu().numpy()
    Z = inputs[atom_key].detach().cpu().numpy()
    info = {}
    for key in info_keys:
        if hasattr(inputs, key):
            item_ = inputs[key]
            if isinstance(item_, str):
                info[key] = item_
            else:
                info[key] = item_.item()
    atoms = Atoms(positions=R, numbers=Z, info=info)
    return atoms


def batch_inputs_to_atoms(
    batch, atom_key="x", pos_key="pos", batch_key="batch", info_keys=[]
):
    """
    Converts a batch of inputs to a list of ASE Atoms objects.

    Args:
        batch: The input batch in PyTorch geometric format.
        pos_key (str): The key in the batch that contains the positions.
    Returns:
        List[Atoms]: The list of ASE Atoms objects.
    """
    atoms_list = []

    for m in batch[batch_key].unique():
        mask = batch[batch_key] == m
        R = batch[pos_key][mask].detach().cpu().numpy()
        Z = batch[atom_key][mask].detach().cpu().numpy()
        info = {}
        for key in info_keys:
            if hasattr(batch, key):
                item_ = batch[key][m]
                if isinstance(item_, str):
                    info[key] = item_
                else:
                    info[key] = item_.item()
        atoms = Atoms(positions=R, numbers=Z, info=info)
        atoms_list.append(atoms)
    return atoms_list


def build_conformer(pos):
    if isinstance(pos, torch.Tensor) or isinstance(pos, np.ndarray):
        pos = pos.tolist()

    conformer = Conformer()

    for i, atom_pos in enumerate(pos):
        conformer.SetAtomPosition(i, Point3D(*atom_pos))

    return conformer


def get_mol_with_conformer(smiles: str, positions: torch.Tensor) -> Chem.Mol:
    mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
    mol.AddConformer(build_conformer(positions))
    return mol


def add_predictions(batch, total_samples, positions, key="pos_generated"):
    
    # Add pos_generated to batch and add slicing information in case loop through
    # the dataset is done with batch_size > 1.
    num_atoms = batch.num_atoms
    sizes = num_atoms * total_samples
    slices = torch.cat(
        [
            torch.zeros(1, device=sizes.device, dtype=torch.long),
            torch.cumsum(sizes, dim=0),
        ]
    )
    batch._slice_dict[key] = slices
    batch._inc_dict[key] = torch.zeros_like(slices[:-1])
    batch._slice_dict["num_samples"] = batch._slice_dict["num_conformers"]
    batch._inc_dict["num_samples"] = batch._inc_dict["num_conformers"]

    setattr(batch, key, positions)
    batch.num_samples = torch.tensor(
        [total_samples] * batch.num_graphs, dtype=torch.long
    )