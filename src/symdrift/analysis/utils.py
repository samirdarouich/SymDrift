import logging
import math
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence, Union

import datamol as dm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import torch
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read

from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Geometry import Point3D
from sklearn.decomposition import PCA
from tqdm import tqdm

from symdrift.datasets import ToyMoleculeDataset
from symdrift.utils import RankedLogger, build_conformer

logging.getLogger("pymatgen.analysis.molecule_matcher").setLevel(logging.WARNING)

logger = RankedLogger(__name__, rank_zero_only=True)


def inputs_to_atoms(inputs, pos_key="pos", info_keys=[]):
    """
    Converts a single input to an ASE Atoms object.

    Args:
        inputs: The input in PyTorch geometric format.
        pos_key (str): The key in the input that contains the positions.
    Returns:
        Atoms: The ASE Atoms object.
    """
    R = inputs[pos_key].detach().cpu().numpy()
    Z = inputs.x.detach().cpu().numpy()
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


def batch_inputs_to_atoms(batch, atom_key="x", pos_key="pos", batch_key="batch", info_keys=[]):
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