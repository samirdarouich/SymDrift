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


def get_canonical_elementwise_permutations(canonical_atomic_numbers):
    """
    canonical_atomic_numbers: (n_atoms,) sorted

    Returns
    -------
    perms : (P, n_atoms)
    """
    device = canonical_atomic_numbers.device

    unique_elements = []
    for a in canonical_atomic_numbers.tolist():
        if a not in unique_elements:
            unique_elements.append(a)

    element_indices = [
        torch.where(canonical_atomic_numbers == elem)[0].tolist()
        for elem in unique_elements
    ]

    element_perms = [
        list(itertools.permutations(indices)) for indices in element_indices
    ]

    all_perms = []
    for prod in itertools.product(*element_perms):
        perm = list(itertools.chain(*prod))
        all_perms.append(perm)

    return torch.tensor(all_perms, device=device)  # (P,n)


def get_brute_force_permutations(x, y, atomic_numbers=None):
    """
    Canonicalized permutation pipeline
    """
    device = x.device
    B, n, d = x.shape

    # -------------------------------------------------
    # 1) Canonicalize atom ordering
    # -------------------------------------------------
    if atomic_numbers is not None:
        sort_idx = torch.argsort(atomic_numbers, dim=1)
        inv_sort_idx = torch.argsort(sort_idx, dim=1)

        gather_idx = sort_idx[..., None].expand(-1, -1, d)

        x = torch.gather(x, 1, gather_idx)
        y = torch.gather(y, 1, gather_idx)

        canonical_atomic_numbers = atomic_numbers[0].sort().values

        perms = get_canonical_elementwise_permutations(
            canonical_atomic_numbers
        )  # (P,n)

    else:
        perms = torch.tensor(list(itertools.permutations(range(n))), device=device)

        sort_idx = None
        inv_sort_idx = None

    P = perms.shape[0]

    # -------------------------------------------------
    # 2) Apply permutations
    # -------------------------------------------------

    perms_exp = perms[None, :, :, None].expand(B, P, n, d)

    y_exp = y[:, None].expand(B, P, n, d)

    y_perm = torch.gather(y_exp, 2, perms_exp)

    # -------------------------------------------------
    # 3) Flatten batch
    # -------------------------------------------------

    x_flat = x[:, None].expand(B, P, n, d).reshape(B * P, n, d)
    y_flat = y_perm.reshape(B * P, n, d)

    return x_flat, y_flat, perms, sort_idx, inv_sort_idx


def get_x_y_pairs(x, y, atomic_numbers=None):
    """
    x: (N, n_atoms, d),
    y: (M, n_atoms, d)
    atomic_numbers: (n_atoms,) atomic numbers of each atom in target structure. Only
    permute within same atomic number if provided.

    returns:
    x_flat, y_flat of shape (N*M, n_atoms, d) for all pairs
    """
    assert x.shape[1] == y.shape[1], "X and Y must have same number of atoms"
    N, n_atoms, d = x.shape
    M = y.shape[0]

    # Create all N x M pairs and flatten to (N*M, n_atoms, d)
    x_pairs = x[:, None, :, :].expand(N, M, n_atoms, d)
    y_pairs = y[None, :, :, :].expand(N, M, n_atoms, d)
    x_flat = x_pairs.reshape(N * M, n_atoms, d)
    y_flat = y_pairs.reshape(N * M, n_atoms, d)

    atomic_numbers_b = None
    if atomic_numbers is not None:
        assert atomic_numbers.shape == (M * n_atoms,), (
            f"Atomic numbers should have shape (M*n_atoms,) not {atomic_numbers.shape}"
        )
        atomic_numbers_b = atomic_numbers.view(M, n_atoms).repeat(
            N, 1
        )  # (N*M, n_atoms)

    return x_flat, y_flat, atomic_numbers_b