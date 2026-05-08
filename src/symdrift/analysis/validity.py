import os
import pickle
from typing import Optional

import numpy as np
from ase.data import chemical_symbols
from tqdm import tqdm

from symdrift.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = [
    "check_validity",
    "get_validity"
]

# from https://github.com/ehoogeboom/e3_diffusion_for_molecules
bonds1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "Br": 194,
        "I": 214,
    },
    "N": {
        "H": 101,
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
        "Cl": 175,
        "Br": 214,
        "S": 168,
        "I": 222,
        "P": 177,
    },
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Br": 172,
        "S": 151,
        "P": 163,
        "Si": 163,
        "Cl": 164,
        "I": 194,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "S": 158,
        "Si": 160,
        "Cl": 166,
        "Br": 178,
        "P": 156,
        "I": 187,
    },
    "B": {"H": 119, "Cl": 175},
    "Si": {
        "Si": 233,
        "H": 148,
        "C": 185,
        "O": 163,
        "S": 200,
        "F": 160,
        "Cl": 202,
        "Br": 215,
        "I": 243,
    },
    "Cl": {
        "Cl": 199,
        "H": 127,
        "C": 177,
        "N": 175,
        "O": 164,
        "P": 203,
        "S": 207,
        "B": 175,
        "Si": 202,
        "F": 166,
        "Br": 214,
    },
    "S": {
        "H": 134,
        "C": 182,
        "N": 168,
        "O": 151,
        "S": 204,
        "F": 158,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {
        "Br": 228,
        "H": 141,
        "C": 194,
        "O": 172,
        "N": 214,
        "Si": 215,
        "S": 225,
        "F": 178,
        "Cl": 214,
        "P": 222,
    },
    "P": {
        "P": 221,
        "H": 144,
        "C": 184,
        "O": 163,
        "Cl": 203,
        "S": 210,
        "F": 156,
        "N": 177,
        "Br": 222,
    },
    "I": {
        "H": 161,
        "C": 214,
        "Si": 243,
        "N": 222,
        "O": 194,
        "S": 234,
        "F": 187,
        "I": 266,
    },
    "As": {"H": 152},
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}


bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}

allowed_bonds_dict = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
}


def generate_bonds_data(save_path: Optional[str] = None, overwrite: bool = False):
    """
    generate the bonds data as connectivity matrix between possible atoms.

    Args:
        save_path: path to save the data
        overwrite: overwrite existing data
    """
    save_path = save_path or f"{os.path.dirname(__file__)}/bonds.pkl"

    if os.path.exists(save_path) and not overwrite:
        logger.info("Bonds data already exists, skipping generation and reloading...")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    atoms = np.array(chemical_symbols)
    indices = np.arange(len(atoms))
    m_bonds_1 = np.ones((len(atoms), len(atoms))) * -np.inf
    m_bonds_2 = m_bonds_1.copy()
    m_bonds_3 = m_bonds_1.copy()
    allowed_bonds = np.zeros((len(atoms)), dtype=np.int32)

    # define the bonds types and allowed bonds per atom
    for at in atoms:
        for at2 in atoms:
            if at in bonds1 and at2 in bonds1[at]:
                m_bonds_1[indices[atoms == at], indices[atoms == at2]] = (
                    bonds1[at][at2] / 100.0
                )
            if at in bonds2 and at2 in bonds2[at]:
                m_bonds_2[indices[atoms == at], indices[atoms == at2]] = (
                    bonds2[at][at2] / 100.0
                )
            if at in bonds3 and at2 in bonds3[at]:
                m_bonds_3[indices[atoms == at], indices[atoms == at2]] = (
                    bonds3[at][at2] / 100.0
                )
        if at in allowed_bonds_dict:
            allowed_bonds[indices[atoms == at]] = allowed_bonds_dict[at]

    data = {
        "bonds_1": m_bonds_1,
        "bonds_2": m_bonds_2,
        "bonds_3": m_bonds_3,
        "allowed_bonds": allowed_bonds,
    }

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    return data


def squared_euclidean_distance(a, b):
    """
    Efficiently compute the squared Euclidean distance between two sets of points.

    Args:
        a: first set of points
        b: second set of points
    """
    distance = (
        (a**2).sum(axis=1)[:, None] - 2 * np.dot(a, b.T) + (b**2).sum(axis=1)[None]
    )

    return np.where(distance < 0, np.zeros(distance.shape), distance)


def check_validity(
    atoms,
    progress_bar=False,
):
    """
    Fast check for the validity of molecules in a batch, including mol connectivity.

    Args:
        atoms: batch of ase atoms
        bonds_relaxation: relaxation coefficients for the covalent radii
        progress_bar: show tqdm progress bar
    """
    with open(f"{os.path.dirname(__file__)}/bonds.pkl", "rb") as f:
        bonds_data = pickle.load(f)
    m_bonds_1 = bonds_data["bonds_1"]
    m_bonds_2 = bonds_data["bonds_2"]
    m_bonds_3 = bonds_data["bonds_3"]
    allowed_bonds = bonds_data["allowed_bonds"]

    # set default covalent radii relaxation coefficients
    bonds_relaxation = [0.1, 0.05, 0.03]

    bonds = []
    stable_atoms = []
    stable_molecules = []
    stable_atoms_wo_h = []
    stable_molecules_wo_h = []
    connected = []
    connected_wo_h = []

    # loop over molecules in the batch
    for atom in tqdm(atoms, disable=not progress_bar):
        # get the atomic numbers and positions for the current molecule
        R = atom.positions
        Z = atom.numbers

        # get covalent radii for the atoms in the current molecule
        ex_bonds_1 = m_bonds_1[Z[None], Z[:, None]]
        ex_bonds_2 = m_bonds_2[Z[None], Z[:, None]]
        ex_bonds_3 = m_bonds_3[Z[None], Z[:, None]]

        # compute distance matrix
        dist = squared_euclidean_distance(R, R) ** 0.5
        np.fill_diagonal(dist, np.inf)

        # get bond types per atom
        bonds_ = np.where(dist < ex_bonds_1 + bonds_relaxation[0], 1, 0)
        bonds_ = np.where(dist < ex_bonds_2 + bonds_relaxation[1], 2, bonds_)
        bonds_ = np.where(dist < ex_bonds_3 + bonds_relaxation[2], 3, bonds_)

        bonds.append(bonds_)

        # check if molecule is stable
        total_bonds = bonds_.sum(1)
        stable_at = allowed_bonds[Z] == total_bonds
        stable_atoms.append(stable_at)
        stable_molecules.append(stable_at.all())

        # check if molecule is stable without hydrogen
        stable_at_wo_h = stable_at.copy()
        stable_at_wo_h[Z == 1] = True
        stable_atoms_wo_h.append(stable_at_wo_h)
        stable_molecules_wo_h.append(stable_at_wo_h.all())

        # check if ALL the molecule is connected
        # using the exponent of the adjacency matrix trick
        bonds_t = (bonds[-1]) + np.eye(bonds[-1].shape[0])
        bonds_t = bonds_t > 0
        for i in range(bonds_t.shape[0]):
            bonds_t = bonds_t.dot(bonds_t)
        connected.append(bonds_t.all(1).any())

        # check if molecule is connected without hydrogen
        bonds_t[:, Z == 1] = True
        connected_wo_h.append(bonds_t.all(1).any())

    results = {
        "bonds": bonds,
        "stable_atoms": stable_atoms,
        "stable_molecules": stable_molecules,
        "connected": connected,
        "stable_atoms_wo_h": stable_atoms_wo_h,
        "stable_molecules_wo_h": stable_molecules_wo_h,
        "connected_wo_h": connected_wo_h,
    }

    return results


def get_validity(atoms, progress_bar=False):
    """
    Get the validity of molecules in a batch, including mol connectivity.

    Args:
        atoms: batch of ase atoms
        progress_bar: show tqdm progress bar
    """
    validity_res = check_validity(atoms, progress_bar=progress_bar)
    stable_ats = np.concatenate(validity_res["stable_atoms"])
    stable_mols = np.array(validity_res["stable_molecules"])
    stable_ats_wo_h = np.concatenate(validity_res["stable_atoms_wo_h"])
    stable_mols_wo_h = np.array(validity_res["stable_molecules_wo_h"])
    connected = np.array(validity_res["connected"])
    connected_wo_h = np.array(validity_res["connected_wo_h"])

    # infer metrics from validity results
    return {
        "frac_stable_atoms": stable_ats.mean(),
        "frac_stable_molecules": stable_mols.mean(),
        "frac_stable_atoms_wo_h": stable_ats_wo_h.mean(),
        "frac_stable_molecules_wo_h": stable_mols_wo_h.mean(),
        "frac_connected_molecules": connected.mean(),
        "frac_connected_molecules_wo_h": connected_wo_h.mean(),
    }
