import logging
import math
import os
import pickle
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import torch
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read
from pymatgen.analysis.molecule_matcher import (
    BruteForceOrderMatcher,
    GeneticOrderMatcher,
    HungarianOrderMatcher,
    KabschMatcher,
)
from pymatgen.core import Molecule
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from sklearn.decomposition import PCA
from tqdm import tqdm
from tspath.datasets import ToyMoleculeDataset
from rdkit.Geometry import Point3D

logger = logging.getLogger(__name__)

__all__ = [
    "generate_bonds_data",
    "check_validity",
    "get_validity",
    "pymatgen_rmse",
    "pymatgen_match",
    "visualize_atoms_list",
    "visualize_reaction",
    "evaluate_toy",
    "calc_coverage_recall",
    "calc_coverage_precision",
    "calc_amr_recall",
    "calc_amr_precision",
    "print_covmat_results",
    "evaluate_covmat",
    "worker_fn_rmsd",
    "worker_fn_rmsd_wo_h",
    "worker_fn_rmsd_rdkit",
    "worker_fn_rmsd_rdkit_wo_h",
    "worker_fn_distance",
    "pca_plot",
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


def rmse_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        aligned, rmse = bfm.fit(mol2)
        return rmse, aligned
    total_permutations = 1
    for c in count:
        total_permutations *= math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        aligned, rmse = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmse = threshold
        aligned = None
        for pair in pairs:
            if pair[-1] < rmse:
                aligned = pair[0]
                rmse = pair[-1]
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            aligned, rmse = bfm.fit(mol2)
    return rmse, aligned


def pymatgen_rmse(
    mol1,
    mol2,
    ignore_chirality: bool = False,
    threshold: float = 0.5,
    same_order: bool = False,
):
    rmse, aligned = rmse_core(mol1, mol2, threshold, same_order=same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(species=mol2.species, coords=coords)
        rmse_reflect, aligned_reflect = rmse_core(
            mol1, mol2_reflect, threshold, same_order=same_order
        )
        if rmse_reflect < rmse:
            rmse = rmse_reflect
            aligned = aligned_reflect
    return rmse, aligned


def pymatgen_match(
    ref, sample, ignore_chirality=False, threshold=0.5, same_order=False
):
    mol_pred = Molecule(
        species=sample.numbers,
        coords=sample.positions,
    )
    mol_ref = Molecule(
        species=ref.numbers,
        coords=ref.positions,
    )

    rmse, aligned = pymatgen_rmse(
        mol_ref,
        mol_pred,
        ignore_chirality=ignore_chirality,
        threshold=threshold,
        same_order=same_order,
    )

    # pymatgen computes rmse instead of rmsd
    rmsd = rmse * 3**0.5

    # sample is aligned and permuted to match ref
    aligned_sample = sample.copy()
    aligned_sample.numbers = ref.numbers
    aligned_sample.positions = aligned.cart_coords
    return rmsd, aligned_sample


def atoms_to_xyz_text(atoms: Atoms):
    xyz_str = f"{len(atoms)}\n\n"
    for atom, pos in zip(atoms, atoms.positions):
        xyz_str += f"{atom.symbol} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n"  # type: ignore
    return xyz_str


def visualize_atoms_list(
    atoms_list: Sequence[Union[Atoms, str]],
    colors: Optional[List[str]] = None,
    style_dicts: Optional[List[Dict[str, Any]]] = None,
) -> py3Dmol.view:
    """
    Visualizes a list of atomic structures represented by Atoms objects using py3Dmol.

    Args:
        atoms_list: List of atoms objects | xyz paths representing atomic structures to
            visualize.
        colors: List of colors to assign to each structure.
        style_dicts: List of style dictionaries to assign to each structure.

    Returns:
      py3Dmol.view:
        The html view object of py3Dmol.
    """
    xyzs = []
    for atoms in atoms_list:
        if isinstance(atoms, Atoms):
            xyzs.append(atoms_to_xyz_text(atoms))
        elif isinstance(atoms, str):
            if ".xyz" not in atoms:
                raise ValueError("Expected xyz file!.")
            with open(atoms) as f:
                xyzs.append(f.read())
        else:
            raise ValueError("Either specify atoms object or xyz file")

    view = py3Dmol.view(width=800, height=400)

    default_style = {"stick": {}, "sphere": {"radius": 0.36}}
    for i, xyz in enumerate(xyzs):
        view.addModel(xyz, "xyz")
        if style_dicts is not None:
            style_dict = style_dicts[i]
        else:
            style_dict = default_style

        if colors is not None:
            if "stick" not in style_dict:
                style_dict["stick"] = {"color": colors[i]}
            else:
                style_dict["stick"].update({"color": colors[i]})

        view.setStyle(
            {"model": i},
            style_dict,
        )
    view.zoomTo()
    return view


def visualize_reaction(atoms_list: Sequence[Union[Atoms, str]], offset: float = 5.0):
    """
    Visualize a chemical reaction given a list of ASE `Atoms` or paths to xyz files.

    This function takes a list of atomic structures (from the ASE `Atoms` class)
    and shifts each structure along one axis by a specified offset, relative to
    its index in the list. The function then visualizes the shifted atomic structures.

    Args:
        atoms_list: A list of atomic structures to visualize. Each element is an ASE
            `Atoms` object or path to a structure file that ASE can read.
        offset: The distance to shift each atomic structure. The i-th structure in the
            list is shifted by `i * offset`.

    Returns:
        py3Dmol.view:
          The html view object of py3Dmol.
    """
    shifted_atoms = []
    for i, atoms in enumerate(atoms_list):
        if isinstance(atoms, Atoms):
            shifted_atom = atoms.copy()
        elif isinstance(atoms, str):
            shifted_atom = read(atoms)
        shifted_atom.translate(offset * i)  # type: ignore
        shifted_atoms.append(shifted_atom)
    return visualize_atoms_list(shifted_atoms)


def evaluate_toy(samples, dataset_name, **dataset_kwargs):

    # Get reference dataset to compute the reference distances
    dataset = ToyMoleculeDataset(
        name=dataset_name,
        n_samples=1,
        T=0,
        augment_with_rotations=False,
        augment_with_permutations=False,
        seed=42,
        **dataset_kwargs,
    )

    # --- Compute reference distances ---
    ref_sample = dataset[0].pos.to(samples.device)
    n_atoms = ref_sample.shape[0]
    ref_dist = torch.cdist(ref_sample.unsqueeze(0), ref_sample.unsqueeze(0))[0]
    idx = torch.triu_indices(n_atoms, n_atoms, offset=1)
    ref_pairwise = ref_dist[idx[0], idx[1]]
    ref_sorted, _ = torch.sort(ref_pairwise)

    # --- Sample distances ---
    samples_ = samples.reshape(-1, n_atoms, 3)
    distances = torch.cdist(samples_, samples_)
    pairwise = distances[:, idx[0], idx[1]]
    pairwise_sorted, _ = torch.sort(pairwise, dim=1)

    mse = ((ref_sorted - pairwise_sorted) ** 2).mean()

    return mse.item()


# code from https://github.com/ML4MolSim/dit_mc/blob/main/dit_mc/evaluate.py


def calc_coverage_recall(rmsd_array, thresholds):
    """
    Compute coverage recall (COV-R) for a set of generated conformers.

    Coverage recall measures the fraction of reference conformers that are
    successfully reproduced by at least one generated conformer within a given
    RMSD threshold.

    For each reference conformer, the minimum RMSD to any generated conformer
    is computed. A reference conformer is considered "covered" if this minimum
    RMSD is below the specified threshold.
    """
    min_rmsd_per_conf = np.nanmin(rmsd_array, axis=1, keepdims=True)  # (num_confs, 1)
    hits_per_conf = min_rmsd_per_conf < thresholds  # (num_confs, num_thresholds)
    coverage_recall = np.mean(hits_per_conf, axis=0)  # (num_thresholds,)
    return coverage_recall


def calc_coverage_precision(rmsd_array, thresholds):
    """
    Compute coverage precision (COV-P) for a set of generated conformers.

    Coverage precision measures the fraction of generated conformers that
    correspond to at least one reference conformer within a given RMSD
    threshold.

    For each generated conformer, the minimum RMSD to any reference conformer
    is computed. A generated conformer is considered valid if this minimum
    RMSD is below the specified threshold.
    """
    thresholds = np.expand_dims(thresholds, 1)  # (num_thresholds, 1)
    min_rmsd_per_pred = np.nanmin(rmsd_array, axis=0, keepdims=True)  # (1, num_preds)
    hits_per_pred = min_rmsd_per_pred < thresholds  # (num_thresholds, num_preds)
    coverage_precision = np.mean(hits_per_pred, axis=1)  # (num_thresholds,)
    return coverage_precision


def calc_amr_recall(rmsd_array):
    """
    Compute the average minimum RMSD with respect to reference conformers
    (AMR-R). rmsd_array is of shape (num_confs, num_preds).

    For each reference conformer, the minimum RMSD to any generated conformer
    is computed.
    """
    min_rmsd_per_conf = np.nanmin(rmsd_array, axis=1)  # (num_confs,)
    amr_recall = np.mean(min_rmsd_per_conf)
    return amr_recall


def calc_amr_precision(rmsd_array):
    """
    Compute the average minimum RMSD with respect to generated conformers
    (AMR-P). rmsd_array is of shape (num_confs, num_preds).

    For each generated conformer, the minimum RMSD to any reference conformer
    is computed.
    """
    min_rmsd_per_pred = np.nanmin(rmsd_array, axis=0)  # (num_preds,)
    amr_precision = np.mean(min_rmsd_per_pred)
    return amr_precision

def mol_from_ase(ase_atoms):
    atomic_numbers = ase_atoms.numbers
    coords = ase_atoms.positions
    
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atomic_numbers))

    for i, (z, pos) in enumerate(zip(atomic_numbers, coords)):
        atom = Chem.Atom(int(z))
        mol_idx = mol.AddAtom(atom)
        conf.SetAtomPosition(mol_idx, Point3D(*pos))

    mol = mol.GetMol()
    mol.AddConformer(conf)
    return mol


def get_best_rmsd_rdkit(ref_mol, gen_mol, use_alignmol=False):
    ref_mol_rdikit = mol_from_ase(ref_mol)
    gen_mol_rdikit = mol_from_ase(gen_mol)
    try:
        if use_alignmol:
            return rdMolAlign.AlignMol(gen_mol_rdikit, ref_mol_rdikit)
        else:
            rmsd = rdMolAlign.GetBestRMS(gen_mol_rdikit, ref_mol_rdikit)
    except:  # noqa
        rmsd = np.nan

    return rmsd


def worker_fn_rmsd_rdkit(job):
    smiles, i, j, ref_i, pred_j, use_alignmol = job
    rmsd = get_best_rmsd_rdkit(ref_i, pred_j, use_alignmol=use_alignmol)
    return smiles, i, j, rmsd


def worker_fn_rmsd_rdkit_wo_h(job):
    smiles, i, j, ref_i, pred_j, use_alignmol = job
    ref_i_woh = ref_i.copy()
    pred_j_woh = pred_j.copy()
    del ref_i_woh[[atom.index for atom in ref_i_woh if atom.symbol == "H"]]
    del pred_j_woh[[atom.index for atom in pred_j_woh if atom.symbol == "H"]]
    rmsd = get_best_rmsd_rdkit(ref_i_woh, pred_j_woh, use_alignmol=use_alignmol)
    return smiles, i, j, rmsd


def worker_fn_rmsd(job):
    smiles, i, j, ref_i, pred_j, same_order = job
    rmsd, _ = pymatgen_match(ref_i, pred_j, same_order=same_order)
    return smiles, i, j, rmsd

def worker_fn_rmsd_wo_h(job):
    smiles, i, j, ref_i, pred_j, same_order = job
    ref_i_woh = ref_i.copy()
    pred_j_woh = pred_j.copy()
    del ref_i_woh[[atom.index for atom in ref_i_woh if atom.symbol == "H"]]
    del pred_j_woh[[atom.index for atom in pred_j_woh if atom.symbol == "H"]]
    rmsd, _ = pymatgen_match(ref_i_woh, pred_j_woh, same_order=same_order)
    return smiles, i, j, rmsd

def worker_fn_distance(job):
    smiles, i, j, ref_i, pred_j, same_order = job
    pos_i = ref_i.positions
    pos_j = pred_j.positions
    distance = torch.cdist(torch.tensor(pos_i), torch.tensor(pos_j))
    if same_order:
        rmse = torch.sqrt((distance**2).mean()).item()
    else:
        Z = torch.tensor(ref_i.numbers)
        unique_types = torch.unique(Z)
        d = []
        for Zi in unique_types:
            for Zj in unique_types:
                mask_i = (Z == Zi)[:, None]  # (N,1)
                mask_j = (Z == Zj)[None, :]  # (1,N)
                pair_mask = mask_i & mask_j  # (N,N)
                d_ = distance[pair_mask].view(1, -1)
                d_ = torch.sort(d_, dim=1)[0]
                d.append(d_)
        d = torch.cat(d, dim=1)
        rmse = torch.sqrt((d**2).mean()).item()
    return smiles, i, j, rmse


WORKER_FN_DICT = {
    "rmsd": worker_fn_rmsd,
    "rmsd_wo_h": worker_fn_rmsd_wo_h,
    "rmsd_rdkit": worker_fn_rmsd_rdkit,
    "rmsd_rdkit_wo_h": worker_fn_rmsd_rdkit_wo_h,
    "distance": worker_fn_distance,
}

def evaluate_covmat(
    preds, refs, thresholds, num_workers=8, worker_fn_type="rmsd", ratio=None
):
    ref_sample_dict = defaultdict(lambda: defaultdict(list))
    for ref in refs:
        ref_sample_dict[ref.info["smiles"]]["refs"].append(ref)
    for pred in preds:
        smi = pred.info["smiles"]
        # Only keep a certain ratio of predictions per reference
        if ratio is not None:
            if len(ref_sample_dict[smi]["preds"]) >= len(ref_sample_dict[smi]["refs"]) * ratio:
                continue
        ref_sample_dict[pred.info["smiles"]]["preds"].append(pred)

    rmsd_results = {
        smiles: np.ones(
            (
                len(ref_sample_dict[smiles]["refs"]),
                len(ref_sample_dict[smiles]["preds"]),
            )
        )
        * np.nan
        for smiles in ref_sample_dict
    }

    def populate_results(res):
        smiles, i, j, rmsd_val = res
        rmsd_results[smiles][i, j] = rmsd_val

    jobs = []
    for smiles, data in ref_sample_dict.items():
        refs = data["refs"]
        preds = data["preds"]
        for i, refs_i in enumerate(refs):
            for j, preds_j in enumerate(preds):
                jobs.append((smiles, i, j, refs_i, preds_j, False))

    if num_workers > 1:
        p = Pool(num_workers)
        map_fn = partial(p.imap_unordered, chunksize=64)
        p.__enter__()
    else:
        map_fn = map

    for res in tqdm(
        map_fn(WORKER_FN_DICT[worker_fn_type], jobs),
        total=len(jobs),
        desc="Computing RMSD matrix",
    ):
        populate_results(res)

    if num_workers > 1:
        p.__exit__(None, None, None)

    coverage_recall, coverage_precision = [], []
    amr_recall, amr_precision = [], []
    for rmsd_array in rmsd_results.values():
        if rmsd_array.shape[1] == 0:
            continue
        coverage_recall.append(calc_coverage_recall(rmsd_array, thresholds))
        coverage_precision.append(calc_coverage_precision(rmsd_array, thresholds))
        amr_recall.append(calc_amr_recall(rmsd_array))
        amr_precision.append(calc_amr_precision(rmsd_array))

    results = {
        "thresholds": np.array(thresholds),
        "CoverageR": coverage_recall,
        "CoverageP": coverage_precision,
        "MatchingR": amr_recall,
        "MatchingP": amr_precision,
    }

    return results


def print_covmat_results(results, threshold):

    df = pd.DataFrame.from_dict(
        {
            "Threshold": results["thresholds"],
            "COV-R_mean": np.mean(results["CoverageR"], 0),
            "COV-R_median": np.median(results["CoverageR"], 0),
            "COV-P_mean": np.mean(results["CoverageP"], 0),
            "COV-P_median": np.median(results["CoverageP"], 0),
        }
    )

    df["R_mean"] = np.mean(results["MatchingR"])
    df["R_median"] = np.median(results["MatchingR"])
    df["P_mean"] = np.mean(results["MatchingP"])
    df["P_median"] = np.median(results["MatchingP"])

    mask = np.abs(results["thresholds"] - threshold) < 1e-6

    metrics = {
        "threshold": results["thresholds"][mask].item(),
        "COV-R_mean": df["COV-R_mean"][mask]
        .to_numpy()
        .item(),  # xxx of reference conformers are recovered within the RMSD threshold. --> 1-xxx are missed conformers.
        "COV-R_median": df["COV-R_median"][mask].to_numpy().item(),
        "COV-P_mean": df["COV-P_mean"][mask]
        .to_numpy()
        .item(),  # Every generated conformer matches a reference conformer within the threshold.
        "COV-P_median": df["COV-P_median"][mask].to_numpy().item(),
        "AMR-R_mean": np.mean(
            results["MatchingR"]
        ).item(),  # On average, each reference conformer has a generated one within xxx Å RMSD.
        "AMR-R_median": np.median(results["MatchingR"]).item(),
        "AMR-P_mean": np.mean(
            results["MatchingP"]
        ).item(),  # On average, each generated conformer has a reference one within xxx Å RMSD.
        "AMR-P_median": np.median(results["MatchingP"]).item(),
    }

    return df, metrics

def pca_plot(refs, samples, embedder, identifier="smiles", save_path=None):

    unique_identifier_ref = set([atom.info.get(identifier, "Unknown") for atom in refs])
    unique_identifier_samples = set(
        [atom.info.get(identifier, "Unknown") for atom in samples]
    )
    unique_identifier = unique_identifier_ref.intersection(unique_identifier_samples)
    
    if len(unique_identifier) == 0:
        logger.debug(
            f"No common {identifier} values between refs and samples, skipping PCA plot."
        )
        return
    
    # Output of embedder is one long vector per molecule, and a mask that indicates 
    # which positions in the vector correspond to atoms.
    # Get ref embedding
    ref_pos = torch.cat(
        [torch.tensor(atom.get_positions()) for atom in refs], dim=0
    ).float()
    ref_atomic_numbers = torch.cat(
        [torch.tensor(atom.get_atomic_numbers()) for atom in refs], dim=0
    ).float()
    batch_ref = torch.cat([torch.ones(len(atom))* i for i, atom in enumerate(refs)]).long()
    ref_emb, ref_mask = embedder(
        positions=ref_pos, Z=ref_atomic_numbers, batch=batch_ref
    )

    # Get samples embedding
    samples_pos = torch.cat(
        [torch.tensor(atom.get_positions()) for atom in samples], dim=0
    ).float()
    samples_atomic_numbers = torch.cat(
        [torch.tensor(atom.get_atomic_numbers()) for atom in samples], dim=0
    ).float()
    batch_samples = torch.cat([torch.ones(len(atom))* i for i, atom in enumerate(samples)]).long()
    samples_emb, samples_mask = embedder(
        positions=samples_pos, Z=samples_atomic_numbers, batch=batch_samples
    )

    if len(unique_identifier) == 1:
        # if only one unique identifier, plot all samples and ref together (as the 
        # embedder will have same shape for all)
        pca = PCA(n_components=2)
        y_2d = pca.fit_transform(ref_emb.view(len(refs), -1))
        x_2d = pca.transform(samples_emb.view(len(samples), -1))

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle(f"PCA variance: {sum(pca.explained_variance_ratio_):.2f}")
        ax.plot(x_2d[:, 0], x_2d[:, 1], "ro", label="Samples")
        ax.plot(y_2d[:, 0], y_2d[:, 1], "bx", label="Dataset")
        ax.legend()
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    else:
        assert "Unknown" not in unique_identifier, (
            f"{identifier} should be given in atom object"
        )

        if len(unique_identifier) > 18:
            logger.debug(
                f"Too many unique {identifier} values ({len(unique_identifier)}), skipping PCA plot."
            )
            return

        n_cols = min(len(unique_identifier), 3)
        n_rows = max(math.ceil(len(unique_identifier) / n_cols), 1)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        unused_axes = list(range(len(unique_identifier), len(axes)))
        for i, identifier_value in enumerate(unique_identifier):
            ax = axes[i]

            # Get reference embedding
            ref_mask_i = torch.tensor([i for i, atom in enumerate(refs) if atom.info[identifier] == identifier_value])
            ref_emb_mask_i = torch.isin(ref_mask, ref_mask_i)
            ref_emb_i = ref_emb[ref_emb_mask_i]

            if len(ref_mask_i) == 1:
                logger.debug(
                    f"Only one reference sample, {identifier}={identifier_value}, skipping..."
                )
                unused_axes.append(i)
                continue
            
            # Get samples embedding
            samples_mask_i = torch.tensor([i for i, atom in enumerate(samples) if atom.info[identifier] == identifier_value])
            samples_emb_mask_i = torch.isin(samples_mask, samples_mask_i)
            samples_emb_i = samples_emb[samples_emb_mask_i]

            pca = PCA(n_components=2)
            y_2d = pca.fit_transform(ref_emb_i.view(len(ref_mask_i), -1))
            x_2d = pca.transform(samples_emb_i.view(len(samples_mask_i), -1))

            ax.plot(x_2d[:, 0], x_2d[:, 1], "ro", label="Samples")
            ax.plot(y_2d[:, 0], y_2d[:, 1], "bx", label="Dataset")
            ax.set_title(
                f"{identifier_value}\nPCA variance: {sum(pca.explained_variance_ratio_):.2f}"
            )
            ax.legend()
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        # remove unused axes
        for j in sorted(unused_axes, reverse=True):
            fig.delaxes(axes[j])

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    plt.close()
