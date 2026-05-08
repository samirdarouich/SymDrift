import logging
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

import datamol as dm
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Geometry import Point3D
from symdrift.utils import RankedLogger
from sydrift.analysis import pymatgen_match, build_conformer
from tqdm import tqdm

logging.getLogger("pymatgen.analysis.molecule_matcher").setLevel(logging.WARNING)

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = [
    "calc_coverage_recall",
    "calc_coverage_precision",
    "calc_amr_recall",
    "calc_amr_precision",
    "evaluate_covmat",
    "print_covmat_results",
]
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


def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    conformer = build_conformer(pos)
    mol.AddConformer(conformer)
    return mol


def get_best_rmsd_rdkit(ref_mol, gen_mol, use_alignmol=False):
    try:
        if use_alignmol:
            return rdMolAlign.AlignMol(gen_mol, ref_mol)
        else:
            rmsd = rdMolAlign.GetBestRMS(gen_mol, ref_mol)
    except:  # noqa
        rmsd = np.nan

    return rmsd


def worker_fn_rmsd_rdkit(job):
    smiles, i, j, ref_i, pred_j, kwargs = job
    rmsd = get_best_rmsd_rdkit(ref_i, pred_j, **kwargs)
    return smiles, i, j, rmsd


def worker_fn_rmsd_rdkit_wo_h(job):
    smiles, i, j, ref_i, pred_j, kwargs = job
    ref_i_woh = RemoveHs(ref_i)
    pred_j_woh = RemoveHs(pred_j)
    rmsd = get_best_rmsd_rdkit(ref_i_woh, pred_j_woh, **kwargs)
    return smiles, i, j, rmsd


def worker_fn_rmsd(job):
    smiles, i, j, ref_i, pred_j, kwargs = job
    rmsd, _ = pymatgen_match(ref_i, pred_j, **kwargs)
    return smiles, i, j, rmsd


def worker_fn_rmsd_wo_h(job):
    smiles, i, j, ref_i, pred_j, kwargs = job
    ref_i_woh = ref_i.copy()
    pred_j_woh = pred_j.copy()
    del ref_i_woh[[atom.index for atom in ref_i_woh if atom.symbol == "H"]]
    del pred_j_woh[[atom.index for atom in pred_j_woh if atom.symbol == "H"]]
    rmsd, _ = pymatgen_match(ref_i_woh, pred_j_woh, **kwargs)
    return smiles, i, j, rmsd


def worker_fn_distance(job):
    smiles, i, j, ref_i, pred_j, kwargs = job
    pos_i = ref_i.positions
    pos_j = pred_j.positions
    distance = torch.cdist(torch.tensor(pos_i), torch.tensor(pos_j))
    if kwargs.get("same_order"):
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
    preds,
    refs,
    thresholds,
    num_workers=8,
    worker_fn_type="rmsd",
    ratio=None,
    identifier="smiles",
    skip_disconnected=True,
    **job_kwargs,
):
    ref_sample_dict = defaultdict(lambda: defaultdict(list))
    skipped = []
    for ref in refs:
        if "." in ref.info[identifier] and skip_disconnected:
            if ref.info[identifier] not in skipped:
                logger.info(
                    f"Skipping disconnected molecule with {identifier}={ref.info[identifier]} for covmat evaluation."
                )
            skipped.append(ref.info[identifier])
            continue
        ref_sample_dict[ref.info[identifier]]["refs"].append(ref)
    for pred in preds:
        smi = pred.info[identifier]
        # Only keep a certain ratio of predictions per reference
        if ratio is not None:
            if (
                len(ref_sample_dict[smi]["preds"])
                >= len(ref_sample_dict[smi]["refs"]) * ratio
            ):
                continue
        ref_sample_dict[pred.info[identifier]]["preds"].append(pred)

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

        if worker_fn_type in ["rmsd_rdkit", "rmsd_rdkit_wo_h"]:
            mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
            refs = [set_rdmol_positions(mol, ref.positions) for ref in refs]
            preds = [set_rdmol_positions(mol, pred.positions) for pred in preds]

        for i, refs_i in enumerate(refs):
            for j, preds_j in enumerate(preds):
                jobs.append((smiles, i, j, refs_i, preds_j, job_kwargs))

    if num_workers > 1:
        with Pool(num_workers) as p:
            map_fn = partial(p.imap_unordered, chunksize=64)

            for res in tqdm(
                map_fn(WORKER_FN_DICT[worker_fn_type], jobs),
                total=len(jobs),
                desc="Computing RMSD matrix",
            ):
                populate_results(res)
    else:
        for res in tqdm(
            map(WORKER_FN_DICT[worker_fn_type], jobs),
            total=len(jobs),
            desc="Computing RMSD matrix",
        ):
            populate_results(res)

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

    return results, rmsd_results


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
