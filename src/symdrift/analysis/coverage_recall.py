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
from tqdm import tqdm
from ase import Atoms
from ase.io import read
from torch_geometric.data import Data
from symdrift.alignment import minimal_distance_permuted
from symdrift.analysis import build_conformer, pymatgen_match
from symdrift.utils import RankedLogger

logging.getLogger("pymatgen.analysis.molecule_matcher").setLevel(logging.WARNING)

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = [
    "calc_coverage_recall",
    "calc_coverage_precision",
    "calc_amr_recall",
    "calc_amr_precision",
    "mol_from_ase",
    "set_rdmol_positions",
    "get_best_rmsd_rdkit",
    "worker_fn_rmsd_rdkit",
    "worker_fn_rmsd_rdkit_wo_h",
    "worker_fn_rmsd",
    "worker_fn_rmsd_wo_h",
    "worker_fn_rmsd_batched",
    "worker_fn_rmsd_wo_h_batched",
    "load_rmsd_predictions",
    "atoms_lists_to_data_list",
    "evaluate_rmsd_single",
    "evaluate_rmsd_batched",
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


def worker_fn_rmsd_batched(
    x, y, data, brute_force_permutations=False, max_iter=5, tol=1e-2, rotate_before=True
):
    """Compute the RMSD between two sets of conformers x and y, where x is of shape
    (num_samples, num_atoms, 3) and y is of shape (num_refs, num_atoms, 3). The RMSD is
    computed as the minimal RMSD between each sample in x and each reference in y,
    after applying the optimal permutation of atoms and rotation to minimize the RMSD.
    The function returns a tensor of shape (num_samples, num_refs) containing the RMSD
    values.

    Arguments:
        x: Tensor of shape (num_samples, num_atoms, 3) containing the coordinates of
            the generated conformers.
        y: Tensor of shape (num_refs, num_atoms, 3) containing the coordinates of
            the reference conformers.
        data: The original data object containing the atomic numbers and permutations
            for the molecule. This is used to determine which permutations to apply when
            computing the RMSD.
        brute_force_permutations: If True, compute the RMSD for all possible
            permutations of atoms.
        max_iter: The maximum number of iterations to perform when optimizing the
            permutation and rotation iteratively using the Hungarian algorithm.
        tol: The tolerance for convergence when optimizing the permutation and rotation.
        rotate_before: If True, perform an initial rotation to align the centroids of x
            and y before optimizing the permutation and rotation.
    """
    assert x.shape[1] == y.shape[1], "X and Y must have same number of atoms"
    n_refs, n_atoms, _ = y.shape

    permutations = getattr(data, "automorphisms", None)
    atomic_numbers = data.x_conf.view(n_refs, n_atoms)

    rmsd_batch, _ = minimal_distance_permuted(
        x=x,
        y=y,
        permutations=permutations,
        atomic_numbers=atomic_numbers,
        brute_force_permutations=brute_force_permutations,
        max_iter=max_iter,
        tol=tol,
        rotate_before=rotate_before,
    )

    return rmsd_batch


def worker_fn_rmsd_wo_h_batched(
    x, y, data, brute_force_permutations=False, max_iter=5, tol=1e-2, rotate_before=True
):
    """Compute the RMSD between two sets of conformers x and y, where x is of shape
    (num_samples, num_atoms, 3) and y is of shape (num_refs, num_atoms, 3). The RMSD is
    computed as the minimal RMSD between each sample in x and each reference in y,
    after applying the optimal permutation of atoms and rotation to minimize the RMSD.
    The function returns a tensor of shape (num_samples, num_refs) containing the RMSD
    values.

    This function does only consider heavy atoms when computing the RMSD.

    Arguments:
        x: Tensor of shape (num_samples, num_atoms, 3) containing the coordinates of
            the generated conformers.
        y: Tensor of shape (num_refs, num_atoms, 3) containing the coordinates of
            the reference conformers.
        data: The original data object containing the atomic numbers and permutations
            for the molecule. This is used to determine which permutations to apply when
            computing the RMSD.
        brute_force_permutations: If True, compute the RMSD for all possible
            permutations of atoms.
        max_iter: The maximum number of iterations to perform when optimizing the
            permutation and rotation iteratively using the Hungarian algorithm.
        tol: The tolerance for convergence when optimizing the permutation and rotation.
        rotate_before: If True, perform an initial rotation to align the centroids of x
            and y before optimizing the permutation and rotation.
    """
    assert x.shape[1] == y.shape[1], "X and Y must have same number of atoms"
    n_refs, n_atoms, _ = y.shape

    permutations = getattr(data, "automorphisms", None)
    atomic_numbers = data.x_conf.view(n_refs, n_atoms)
    hydrogen_mask = data.x == 1

    # Assumes that the order of atoms in x and y is the same (and x_conf is repeated for
    # each sample in x if there are multiple samples)
    atomic_numbers_woh = atomic_numbers[:, ~hydrogen_mask]
    x_woh = x[:, ~hydrogen_mask]
    y_woh = y[:, ~hydrogen_mask]

    # If permutations are provided, we need to filter out permutations were only
    # hydrogen moves and reindex the remaining permutations to match the new indexing of
    # atoms after removing hydrogens.
    if permutations is not None:
        permutations = permutations.view(-1, n_atoms)

        keep_mask = data.x != 1
        old_to_new = -torch.ones_like(data.x)
        old_to_new[keep_mask] = torch.arange(
            keep_mask.sum(), device=atomic_numbers.device
        )
        perms_woh = old_to_new[permutations]
        perms_woh = perms_woh[:, (perms_woh >= 0).all(dim=0)].unique(dim=0)

        P = perms_woh.shape[0]
        atomic_numbers_woh_repeated = data.x[keep_mask].repeat(P, 1)
        batch_indices = torch.arange(P, device=atomic_numbers.device).unsqueeze(1)
        assert (
            (
                atomic_numbers_woh_repeated[batch_indices, perms_woh]
                == atomic_numbers_woh_repeated
            )
            .all()
            .item()
        ), "Permutations must preserve atomic numbers after removing hydrogens."

        assert x_woh.shape[1] == y_woh.shape[1] == perms_woh.shape[1], (
            f"Number of atoms in x and y must match the number of atoms in permutations. "
            f"Got {x_woh.shape[1]} and {y_woh.shape[1]} atoms, but permutations has {permutations.shape[1]} atoms."
        )
    else:
        perms_woh = None

    rmsd_batch, _ = minimal_distance_permuted(
        x=x_woh,
        y=y_woh,
        permutations=perms_woh,
        atomic_numbers=atomic_numbers_woh,
        brute_force_permutations=brute_force_permutations,
        max_iter=max_iter,
        tol=tol,
        rotate_before=rotate_before,
    )

    return rmsd_batch


WORKER_FN_DICT = {
    "rmsd": worker_fn_rmsd,
    "rmsd_wo_h": worker_fn_rmsd_wo_h,
    "rmsd_rdkit": worker_fn_rmsd_rdkit,
    "rmsd_rdkit_wo_h": worker_fn_rmsd_rdkit_wo_h,
    "rmsd_batched": worker_fn_rmsd_batched,
    "rmsd_wo_h_batched": worker_fn_rmsd_wo_h_batched,
}


def load_rmsd_predictions(path_generated, path_reference=None, device="cpu"):
    """Load generated (and reference) conformers for RMSD-based evaluation.

    Supports two input formats:
      - A single ".pt" file containing a list of PyG `Data` objects, each carrying
        both `pos` (reference conformers) and `pos_generated` (generated
        conformers) for one molecule. `path_reference` must not be given in this
        case.
      - A pair of ".xyz" files: `path_generated` with generated conformers and
        `path_reference` with reference conformers, both read via
        `ase.io.read(path, ":")`.

    Returns:
        preds, refs: For the ".pt" case, `preds` is a `list[Data]` and `refs` is
        `None`. For the ".xyz" case, `preds` and `refs` are both `list[Atoms]`.
        Either return value can be passed straight into `evaluate_covmat`.
    """
    if path_generated.endswith(".pt"):
        if path_reference is not None:
            raise ValueError(
                "path_reference must not be provided when path_generated is a .pt file."
            )
        preds = torch.load(path_generated, weights_only=False, map_location=device)
        refs = None
    elif path_generated.endswith(".xyz"):
        if path_reference is None:
            raise ValueError(
                "path_reference must be provided when path_generated is a .xyz file."
            )
        preds = read(path_generated, ":")
        refs = read(path_reference, ":")
    else:
        raise ValueError(
            f"Unsupported file extension for path_generated: {path_generated}"
        )

    return preds, refs


def atoms_lists_to_data_list(preds_atoms, refs_atoms, identifier="smiles"):
    """Convert a pair of (generated, reference) `ase.Atoms` lists into the
    canonical `list[Data]` shape expected by `evaluate_rmsd_single` and
    `evaluate_rmsd_batched`, where each `Data` bundles all reference (`pos`) and
    generated (`pos_generated`) conformers for one molecule.

    Assumes that, for a given identifier, all reference and generated conformers
    share the same atom count and atom ordering.
    """
    ref_sample_dict = defaultdict(lambda: defaultdict(list))
    for ref in refs_atoms:
        ref_sample_dict[ref.info[identifier]]["refs"].append(ref)
    for pred in preds_atoms:
        ref_sample_dict[pred.info[identifier]]["preds"].append(pred)

    data_list = []
    for identifier_value, group in ref_sample_dict.items():
        refs = group["refs"]
        preds = group["preds"]
        if len(refs) == 0:
            continue

        n_atoms = len(refs[0])
        atomic_numbers = torch.tensor(refs[0].numbers, dtype=torch.long)

        data = Data()
        data.pos = torch.tensor(
            np.stack([ref.positions for ref in refs]).reshape(-1, 3),
            dtype=torch.float,
        )
        data.pos_generated = torch.tensor(
            np.stack([pred.positions for pred in preds]).reshape(-1, 3)
            if len(preds) > 0
            else np.zeros((0, 3)),
            dtype=torch.float,
        )
        data.num_atoms = torch.tensor(n_atoms)
        data.x = atomic_numbers
        data.x_conf = atomic_numbers.repeat(len(refs))
        setattr(data, identifier, identifier_value)
        data_list.append(data)

    return data_list


def evaluate_rmsd_single(
    data_list,
    num_workers=8,
    worker_fn_type="rmsd",
    ratio=None,
    identifier="smiles",
    skip_disconnected=True,
    **job_kwargs,
):
    """Compute the RMSD matrix between reference and generated conformers for
    each molecule in `data_list`, using CPU-based multiprocessing.

    Arguments:
        data_list: Iterable of PyG `Data` objects, each carrying `pos` (reference
            conformers), `pos_generated` (generated conformers), `x` (atomic
            numbers), `num_atoms` and the `identifier` field (e.g. `smiles`).
    """
    assert "batched" not in worker_fn_type, (
        "Batched worker functions are not supported in evaluate_rmsd_single. Use evaluate_rmsd_batched instead."
    )
    assert worker_fn_type in WORKER_FN_DICT, (
        f"Unsupported worker function type: {worker_fn_type}"
    )
    use_rdkit = worker_fn_type in ("rmsd_rdkit", "rmsd_rdkit_wo_h")

    rmsd_results = {}
    jobs = []
    skipped = []

    for d in data_list:
        identifier_value = d[identifier]
        if (
            type(identifier_value) == str
            and "." in identifier_value
            and skip_disconnected
        ):
            if identifier_value not in skipped:
                logger.info(
                    f"Skipping disconnected molecule with {identifier}={identifier_value} for covmat evaluation."
                )
                skipped.append(identifier_value)
            continue

        n_atoms = d.num_atoms.item()
        refs_pos = d.pos.view(-1, n_atoms, 3)
        preds_pos = d.pos_generated.view(-1, n_atoms, 3)
        num_refs = refs_pos.shape[0]
        num_preds = preds_pos.shape[0]

        # Only keep a certain ratio of predictions per reference
        if ratio is not None and num_preds > int(num_refs * ratio):
            num_preds = int(num_refs * ratio)
            preds_pos = preds_pos[:num_preds]

        atomic_numbers = d.x.cpu().numpy()

        # Use Graphautomorphism permutations defined by the SMILES to speed up RMSD
        # computation
        if use_rdkit:
            mol = dm.to_mol(identifier_value, remove_hs=False, ordered=True)
            refs_conf = [set_rdmol_positions(mol, pos) for pos in refs_pos]
            preds_conf = [set_rdmol_positions(mol, pos) for pos in preds_pos]
        else:
            refs_conf = [
                Atoms(positions=pos.cpu().numpy(), numbers=atomic_numbers)
                for pos in refs_pos
            ]
            preds_conf = [
                Atoms(positions=pos.cpu().numpy(), numbers=atomic_numbers)
                for pos in preds_pos
            ]

        rmsd_results[identifier_value] = np.ones((num_refs, num_preds)) * np.nan

        for i, refs_i in enumerate(refs_conf):
            for j, preds_j in enumerate(preds_conf):
                jobs.append((identifier_value, i, j, refs_i, preds_j, job_kwargs))

    def populate_results(res):
        smiles, i, j, rmsd_val = res
        rmsd_results[smiles][i, j] = rmsd_val

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

    logger.info(f"Total RMSD computations: {len(jobs)}")
    
    return rmsd_results


def evaluate_rmsd_batched(
    data_list,
    worker_fn_type="rmsd_batched",
    batch_size=64,
    ratio=None,
    identifier="smiles",
    skip_disconnected=True,
    **job_kwargs,
):
    """Compute the RMSD matrix between reference and generated conformers for
    each molecule in `data_list`, using batched GPU tensor operations.

    Arguments:
        data_list: Iterable of PyG `Data` objects, each carrying `pos` (reference
            conformers), `pos_generated` (generated conformers), `x`/`x_conf`
            (atomic numbers), `num_atoms`, optionally `automorphisms`, and the
            `identifier` field (e.g. `smiles`).
    """
    assert worker_fn_type in ["rmsd_batched", "rmsd_wo_h_batched"], (
        "Only batched worker functions are supported in evaluate_rmsd_batched."
    )
    batch_size = max(batch_size, 1)
    rmsd_results = {}
    total_rmsd_computations = 0
    for d in tqdm(data_list, desc="Processing molecules", position=0):
        identifier_value = d[identifier]
        if type(identifier_value) == str and "." in identifier_value and skip_disconnected:
            logger.info(
                f"Skipping disconnected molecule with {identifier}={identifier_value} for covmat evaluation."
            )
            continue

        n_atoms = d.num_atoms.item()
        refs = d.pos.view(-1, n_atoms, 3)
        preds = d.pos_generated.view(-1, n_atoms, 3)
        num_refs = refs.shape[0]
        num_preds = preds.shape[0]

        if ratio is not None and num_preds > int(num_refs * ratio):
            num_preds = int(num_refs * ratio)
            preds = preds[:num_preds]
        total_rmsd_computations += num_refs * num_preds

        # compute RMSD between generated and reference conformers in batches over
        # samples.
        rmsd_list = []
        for i0 in tqdm(
            range(0, num_preds, batch_size),
            desc="Computing RMSD in batches",
            position=1,
            leave=False,
        ):
            i1 = min(i0 + batch_size, num_preds)
            x_batch = preds[i0:i1]
            rmsd_batch = WORKER_FN_DICT[worker_fn_type](
                x=x_batch,
                y=refs,
                data=d,
                **job_kwargs,
            )
            rmsd_list.append(rmsd_batch)

        rmsd = torch.cat(rmsd_list, dim=0)
        # transpose as its defined over (num_samples, num_refs) and we want
        # (num_refs, num_samples)
        rmsd_results[identifier_value] = rmsd.cpu().numpy().T

    logger.info(f"Total RMSD computations: {total_rmsd_computations}")

    return rmsd_results

def _is_atoms_list(obj):
    return len(obj) > 0 and isinstance(obj[0], Atoms)


def evaluate_covmat(
    preds: list[Atoms] | list[Data],
    refs: list[Atoms] | None = None,
    thresholds=None,
    num_parallel=8,
    worker_fn_type="rmsd",
    ratio=None,
    identifier="smiles",
    skip_disconnected=True,
    **job_kwargs,
):
    """Compute coverage-recall/precision and AMR metrics over a set of molecules.

    Accepts either data format, regardless of which worker_fn_type is chosen:
      - A single ".pt"-style input: `preds` is a `list[Data]`, each carrying both
        `pos` (reference conformers) and `pos_generated` (generated conformers)
        for one molecule. `refs` is not used (pass None or omit).
      - A ".xyz"-style input: `preds`/`refs` are both `list[Atoms]` (generated /
        reference conformers respectively), as returned e.g. by
        `load_rmsd_predictions`. These are converted into the `list[Data]` shape
        above via `atoms_lists_to_data_list` before evaluation.

    Args:
        preds: See above.
        refs: See above.
        thresholds: List of RMSD thresholds to compute coverage metrics at.
        num_parallel: Number of parallel workers / batch size to use for RMSD computation.
        worker_fn_type: Type of worker function to use for RMSD computation. Must be one of
            "rmsd", "rmsd_wo_h", "rmsd_rdkit", "rmsd_rdkit_wo_h", "rmsd_batched", or
            "rmsd_wo_h_batched".
        ratio: If not None, limits the number of predictions per reference conformer to
            at most ratio * num_refs.
        identifier: The key  to use as identifier for grouping conformers. Either present
            via data object or ase.atoms.info. Default is "smiles".
        skip_disconnected: If True, skip molecules that are identified as disconnected
            based on the presence of a "." in the SMILES.
        **job_kwargs: Additional keyword arguments to pass to the worker function.
    """
    if thresholds is None:
        raise ValueError("thresholds must be provided.")
    thresholds = np.asarray(thresholds)

    if _is_atoms_list(preds):
        if refs is None:
            raise ValueError(
                "refs (list of ase.Atoms) must be provided when preds is a list of ase.Atoms."
            )
        data_list = atoms_lists_to_data_list(preds, refs, identifier=identifier)
    else:
        data_list = preds

    is_batched = "batched" in worker_fn_type

    if is_batched:
        rmsd_results = evaluate_rmsd_batched(
            data_list,
            worker_fn_type=worker_fn_type,
            batch_size=num_parallel,
            ratio=ratio,
            identifier=identifier,
            skip_disconnected=skip_disconnected,
            **job_kwargs,
        )
    else:
        rmsd_results = evaluate_rmsd_single(
            data_list,
            num_workers=num_parallel,
            worker_fn_type=worker_fn_type,
            ratio=ratio,
            identifier=identifier,
            skip_disconnected=skip_disconnected,
            **job_kwargs,
        )

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
        "thresholds": thresholds,
        "CoverageR": coverage_recall,
        "CoverageP": coverage_precision,
        "MatchingR": amr_recall,
        "MatchingP": amr_precision,
    }

    return results, rmsd_results


evaluate_covmat_batched = evaluate_covmat

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
