import logging
import math

import numpy as np
from pymatgen.analysis.molecule_matcher import (
    BruteForceOrderMatcher,
    GeneticOrderMatcher,
    HungarianOrderMatcher,
    KabschMatcher,
)
from pymatgen.core import Molecule
from torch_scatter import scatter_mean

from symdrift.alignment import (
    brute_force_and_kabch_batched,
    hungarian_and_kabch_batched,
    kabsch_batched,
    kabsch_batched_scatter,
)
from symdrift.utils import RankedLogger

logging.getLogger("pymatgen.analysis.molecule_matcher").setLevel(logging.WARNING)

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = [
    "rmse_core",
    "pymatgen_rmse",
    "pymatgen_match",
    "get_rmsd_batched_scatter",
    "get_rmsd_batched",
]


def rmse_core(mol1, mol2, same_order=False, threshold=0.5, max_permutations=1e4):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        aligned, rmse = bfm.fit(mol2)
        return rmse, aligned
    total_permutations = 1
    for c in count:
        total_permutations *= math.factorial(c)  # type: ignore
    if total_permutations < max_permutations:
        logger.debug(
            f"Using brute force matcher with {total_permutations} permutations"
        )
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
            logger.debug("Using Hungarian algorithm matcher.")
            bfm = HungarianOrderMatcher(mol1)
            aligned, rmse = bfm.fit(mol2)
        else:
            logger.debug("Using Genetic algorithm matcher.")
    return rmse, aligned


def pymatgen_rmse(
    mol1,
    mol2,
    ignore_chirality: bool = False,
    same_order: bool = False,
    threshold: float = 0.5,
    max_permutations: int = 1e4,
):
    rmse, aligned = rmse_core(
        mol1,
        mol2,
        same_order=same_order,
        threshold=threshold,
        max_permutations=max_permutations,
    )
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(species=mol2.species, coords=coords)
        rmse_reflect, aligned_reflect = rmse_core(
            mol1,
            mol2_reflect,
            same_order=same_order,
            threshold=threshold,
            max_permutations=max_permutations,
        )
        if rmse_reflect < rmse:
            rmse = rmse_reflect
            aligned = aligned_reflect
    return rmse, aligned


def pymatgen_match(
    ref,
    sample,
    ignore_chirality=False,
    same_order=False,
    threshold=0.5,
    max_permutations=1e4,
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
        same_order=same_order,
        threshold=threshold,
        max_permutations=max_permutations,
    )

    # pymatgen computes rmse instead of rmsd
    rmsd = rmse * 3**0.5

    # sample is aligned and permuted to match ref
    aligned_sample = sample.copy()
    aligned_sample.numbers = ref.numbers
    aligned_sample.positions = aligned.cart_coords
    return rmsd, aligned_sample


def get_rmsd_batched_scatter(xi, xj, batch, align=False):
    if align:
        xj = kabsch_batched_scatter(xi, xj, batch)
    diff = (xi - xj) ** 2
    rmsd = scatter_mean(diff.sum(-1), batch, dim=0).sqrt()
    return rmsd


def get_rmsd_batched(
    x,
    y,
    atomic_numbers=None,
    align=False,
    permute=False,
    brute_force_permutations=False,
):
    """
    Compute RMSD between two batches of structures x and y, where x and y are of shape
    (B, N, d). The RMSD is computed for each pair of structures in the batch.
    RMSD(x,y) = sqrt(1/N * sum((x-y)^2))
    """
    assert x.shape == y.shape, "X and Y must have same shape"
    if align:
        if permute:
            if brute_force_permutations:
                y, _ = brute_force_and_kabch_batched(x, y, atomic_numbers)
            else:
                y, _ = hungarian_and_kabch_batched(x, y, atomic_numbers)
        else:
            y, _ = kabsch_batched(x, y)
    B, n_atoms, d = x.shape
    rmsd = (((x - y) ** 2).sum(dim=(-2, -1)) / n_atoms).sqrt()
    return rmsd
