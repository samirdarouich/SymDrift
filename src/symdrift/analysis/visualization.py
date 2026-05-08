import math
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
import torch
from ase import Atoms
from ase.io import read
from sklearn.decomposition import PCA

from symdrift.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = [
    "visualize_atoms_list",
    "visualize_reaction",
    "pca_plot",
]

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


def pca_plot(refs, samples, embedder, identifier="smiles", save_path=None):
    # Check if there are enough reference samples to perform PCA
    if len(refs) < 2:
        logger.debug(
            f"Not enough reference samples ({len(refs)}) to perform PCA plot, skipping..."
        )
        return

    # Check if there are common identifier values between refs and samples
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
    batch_ref = torch.cat(
        [torch.ones(len(atom)) * i for i, atom in enumerate(refs)]
    ).long()
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
    batch_samples = torch.cat(
        [torch.ones(len(atom)) * i for i, atom in enumerate(samples)]
    ).long()
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
            ref_mask_i = torch.tensor(
                [
                    i
                    for i, atom in enumerate(refs)
                    if atom.info[identifier] == identifier_value
                ]
            )
            ref_emb_mask_i = torch.isin(ref_mask, ref_mask_i)
            ref_emb_i = ref_emb[ref_emb_mask_i]

            if len(ref_mask_i) == 1:
                logger.debug(
                    f"Only one reference sample, {identifier}={identifier_value}, skipping..."
                )
                unused_axes.append(i)
                continue

            # Get samples embedding
            samples_mask_i = torch.tensor(
                [
                    i
                    for i, atom in enumerate(samples)
                    if atom.info[identifier] == identifier_value
                ]
            )
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
