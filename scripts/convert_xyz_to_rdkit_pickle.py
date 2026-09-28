"""Convert an (extended) xyz file with conformers into the GEOM-style rdkit pickle
format used by the conformer datasets.

Each structure in the xyz file needs a SMILES string in its info dict (key given by
--smiles_key). The atom order of each structure must match the atom order of the
SMILES with explicit hydrogens (as obtained by
``datamol.to_mol(smiles, remove_hs=False, ordered=True)``). All structures sharing
the same SMILES are grouped as conformers of one molecule.

Outputs (in <data_root>/<name>/raw):
    - one ``<smiles>.pickle`` file per molecule
    - ``split_<name>_random.npz`` with a random train/val/test split

Example:
    python scripts/convert_xyz_to_rdkit_pickle.py --xyz my_conformers.xyz --name my_dataset
"""

import argparse
import os
import pickle

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read
from tqdm import tqdm

from symdrift.analysis.utils import get_mol_with_conformer

# Conversion factors to kcal/mol
ENERGY_UNITS = {"eV": 23.0605, "kcal/mol": 1.0, "hartree": 627.509}


def get_boltzmann_weights(energies, temperature=300, unit="eV"):
    k_B = 0.0019872041  # Boltzmann constant in kcal/(mol*K)
    beta = 1 / (k_B * temperature)
    energies_kcal = np.asarray(energies) * ENERGY_UNITS[unit]
    weights = np.exp(-beta * (energies_kcal - energies_kcal.min()))
    return (weights / weights.sum()).tolist()


def get_energy(atoms, energy_key):
    # Energies in the comment line of an extxyz file are attached as calculator
    if atoms.calc is not None:
        try:
            return atoms.get_potential_energy()
        except PropertyNotImplementedError:
            pass
    return atoms.info.get(energy_key, None)


def main(args):
    out_folder = os.path.join(args.data_root, args.name, "raw")
    os.makedirs(out_folder, exist_ok=True)

    # Group conformers by SMILES
    dataset = read(args.xyz, ":")
    dataset_ase = {}
    for conformer in dataset:
        smiles = conformer.info[args.smiles_key]
        dataset_ase.setdefault(smiles, []).append(conformer)
    print(f"Read {len(dataset)} conformers of {len(dataset_ase)} molecules")

    smiles_files = []
    for smiles, conformers in tqdm(dataset_ase.items(), desc="Writing pickles"):
        energies = [get_energy(c, args.energy_key) for c in conformers]

        # Use uniform weights if no energies are provided
        if any(e is None for e in energies):
            energies = [0.0] * len(conformers)
        weights = get_boltzmann_weights(energies, args.temperature, args.energy_unit)

        conformer_data = {"smiles": smiles, "conformers": []}
        for energy, weight, conformer in zip(energies, weights, conformers):
            conformer_data["conformers"].append(
                {
                    "totalenergy": energy,
                    "boltzmannweight": weight,
                    "rd_mol": get_mol_with_conformer(smiles, conformer.positions),
                }
            )

        # The file name (without extension) is used as identifier in the split file
        smiles_file = smiles.replace("/", "_")
        with open(os.path.join(out_folder, f"{smiles_file}.pickle"), "wb") as f:
            pickle.dump(conformer_data, f)
        smiles_files.append(smiles_file)

    # Create random train/val/test split
    frac_train, frac_val, _ = args.split
    rng = np.random.RandomState(args.seed)
    shuffled = smiles_files.copy()
    rng.shuffle(shuffled)
    N = len(shuffled)
    n_train = int(frac_train * N)
    n_val = int(frac_val * N)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    print(f"Train: {len(train)} molecules")
    print(f"Val: {len(val)} molecules")
    print(f"Test: {len(test)} molecules")
    split_path = os.path.join(out_folder, f"split_{args.name}_random.npz")
    np.savez(split_path, train=train, val=val, test=test)
    print(f"Saved pickles and split to <{out_folder}>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an xyz file with conformers into rdkit pickles and a random split."
    )
    parser.add_argument(
        "--xyz", type=str, required=True, help="Path to the (extended) xyz file"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name; outputs are written to <data_root>/<name>/raw",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root folder containing the dataset folders (default: data)",
    )
    parser.add_argument(
        "--smiles_key",
        type=str,
        default="smiles",
        help="Key of the SMILES string in the xyz info (default: smiles)",
    )
    parser.add_argument(
        "--energy_key",
        type=str,
        default="energy",
        help="Key of the energy in the xyz info, if not attached as calculator. "
        "Without energies, uniform Boltzmann weights are used (default: energy)",
    )
    parser.add_argument(
        "--energy_unit",
        type=str,
        default="eV",
        choices=list(ENERGY_UNITS),
        help="Unit of the energies (default: eV)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300,
        help="Temperature in K for the Boltzmann weights (default: 300)",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test fractions (default: 0.8 0.1 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random split (default: 42)"
    )
    main(parser.parse_args())
