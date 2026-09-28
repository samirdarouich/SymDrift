import argparse
import glob
import os

import numpy as np
import pandas as pd
from ase.io import read, write

from symdrift.datasets import (
    ConformerDatasetDisk,
    ConformerDatasetInMemory,
    ReactionDataset,
)

GEOM_DATASETS = {
    "geom_qm9": ConformerDatasetInMemory,
    "geom_drugs": ConformerDatasetDisk,
}


def process_geom(dataset_name, data_root):
    data_dir = os.path.join(data_root, dataset_name)

    # Process the raw GEOM dataset files
    dataset = GEOM_DATASETS[dataset_name](
        source=dataset_name,
        root=data_dir,
    )

    # Load the original split indices
    orig_split = np.load(f"{data_dir}/split0.npy", allow_pickle=True)
    all_files = sorted(glob.glob(os.path.join(data_dir, "raw", "*.pickle")))

    # Find all molecules that are still valid after processing
    split_files_cleaned = {}
    for j, split_name in enumerate(["train", "val", "test"]):
        split_files_cleaned[split_name] = []

        for i, f in enumerate(all_files):
            file_identifier = os.path.basename(f).split(".")[0]
            if i in orig_split[j] and file_identifier in dataset.file_identifier:
                split_files_cleaned[split_name].append(file_identifier)

    # Subselect the valid test samples from the 1000 intended test smiles
    test_smiles = pd.read_csv(f"{data_dir}/test_smiles.csv")

    cleaned_test = []
    for row in test_smiles.itertuples():
        smi = row.smiles
        if smi in split_files_cleaned["test"]:
            cleaned_test.append(smi)
    print(f"Number of cleaned test samples: {len(cleaned_test)}")
    split_files_cleaned["test"] = cleaned_test

    # Save split
    np.savez(
        f"{data_dir}/raw/split_{dataset_name}_geomol_cleaned.npz",
        **split_files_cleaned,
    )


def process_rdb7(data_root):
    data_dir = os.path.join(data_root, "rdb7")
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    rxn_data = pd.read_csv(f"{data_dir}/rdb7_full.csv")
    xyz = read(f"{data_dir}/rdb7_full.xyz", ":")  # Read all structures

    # Write reactant, transition state and product with reaction information
    out_path = f"{data_dir}/rdb7.xyz"
    if os.path.exists(out_path):
        os.remove(out_path)

    for idx, row in rxn_data.iterrows():
        rxn_id = row["rxn"]
        r_smiles, p_smiles = row["smiles"].split(">>")
        r, ts, p = xyz[3 * idx], xyz[3 * idx + 1], xyz[3 * idx + 2]
        r.info["rxn"] = rxn_id
        r.info["type"] = "reactant"
        r.info["smiles"] = r_smiles
        ts.info["rxn"] = rxn_id
        ts.info["type"] = "transition_state"
        p.info["rxn"] = rxn_id
        p.info["type"] = "product"
        p.info["smiles"] = p_smiles
        write(out_path, [r, ts, p], append=True)

    # split contains the reaction indices
    split = np.load(f"{data_dir}/split_rdb7_random.pkl", allow_pickle=True)
    print("Train size:", len(split["train"]))
    print("Val size:", len(split["val"]))
    print("Test size:", len(split["test"]))
    np.savez(
        f"{raw_dir}/split_rdb7_random.npz",
        train=split["train"],
        val=split["val"],
        test=split["test"],
    )

    # Process the raw RDB7 dataset file
    ReactionDataset(source="rdb7", root=data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process raw datasets and convert split files."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        nargs="+",
        choices=[*GEOM_DATASETS, "rdb7"],
        required=True,
        help="Dataset(s) to process",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root folder containing the dataset folders (default: data)",
    )
    args = parser.parse_args()

    for dataset_name in args.dataset:
        if dataset_name == "rdb7":
            process_rdb7(args.data_root)
        else:
            process_geom(dataset_name, args.data_root)
