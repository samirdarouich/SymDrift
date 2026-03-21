from ase.io import read
from tspath.analysis import get_best_rmsd_rdkit, pymatgen_match
import numpy as np

atoms = read(
    "/home/samirdarouich/projects/TS_physics/tspath/runs/conformer_test/dataset_geom_qm9/split_geomol_debug/torchmd/temp_0.05/norm_True/embedder_distance/all_drift/plots/step_final.xyz",
    ":",
)

rng = np.random.default_rng(42)

n_select = min(200, len(atoms))
selected_idx = rng.choice(len(atoms), size=n_select, replace=False)

selected_atoms = [atoms[i] for i in selected_idx]
target_mols = selected_atoms[: n_select // 2]
gen_mols = selected_atoms[n_select // 2 : n_select]

diffs = []
for i, (target, gen) in enumerate(zip(target_mols, gen_mols)):

    rmsd = get_best_rmsd_rdkit(target, gen)
    rmsd_pymatgen, _ = pymatgen_match(target, gen)
    
    gen = gen.copy()
    atomic_numbers = np.array(gen.get_atomic_numbers())
    positions = gen.get_positions().copy()

    for z in np.unique(atomic_numbers):
        idx = np.where(atomic_numbers == z)[0]
        positions[idx] = positions[rng.permutation(idx)]

    gen.set_positions(positions)

    # recompute metrics after permutation
    rmsd = get_best_rmsd_rdkit(target, gen)
    rmsd_pymatgen, _ = pymatgen_match(target, gen)
    
    diff = abs(rmsd - rmsd_pymatgen)
    diffs.append(diff)

print(f"Average RMSD difference: {np.mean(diffs):.4f} Å")