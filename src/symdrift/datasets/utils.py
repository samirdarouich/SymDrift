# allowable multiple choice node and edge features
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from typing import Optional

import numpy as np
import pynauty
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, Conformer
from rdkit.Geometry import Point3D
from sympy.combinatorics import Permutation, PermutationGroup
from torch_geometric.data import Data

__all__ = [
    "chirality",
    "allowable_features",
    "get_atomic_number_and_charge",
    "GetNumRings",
    "safe_index",
    "atom_to_feature_vector",
    "bond_to_feature_vector",
    "compute_edge_index",
    "get_neighbor_ids",
    "get_chiral_tensors",
    "get_automorphisms",
    "compute_orbit_id_matrix_automorphism",
    "compute_orbit_id_matrix_atom_type",
    "build_conformer",
    "load_pkl",
    "check_disconnected_components",
    "ConformerData",
]

# similar to GeoMol
chirality = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1.0,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1.0,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,
}

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "misc",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_implicit_valence": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def get_atomic_number_and_charge(mol: Chem.Mol):
    """Returns atoms number and charge for rdkit molecule"""
    return np.array(
        [[atom.GetAtomicNum(), atom.GetFormalCharge()] for atom in mol.GetAtoms()]
    )


def GetNumRings(atom):
    return sum([atom.IsInRingSize(i) for i in range(3, 7)])


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except Exception as e:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """Node Invariant Features for an Atom."""
    atom_feature = [
        safe_index(
            allowable_features["possible_chirality_list"],
            chirality[atom.GetChiralTag()],
        ),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(
            allowable_features["possible_implicit_valence"], atom.GetImplicitValence()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
        GetNumRings(atom),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    # in case of no bond, assign "misc" category
    bond_type = str(bond.GetBondType()) if bond is not None else "misc"
    bond_feature = [
        safe_index(allowable_features["possible_bond_type_list"], bond_type),
        # allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        # allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def compute_edge_index(
    mol,
    no_reverse: bool = False,
    with_edge_attr=False,
    edge_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes edge index from mol object"""
    if edge_index is not None:
        bond_types = []
        for i, j in zip(*edge_index):
            b = mol.GetBondBetweenAtoms(int(i), int(j))
            bond_types.append(bond_to_feature_vector(b))
    else:
        edge_list = []
        bond_types = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list.append((i, j))
            bond_types.append(bond_to_feature_vector(bond))

            if not no_reverse:
                edge_list.append((j, i))
                bond_types.append(bond_to_feature_vector(bond))

        if len(edge_list) == 0:
            return torch.empty((2, 0)).long()

        edge_index = torch.from_numpy(np.array(edge_list).T).long()

    if with_edge_attr:
        edge_attr = torch.tensor(bond_types, dtype=torch.float32)  # (num_edges, 1)
        return edge_index, edge_attr

    return edge_index, None


def get_neighbor_ids(data):
    """
    Takes the edge indices and returns dictionary mapping atom index to neighbor indices
    Note: this only includes atoms with degree > 1
    """
    batch_nbrs = deepcopy(data.neighbors)
    batch_nbrs = [obj[0] for obj in batch_nbrs]
    neighbors = batch_nbrs.pop(0)  # get first element
    n_atoms_per_mol = data.batch.bincount()  # get atom count per graph
    n_atoms_prev_mol = 0

    for i, n_dict in enumerate(batch_nbrs):
        new_dict = {}
        n_atoms_prev_mol += n_atoms_per_mol[i].item()
        for k, v in n_dict.items():
            new_dict[k + n_atoms_prev_mol] = v + n_atoms_prev_mol
        neighbors.update(new_dict)

    return neighbors


def get_chiral_tensors(mol):
    """Only consider chiral atoms with 4 neighbors"""
    chiral_index = torch.tensor(
        [
            i
            for i, atom in enumerate(mol.GetAtoms())
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.int32,
    ).view(1, -1)  # (1, n_chiral_centers)
    # (n_chiral_centers, 4)
    chiral_nbr_index = torch.tensor(
        [
            [n.GetIdx() for n in atom.GetNeighbors()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.int32,
    ).view(1, -1)  # (1, n_chiral_centers * 4)
    # (n_chiral_centers,)
    chiral_tag = torch.tensor(
        [
            chirality[atom.GetChiralTag()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.float32,
    )

    return chiral_index, chiral_nbr_index, chiral_tag

def get_mol_properties(mol):
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    atomic_numbers = np.array(get_atomic_numbers_from_mol(mol))
    bonds = np.array(
        [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    )
    return num_atoms, num_bonds, atomic_numbers, bonds

def get_graph_from_mol(mol, use_atom_features=True):
    # Get atom types and edges from the molecule
    _, _, x, graph_edges = get_mol_properties(mol)
    
    # Using node features makes the graph even more unique. So only nodes that are 
    # truely interchangeable are considered the same.
    if use_atom_features:
        atomic_features = np.array(
            [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
        )
    
    num_nodes = x.shape[0]
    graph = pynauty.Graph(number_of_vertices=num_nodes, directed=False)
    for edge in graph_edges:
        graph.connect_vertex(edge[0].item(), [edge[1].item()])

    color_groups = defaultdict(set)
    for node in range(num_nodes):
        color = [x[node].item()]
        if use_atom_features:
            color += atomic_features[node].tolist()
        color_groups[tuple(color)].add(node)
    vertex_colors = list(color_groups.values())
    graph.set_vertex_coloring(vertex_colors)
    return x, graph_edges, graph

def get_automorphisms(mol, ignore_hs=False, max_no_perm=None, use_atom_features=True):
    """Returns a list of permutations corresponding to the automorphisms of the molecule"""

    # Get the graph representation of the molecule
    x, _, graph = get_graph_from_mol(mol, use_atom_features=use_atom_features)

    # Get the automorphism group of the graph
    generators, _, _, _, _ = pynauty.autgrp(graph)
    generators_sympy = [Permutation(g) for g in generators]

    # Construct the permutation group
    aut_group = PermutationGroup(generators_sympy)

    # Print all isomorphisms (automorphisms)
    num_nodes = x.shape[0]
    all_perms = [np.arange(num_nodes)]

    if ignore_hs:
        mask = x != 1
    else:
        mask = np.ones_like(x).astype(bool)

    for perm in aut_group.generate():
        if len(perm.array_form) == 0:
            continue
        perm_array = np.array(perm.array_form)
        moved = np.where(perm_array != np.arange(len(perm_array)))[0]
        signature = moved[mask[moved]]
        if len(signature) == 0:
            continue
        all_perms.append(perm.array_form)
        if max_no_perm is not None and len(all_perms) >= max_no_perm:
            break

    return torch.stack([torch.tensor(perm, dtype=torch.long) for perm in all_perms])

def compute_orbit_id_matrix_automorphism(
    automorphisms: torch.Tensor, chunk_size: int = 512
) -> torch.Tensor:
    """
    Compute a canonical orbit-ID matrix from a set of graph automorphisms.

    Each entry [i, j] holds the canonical hash of the edge orbit containing pair
    (i, j): the minimum over all automorphisms π of hash(min(π(i),π(j)), max(π(i),π(j))).
    Two edges are in the same orbit if they share the same hash value.

    Permutations are processed in chunks to keep peak memory at
    O(chunk_size x n_pairs) rather than O(n_perms x n_pairs).

    Args:
        automorphisms: [n_perms, n_atoms] long tensor of permutation arrays.
        chunk_size:    number of permutations to process at once.

    Returns:
        Symmetric [n_atoms, n_atoms] long tensor of orbit IDs.
    """
    n_perms, n_atoms = automorphisms.shape
    rows, cols = torch.triu_indices(n_atoms, n_atoms, offset=1)

    # Initialise with a value larger than any valid hash (max hash = (N-1)*N + (N-1))
    orbit_ids = torch.full((rows.shape[0],), n_atoms * n_atoms, dtype=torch.long)

    for start in range(0, n_perms, chunk_size):
        G_chunk = automorphisms[start : start + chunk_size]  # [chunk, n_atoms]
        pi_rows = G_chunk[:, rows]                           # [chunk, n_pairs]
        pi_cols = G_chunk[:, cols]                           # [chunk, n_pairs]
        p_min = torch.minimum(pi_rows, pi_cols)
        p_max = torch.maximum(pi_rows, pi_cols)
        pair_hash = p_min * n_atoms + p_max                  # [chunk, n_pairs]
        orbit_ids = torch.minimum(orbit_ids, pair_hash.min(dim=0).values)

    orbit_id_matrix = torch.zeros(n_atoms, n_atoms, dtype=torch.long)
    orbit_id_matrix[rows, cols] = orbit_ids
    orbit_id_matrix[cols, rows] = orbit_ids
    return orbit_id_matrix

def compute_orbit_id_matrix_atom_type(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Compute a canonical orbit-ID matrix based on atom types.
    Each entry [i, j] holds the canonical hash of the edge orbit containing pair
    (i, j): hash(min(Zi, Zj), max(Zi, Zj)) where Zi is the atomic number of atom i.
    Two edges are in the same orbit if they share the same hash value.
    
    Args:
        atomic_numbers: [n_atoms] long tensor of atomic numbers.
    
    Returns:
        Symmetric [n_atoms, n_atoms] long tensor of orbit IDs.
    """
    n_atoms = len(atomic_numbers)
    triu_r, triu_c = torch.triu_indices(n_atoms, n_atoms, offset=1)

    Zi, Zj = atomic_numbers[triu_r], atomic_numbers[triu_c]
    Zmax_val = atomic_numbers.max() + 1
    pair_hash = torch.minimum(Zi, Zj) * Zmax_val + torch.maximum(Zi, Zj)

    orbit_id_matrix = torch.zeros(n_atoms, n_atoms, dtype=torch.long)
    orbit_id_matrix[triu_r, triu_c] = pair_hash
    orbit_id_matrix[triu_c, triu_r] = pair_hash
    return orbit_id_matrix

def build_conformer(pos):
    if isinstance(pos, torch.Tensor) or isinstance(pos, np.ndarray):
        pos = pos.tolist()

    conformer = Conformer()

    for i, atom_pos in enumerate(pos):
        conformer.SetAtomPosition(i, Point3D(*atom_pos))

    return conformer


def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def check_disconnected_components(mol):
    """Check for disconnected components using Union-Find algorithm."""
    # Initialize parent array for union-find
    n_nodes = mol.GetNumAtoms()
    parent = list(range(n_nodes))

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Add reverse edge for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Process all edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        union(src, dst)

    # Count unique components
    components = {}
    for node in range(n_nodes):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)

    return list(components.values())


def get_atomic_numbers_from_mol(mol):
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return np.array(atomic_numbers, dtype=np.int8)


def check_smiles(smiles, smiles_mol):
    # filter mols rdkit can't intrinsically handle
    if smiles_mol is None:
        return False

    # skip conformers with fragments
    if "." in smiles:
        return False

    # skip mols with atoms with more than 4 neighbors for now
    num_neighbors = [len(a.GetNeighbors()) for a in smiles_mol.GetAtoms()]
    if np.max(num_neighbors) > 4:
        return False

    return True


def check_conformer(conformer_mol, num_atoms, num_bonds, atomic_numbers, bonds):
    edge_case = False

    # mark mols with fragments as edge cases
    if len(Chem.GetMolFrags(conformer_mol)) > 1:
        edge_case = True

    # skip mols with atoms with more than 4 neighbors for now
    # this is in line with the MCF preprocessing
    # here: https://github.com/apple/ml-mcf/blob/main/process_data.py
    num_neighbors = [len(a.GetNeighbors()) for a in conformer_mol.GetAtoms()]
    if np.max(num_neighbors) > 4:
        return "invalid"

    # mark mols that appear to be not a real conformer as edge cases
    # this ensures the same connectivity structure across all mols
    other_num_atoms, other_num_bonds, other_atomic_numbers, other_bonds = (
        get_mol_properties(conformer_mol)
    )
    if num_atoms != other_num_atoms:
        edge_case = True

    if num_bonds != other_num_bonds:
        edge_case = True

    if not edge_case:
        # being here means number of atoms and bonds are matching
        target = np.zeros(num_atoms)
        if not np.allclose(atomic_numbers - other_atomic_numbers, target):
            edge_case = True

        target = np.zeros((num_bonds, 2))
        if not np.allclose(bonds - other_bonds, target):
            edge_case = True

    if edge_case:
        return "edge_case"
    return "valid"


# taken from: https://github.com/ML4MolSim/dit_mc/blob/4b1615c523a36fa107bc6c86ddeee5544b78c2a6/tf_datasets/geom/preprocessing.py
def filter_mols(mol_dict):
    confs = mol_dict["conformers"]
    smiles = mol_dict["smiles"]

    mol = Chem.MolFromSmiles(smiles)

    # filter smiles mol
    if not check_smiles(smiles, mol):
        return []

    # we load the first conformer to get the properties
    # maybe we should rather load it directly from the smiles mol?
    mol = confs[0]["rd_mol"]
    num_atoms, num_bonds, atomic_numbers, bonds = get_mol_properties(mol)

    mols = []
    for conf in confs:
        mol = conf["rd_mol"]
        result = check_conformer(mol, num_atoms, num_bonds, atomic_numbers, bonds)
        edge_case = False

        if result == "invalid":
            continue
        elif result == "edge_case":
            edge_case = True

        mols.append(
            {
                "rd_mol": mol,
                "edge_case": edge_case,
                "boltzmannweight": conf["boltzmannweight"],
                "totalenergy": conf["totalenergy"],
            }
        )
    return mols


class ConformerData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "conformer_index":
            # Instead of adding n_atoms, we add the number of conformers
            # present in the current data object.
            return self.num_conformers
        return super().__inc__(key, value, *args, **kwargs)
