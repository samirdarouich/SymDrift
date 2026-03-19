# allowable multiple choice node and edge features
from copy import deepcopy

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import Conformer
from rdkit.Geometry import Point3D
from rdkit.Chem import rdmolops
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

# similar to GeoMol
BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
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
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        # allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        # allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def compute_edge_index(
    mol, no_reverse: bool = False, with_edge_attr=False, with_shortest_hops=False
) -> torch.Tensor:
    """Computes edge index from mol object"""
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

    shortest_hops = None
    if with_shortest_hops:
        num_atoms = mol.GetNumAtoms()
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        shortest_path_result, _ = algos.floyd_warshall(adj_matrix)
        mask = ~np.eye(num_atoms, dtype=np.bool_) # get rid of self interactions
        shortest_hops = torch.from_numpy(shortest_path_result[mask]).long()
        
    if with_edge_attr:
        edge_attr = torch.tensor(bond_types, dtype=torch.float32)  # (num_edges, 1)
        return edge_index, edge_attr, shortest_hops

    return edge_index, None, shortest_hops


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
    ).view(
        1, -1
    )  # (1, n_chiral_centers)
    # (n_chiral_centers, 4)
    chiral_nbr_index = torch.tensor(
        [
            [n.GetIdx() for n in atom.GetNeighbors()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.int32,
    ).view(
        1, -1
    )  # (1, n_chiral_centers * 4)
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


def build_conformer(pos):
    if isinstance(pos, torch.Tensor) or isinstance(pos, np.ndarray):
        pos = pos.tolist()

    conformer = Conformer()

    for i, atom_pos in enumerate(pos):
        conformer.SetAtomPosition(i, Point3D(*atom_pos))

    return conformer