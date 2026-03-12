import torch
from torch_geometric.transforms import BaseTransform
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tspath.alignment import kabsch_batched_scatter

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Constants
BOND_TYPES = {
    BT.SINGLE: 1,
    BT.DOUBLE: 2,
    BT.TRIPLE: 3,
    BT.AROMATIC: 4,
}

__all__ = [
    "RemoveCOMReaction",
    "RemoveCOM",
    "RandomRotate",
    "RandomPermute",
    "AlignReaction",
]


class RemoveCOMReaction(BaseTransform):
    def forward(self, data):
        for pos_key in ["pos_ts", "pos_r", "pos_p"]:
            pos = data[pos_key]
            com = pos.mean(dim=0, keepdim=True)
            data[pos_key] = pos - com
        return data


class RemoveCOM(BaseTransform):
    def forward(self, data):
        for pos_key in ["pos"]:
            pos = data[pos_key]
            com = pos.mean(dim=0, keepdim=True)
            data[pos_key] = pos - com
        return data


class RandomRotate(BaseTransform):
    def forward(self, data):

        # Sample random unit quaternion
        q = torch.randn(4)
        q = q / q.norm()

        w, x, y, z = q

        # Convert quaternion to rotation matrix
        R = torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

        data.pos = data.pos @ R.T
        return data


class RandomPermute(BaseTransform):
    def forward(self, data):
        perm = torch.randperm(data.num_nodes)
        data.pos = data.pos[perm]
        data.x = data.x[perm]
        return data


class AlignReaction(BaseTransform):
    def forward(self, data):
        pos_r = data.pos_r
        pos_p = data.pos_p

        # align product to reactant
        pos_p_aligned = kabsch_batched_scatter(
            pos_r, pos_p, torch.zeros(pos_r.shape[0], dtype=torch.long)
        )

        data.pos_p = pos_p_aligned
        return data

class FeaturizeMolecule(BaseTransform):
    def forward(self, data):
        
        bonded_edge_index = [[],[]]
        edge_features = []

        if hasattr(data, "smiles"):
            smiles = data.smiles
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
                mol_x = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                if data.x.tolist()==mol_x: 
                    for bond in mol.GetBonds():
                        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        bonded_edge_index[0].extend([i, j])
                        bonded_edge_index[1].extend([j, i])
                        edge_features.extend([BOND_TYPES[bond.GetBondType()] if bond else 0]*2)

        bonded_edge_index = torch.tensor(bonded_edge_index, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.long)
        data.bonded_edge_index = bonded_edge_index
        data.edge_features = edge_features
        return data