import torch
from torch_geometric.utils import get_laplacian, scatter, to_dense_adj
from tspath.utils import batch_center_systems
import logging

logger = logging.getLogger(__name__)

class HarmonicSampler:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.eig_val_cache = {}
        self.eig_vec_cache = {}

    def diagonalize(self, n_nodes, edges, batch=None, smiles=None):
        a = self.alpha * torch.ones((edges.shape[0],), device=edges.device)
        # compute L = D - A, where D is the degree matrix and A is the adjacency matrix. L is the graph Laplacian.
        # edge_weight is is the sparse representation of the bond order between atoms,
        # -alpha for connected atoms, x for self-loops (i-i), where x counts the number
        # of bonds for atom i.
        edge_index, edge_weight = get_laplacian(
            edges.T,
            a,
            num_nodes=n_nodes,
        )

        # H is the connectivity matrix that encodes the bond structure of the molecule.
        # It has the same entries as the sparse edge weightm but 0 for non-connected atoms
        H = to_dense_adj(
            edge_index=edge_index, edge_attr=edge_weight, max_num_nodes=n_nodes
        ).squeeze()

        if batch is None:
            D, P = torch.linalg.eigh(H)
            return D, P

        Ds, Ps = [], []
        batch_size = batch.max() + 1

        for i in range(batch_size):
            idx = torch.where(batch == i)[0]
            start = idx.min()
            end = idx.max() + 1

            D, P = None, None
            if smiles is not None:
                D, P = self.check_cache(smiles[i])

                if (D is not None) and (P is not None):
                    D = D.to(edge_index.device)
                    P = P.to(edge_index.device)

            if (D is None) or (P is None):
                D, P = torch.linalg.eigh(H[start:end, start:end])

                if smiles is not None:
                    self.eig_val_cache[smiles[i]] = D.cpu()
                    self.eig_vec_cache[smiles[i]] = P.cpu()

            Ds.append(D)
            Ps.append(P)

        return torch.cat(Ds), torch.block_diag(*Ps)

    def check_cache(self, smiles):
        D = self.eig_val_cache.get(smiles, None)
        P = self.eig_vec_cache.get(smiles, None)
        return D, P

    def sample(self, size, edge_index, batch=None, smiles=None, epsilon=1e-5):
        # transpose if (2, n_edges)
        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        n_nodes = size[0]

        # D is the mode stiffness, thus how strong each mode is. P are the normal modes.
        # if D is 0 then this is only a rotation/translation mode.
        # The modes i,j describe how atom i moves when mode j is excited.
        D, P = self.diagonalize(n_nodes=n_nodes, edges=edge_index, batch=batch, smiles=smiles)

        # In a real system each molecule has 3 translational and 3 rotational zero-modes.
        # Thus leaving 3N-6 non-zero modes for N atoms. In the case of a harmonic prior
        # based on chemical bonds, there is only one zero mode, the global translation.
        # This is only true, if the whole system is connected. In case the system has
        # multiple disconnected components, each component has its own translation mode.
        # There are no rotational modes, since the bonds do not constrain rotations.
        # Instead of zeroing only the first mode, we zero all modes that are below a
        # threshold. This is necessary to account for multiple disconnected components.
        std = 1.0 / torch.sqrt(D)
        translational_modes_mask = D < epsilon
        std[translational_modes_mask] = 0.0

        expected_translational_modes = 1 if batch is None else (batch.max().item() + 1)
        if translational_modes_mask.sum().item() > expected_translational_modes:
            logger.warning("Graph has multiple disconnected graphs. Be sure it is intended.")

        noise = torch.randn(size).to(D.device)
        noise = std[:, None] * noise
        noise[noise.isnan()] = 0.0
        sample = P @ (noise)
        
        sample = batch_center_systems(sample, batch)

        return sample

    def energy(self, x, edge_index, epsilon=1e-5, batch=None, smiles=None):
        n_nodes = x.size(0)
        x = batch_center_systems(x)

        if batch is None:
            batch = torch.zeros(n_nodes).to(x.device).long()

        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        D, P = self.diagonalize(n_nodes, edges=edge_index, batch=batch, smiles=smiles)

        translational_modes_mask = D < epsilon
        
        energy_unpooled = D[:, None] * (P.T @ x) ** 2
        energy_unpooled[translational_modes_mask] = 0.0
        energy_unpooled = energy_unpooled.sum(-1)
        energy = 0.5 * scatter(energy_unpooled, batch)

        return energy.view(-1, 1)