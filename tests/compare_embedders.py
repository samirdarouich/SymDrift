from ase.io import read
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import radius_graph
from typing import Optional

class DistanceEmbedder(nn.Module):
    def __init__(self, r_max: Optional[float]=None, invariant=False):
        super().__init__()
        self.invariant = invariant
        if r_max is None:
            r_max = 1e6
        self.r_max = r_max
        
    def forward(
        self, 
        positions: Tensor, 
        batch: Optional[Tensor] = None,
        Z: Optional[Tensor] = None,
        invariant: Optional[bool] = None,
        **kwargs
    ):
        """
        Embed the positions of atoms using a distance-based embedding.
        
        Args:
            positions (Tensor): 
                Tensor of shape (B*n_atoms, 3) containing the positions of the atoms.
            batch (Tensor, optional):
                Tensor of shape (B*n_atoms,) containing the batch indices for each atom.
            Z (Tensor, optional): 
                Tensor of shape (B*n_atoms) containing the atomic numbers of the atoms.
                Needed if `invariant` is True to compute the invariant embedding. 
            edge_index (Tensor, optional):
              Tensor of shape (2, n_edges) containing the indices of neighboring atoms.
            invariant (bool, optional): 
                If True, the embedding will be invariant to permutations of atoms. 
                If False, the embedding will be based on the full distance matrix. 
                If None, it will use the class attribute `self.invariant`.
        """
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
        if invariant is not None:
            # if desired overwrite invariant attribute with forward argument
            self.invariant = invariant
            
        # reshape positions to (B, n_atoms, 3) and compute pairwise distances
        B = batch.max().item() + 1
        n_atoms = positions.shape[0] // B
        pos = positions.view(B, n_atoms, 3)
        
        # Compute full distance matrix
        distance = torch.cdist(pos, pos)
        if self.invariant:
            z = Z.view(B, n_atoms)
            unique_types = torch.unique(z)
            d = []
            for Zi in unique_types:
                for Zj in unique_types:
                    mask_i = (z == Zi)[:, :, None]  # (B, N,1)
                    mask_j = (z == Zj)[:, None, :]  # (B, 1,N)
                    pair_mask = mask_i & mask_j     # (B, N, N)
                
                    d_ = distance[pair_mask].view(B, -1)
                    d_ = torch.sort(d_, dim=1)[0]
                    d.append(d_)
            d = torch.cat(d, dim=1)
            return d
        else:
            return distance.view(B, -1)
        

class DistanceEmbedderPyG(DistanceEmbedder):
    
    def forward(
        self, 
        positions: Tensor, 
        batch: Optional[Tensor] = None,
        Z: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        invariant: Optional[bool] = None,
        **kwargs
    ):
        """
        Embed the positions of atoms using a distance-based embedding.
        
        Args:
            positions (Tensor): 
                Tensor of shape (B*n_atoms, 3) containing the positions of the atoms.
            batch (Tensor, optional):
                Tensor of shape (B*n_atoms,) containing the batch indices for each atom.
            Z (Tensor, optional): 
                Tensor of shape (B*n_atoms) containing the atomic numbers of the atoms.
                Needed if `invariant` is True to compute the invariant embedding. 
            edge_index (Tensor, optional):
              Tensor of shape (2, n_edges) containing the indices of neighboring atoms.
            invariant (bool, optional): 
                If True, the embedding will be invariant to permutations of atoms. 
                If False, the embedding will be based on the full distance matrix. 
                If None, it will use the class attribute `self.invariant`.
        """
        if invariant is not None:
            # if desired overwrite invariant attribute with forward argument
            self.invariant = invariant
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
        if edge_index is None:
            row, col = radius_graph(positions, r=self.r_max, batch=batch, loop=True)
            # # mask out all symmetric entries (keep only one of (i,j) and (j,i))
            # mask = row < col
            # row, col = row[mask], col[mask]
        
        # compute distances for the edges (assuming fully connected graph, reconstructs 
        # the full distance matrix)
        dist = (positions[row] - positions[col]).norm(dim=-1)
        
        # get batch indices for the edges
        edge_batch = batch[row]
        
        # sort edges with the same pair_type
        Zi, Zj = Z[row], Z[col]

        # sorting key: (graph, pair_type, distance)
        Zmax_val = Z.max() + 1
        pair_type = Zi * Zmax_val + Zj
        max_dist = dist.max().detach() + 1.0
        group_id = edge_batch * (Zmax_val**2) + pair_type
        
        idx = torch.argsort(group_id)
        key = group_id[idx] * max_dist + dist[idx]
        
        # apply permutation
        perm = torch.argsort(key)
        dist_sorted = dist[idx][perm]
        edge_batch_sorted = edge_batch[idx][perm]
        assert (edge_batch_sorted[1:] >= edge_batch_sorted[:-1]).all()

        return dist_sorted, edge_batch_sorted


# This compares the two implementations
atoms = read("/home/samirdarouich/projects/TS_physics/tspath/data/geom_qm9/atoms/test.xyz","1:3")
positions = torch.stack([torch.tensor(atom.positions, dtype=torch.float) for atom in atoms]).view(-1, 3)
Z = torch.tensor([atom.numbers for atom in atoms], dtype=torch.long).view(-1)
batch = torch.cat([torch.full((len(atom),), i, dtype=torch.long) for i, atom in enumerate(atoms)]).view(-1)

d = DistanceEmbedder(invariant=True)(positions, batch=batch, Z=Z)

d_pyg, mask = DistanceEmbedderPyG(invariant=True)(positions, batch=batch, Z=Z)

assert torch.allclose(d_pyg.view(2,-1),d)


from tspath.model import DistanceEmbedder, GaussianMomentEmbedder
from tspath.analysis import pca_plot

refs = read("/home/samirdarouich/projects/TS_physics/tspath/data/geom_qm9/atoms/test.xyz",":50")
samples = read("/home/samirdarouich/projects/TS_physics/tspath/runs/sample_db.xyz", ":50")

pca_plot(refs, samples, DistanceEmbedder(invariant=True), identifier="smiles", save_path="pca_plot_distance.png")
pca_plot(refs, samples, GaussianMomentEmbedder(), identifier="smiles", save_path="pca_plot_gmm.png")





