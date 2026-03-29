
import torch
import torch.nn as nn
from einops import rearrange

class AtomCGREmbedding(nn.Module):
    def __init__(self, n_atom_rdkit_feats, last_channel):
        super().__init__()
        self.half_last_channel_dim = last_channel // 2
        self.atom_embedding = nn.Embedding(100, self.half_last_channel_dim)
        self.atom_feat_embedding = nn.Linear(n_atom_rdkit_feats, self.half_last_channel_dim, bias=False)

    def forward(self, z_N, r_feat_N_F, p_feat_N_F):
        a_emb = self.atom_embedding(z_N)
        af_emb_r = self.atom_feat_embedding(r_feat_N_F.float())
        af_emb_p = self.atom_feat_embedding(p_feat_N_F.float())
        z1 = a_emb + af_emb_r
        z2 = af_emb_p - af_emb_r
        # return torch.cat([z1, z2], dim=-1)
        return rearrange([z1, z2], 'a n d -> n (a d)', a=2, d=self.half_last_channel_dim)
    
class EdgeCGREmbedding(nn.Module):
    def __init__(self, hidden_dim=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = nn.Embedding(100, hidden_dim)
        self.edge_cat = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_type_r, edge_type_p):
        edge_attr_r = self.bond_emb(edge_type_r)
        edge_attr_p = self.bond_emb(edge_type_p)
        # return self.edge_cat(torch.cat([edge_attr_r, edge_attr_p], dim=-1))
        return self.edge_cat(rearrange([edge_attr_r, edge_attr_p], 'a e d -> e (a d)', a=2, d=self.hidden_dim))