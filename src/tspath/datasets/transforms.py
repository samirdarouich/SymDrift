import torch
from torch_geometric.transforms import BaseTransform
from tspath.alignment import kabsch_batched_scatter

__all__ = ["RandomRotate", "RandomPermute"]

class RemoveCOMReaction(BaseTransform):
    def forward(self, data):
        for pos_key in ['pos_ts', 'pos_r', 'pos_p']:
            pos = data[pos_key]
            com = pos.mean(dim=0, keepdim=True)
            data[pos_key] = pos - com
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
    
class RemoveCOM(BaseTransform):
    def forward(self, data):
        for pos_key in ['pos']:
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
        R = torch.tensor([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ])

        data.pos = data.pos @ R.T
        return data

class RandomPermute(BaseTransform):
    def forward(self, data):
        perm = torch.randperm(data.num_nodes)
        data.pos = data.pos[perm]
        data.x = data.x[perm]
        return data