import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

from se3_transformer_pytorch.basis import get_basis

class SE3Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        depth = 6,
        num_degrees = 4
    ):
        super().__init__()
        self.num_degrees = num_degrees

    def forward(self, feats, coors, mask = None):
        num_degrees = self.num_degrees

        rel_pos  = rearrange(coors, 'b n d -> b n () d') - rearrange(coors, 'b n d -> b () n d')
        rel_dist = rel_pos.norm(dim = -1, keepdim = True)
        basis    = get_basis(rel_pos, num_degrees - 1)

        return feats
