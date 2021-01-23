import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

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

    def forward(self, x):
        return x
