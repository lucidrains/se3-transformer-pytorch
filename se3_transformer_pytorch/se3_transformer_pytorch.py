import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

from se3_transformer_pytorch.basis import get_basis

# helpers

# classes

class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""
    def __init__(
        self,
        *,
        num_freq,
        in_dim,
        out_dim,
        edge_dim = 0,
        mid_dim = 32
    ):
        """NN parameterized radial profile function.

        Args:
            num_freq: number of output frequencies
            in_dim: multiplicity of input (num input channels)
            out_dim: multiplicity of output (num output channels)
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.net = nn.Sequential(
            nn.Linear(edge_dim + 1, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, num_freq * in_dim * out_dim)
        )

        self.apply(self.init_)

    def init_(self, m):
        if m in {nn.Linear}:
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        y = self.net(x)
        return rearrange(y, '... (o i f) -> ... o () i () f', i = self.in_dim, o = self.out_dim)

class PairwiseConv(nn.Module):
    """SE(3)-equivariant convolution between two single-type features"""
    def __init__(
        self,
        *,
        degree_in,
        nc_in,
        degree_out,
        nc_out,
        edge_dim = 0
    ):
        """SE(3)-equivariant convolution between a pair of feature types.

        This layer performs a convolution from nc_in features of type degree_in
        to nc_out features of type degree_out.

        Args:
            degree_in: degree of input fiber
            nc_in: number of channels on input
            degree_out: degree of out order
            nc_out: number of channels on output
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        self.num_freq = 2 * min(degree_in, degree_out) + 1
        self.d_out = 2 * degree_out + 1
        self.edge_dim = edge_dim

        self.rp = RadialFunc(self.num_freq, nc_in, nc_out, edge_dim)

    def forward(self, feat, basis):
        R = self.rp(feat)
        kernel = torch.sum(R * basis[f'{self.degree_in},{self.degree_out}'], dim = -1)
        return kernel.view(kernel.shape[0], self.d_out * self.nc_out, -1)

# main class

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
