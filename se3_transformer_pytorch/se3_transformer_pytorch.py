from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn, einsum

from itertools import product
from collections import namedtuple, defaultdict

from einops import rearrange, repeat
from se3_transformer_pytorch.basis import get_basis
from se3_transformer_pytorch.utils import exists, default

# helpers

def batched_index_select(values, indices):
    b, n, d, m, j = *values.shape, indices.shape[2]
    values = values[:, :, None, :, :].expand(-1, -1, j, -1, -1)
    return values.gather(1, indices[:, :, :, None, None].expand(-1, -1, -1, d, m))

def masked_mean(tensor, mask, dim = -1):
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)
    return tensor.sum(dim = dim) / mask.sum(dim = dim)

# fiber helpers

FiberEl = namedtuple('FiberEl', ['degrees', 'dim'])

class Fiber(nn.Module):
    def __init__(
        self,
        structure
    ):
        super().__init__()
        if isinstance(structure, dict):
            structure = structure.items()
        self.structure = structure

    @property
    def degrees(self):
        return map(lambda t: t[0], self.structure)

    @staticmethod
    def create(num_degrees, dim):
        return Fiber([FiberEl(degree, dim) for degree in range(num_degrees)])

    def __iter__(self):
        return iter(self.structure)

    def __mul__(self, fiber):
        return product(self.structure, fiber.structure)

# classes

class NormSE3(nn.Module):
    """Norm-based SE(3)-equivariant nonlinearity.
    
    Nonlinearities are important in SE(3) equivariant GCNs. They are also quite 
    expensive to compute, so it is convenient for them to share resources with
    other layers, such as normalization. The general workflow is as follows:

    > for feature type in features:
    >    norm, phase <- feature
    >    output = fnc(norm) * phase
    
    where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.
    """
    def __init__(
        self,
        fiber,
        nonlin = nn.ReLU(inplace=True),
        eps = 1e-12
    ):
        """Initializer.

        Args:
            fiber: Fiber() of feature multiplicities and types
            nonlin: nonlinearity to use everywhere
            num_layers: non-negative number of linear layers in fnc
        """
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.eps = eps

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for degree, chan in fiber:
            self.transform[str(degree)] = nn.Sequential(nn.LayerNorm(chan), nonlin)

    def forward(self, features):
        output = {}
        for degree, t in features.items():
            # Compute the norms and normalized features
            norm = t.norm(dim = -1, keepdim = True).clamp(min = self.eps)
            phase = t / norm

            # Transform on norms
            fn = self.transform[degree]
            transformed = fn(norm.squeeze(-1))[..., None]

            # Nonlinearity on norm
            output[degree] = (transformed * phase).view(*t.shape)

        return output

class ConvSE3(nn.Module):
    """A tensor field network layer as a DGL module.
    
    GConvSE3 stands for a Graph Convolution SE(3)-equivariant layer. It is the 
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.

    At each node, the activations are split into different "feature types",
    indexed by the SE(3) representation type: non-negative integers 0, 1, 2, ..
    """
    def __init__(
        self,
        fiber_in,
        fiber_out,
        self_interaction = True,
        edge_dim = 0
    ):
        """SE(3)-equivariant Graph Conv Layer

        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
            self_interaction: include self-interaction in convolution
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction

        # Neighbor -> center weights
        self.kernel_unary = nn.ModuleDict()

        for (di, mi), (do, mo) in (self.fiber_in * self.fiber_out):
            self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo)

        # Center -> center weights
        self.kernel_self = nn.ParameterDict()

        if self_interaction:
            structure = filter(lambda t: t[0] in self.fiber_out.degrees, self.fiber_in.structure)

            for d_in, m_in in structure:
                m_out = dict(self.fiber_out.structure)[d_in]
                weight = nn.Parameter(torch.randn(1, m_out, m_in) / sqrt(m_in))
                self.kernel_self[f'{d_in}'] = weight

    def forward(self, inp, edge_info, rel_dist = None, basis = None):
        """Forward pass of the linear layer

        Args:
            inp:    dict of features
            r:      inter-atomic distances
            basis:  pre-computed Q * Y
        Returns: 
            tensor with new features [B, n_points, n_features_out]
        """
        neighbor_indices, neighbor_masks = edge_info
        rel_dist = rearrange(rel_dist, 'b m n -> b m n ()')

        kernels = {}
        outputs = {}

        for (di, mi), (do, mo) in (self.fiber_in * self.fiber_out):
            etype = f'({di},{do})'
            kernel_fn = self.kernel_unary[etype]
            kernels[etype] = kernel_fn(rel_dist, basis = basis)
        
        for degree_out in self.fiber_out.degrees:
            output = 0
            degree_out_key = str(degree_out)

            for degree_in, m_in in self.fiber_in:
                x = inp[str(degree_in)]
                x = batched_index_select(x, neighbor_indices)
                x = x.view(*x.shape[:3], (2 * degree_in + 1) * m_in, 1)

                etype = f'({degree_in},{degree_out})'
                kernel = kernels[etype]
                output = output + torch.matmul(kernel, x)

            output = masked_mean(output, neighbor_masks, dim = 2)
            output = output.view(*x.shape[:2], -1, 2 * degree_out + 1)

            if self.self_interaction and degree_out_key in self.kernel_self.keys():
                x = inp[degree_out_key]
                kernel = self.kernel_self[degree_out_key]
                output = output + torch.matmul(kernel, x)

            outputs[degree_out_key] = output

        return outputs

class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""
    def __init__(
        self,
        num_freq,
        in_dim,
        out_dim,
        edge_dim = 0,
        mid_dim = 128
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
        return kernel.view(*kernel.shape[:3], self.d_out * self.nc_out, -1)

# main class

class SE3Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_neighbors = 12,
        heads = 8,
        dim_head = 64,
        depth = 6,
        num_degrees = 2,
        input_degrees = 1,
        output_degrees = 2
    ):
        super().__init__()
        assert num_neighbors > 0, 'neighbors must be at least 1'
        self.dim = dim

        self.num_degrees = num_degrees
        self.num_neighbors = num_neighbors

        fiber_in     = Fiber.create(input_degrees, dim)
        fiber_hidden = Fiber.create(num_degrees, dim)
        fiber_out    = Fiber.create(output_degrees, dim)

        self.conv_in  = ConvSE3(fiber_in, fiber_hidden)
        self.norm     = NormSE3(fiber_hidden)
        self.conv_out = ConvSE3(fiber_hidden, fiber_out)

    def forward(self, feats, coors, mask = None, return_type = None):
        if torch.is_tensor(feats):
            feats = {'0': feats[..., None]}

        b, n, d, *_, device = *feats['0'].shape, feats['0'].device
        assert d == self.dim, f'feature dimension {d} must be equal to dimension given at init {self.dim}'

        num_degrees, neighbors = self.num_degrees, self.num_neighbors

        rel_pos  = rearrange(coors, 'b n d -> b n () d') - rearrange(coors, 'b n d -> b () n d')
        rel_dist = rel_pos.norm(dim = -1)

        # get neighbors and neighbor mask, excluding self
        
        mask_value = torch.finfo(rel_dist.dtype).max
        self_mask = torch.eye(n, device = device).bool()
        masked_rel_dist = rel_dist.masked_fill_(self_mask, mask_value)
        neighbor_rel_dist, neighbor_indices = masked_rel_dist.topk(neighbors, dim = -1, largest = False)

        neighbor_rel_pos = rel_pos.gather(2, neighbor_indices[..., None].expand(-1, -1, -1, 3))
        basis = get_basis(neighbor_rel_pos, num_degrees - 1)

        neighbor_mask = None
        if exists(mask):
            neighbor_mask = mask[:, :, None].expand(-1, -1, neighbors).gather(1, neighbor_indices)

        # main logic
        edge_info = (neighbor_indices, neighbor_mask)
        x = feats

        x = self.conv_in(x, edge_info, rel_dist = neighbor_rel_dist, basis = basis)
        x = self.norm(x)
        x = self.conv_out(x, edge_info, rel_dist = neighbor_rel_dist, basis = basis)

        if exists(return_type):
            return x[str(return_type)]

        return x
