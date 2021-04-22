from math import sqrt
from itertools import product
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, einsum

from se3_transformer_pytorch.basis import get_basis
from se3_transformer_pytorch.utils import exists, default, uniq, map_values, batched_index_select, masked_mean, to_order, fourier_encode, cast_tuple
from se3_transformer_pytorch.reversible import ReversibleSequence, SequentialSequence

from einops import rearrange, repeat

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
    def dims(self):
        return uniq(map(lambda t: t[1], self.structure))

    @property
    def degrees(self):
        return map(lambda t: t[0], self.structure)

    @staticmethod
    def create(num_degrees, dim):
        dim_tuple = dim if isinstance(dim, tuple) else ((dim,) * num_degrees)
        return Fiber([FiberEl(degree, dim) for degree, dim in zip(range(num_degrees), dim_tuple)])

    def __getitem__(self, degree):
        return dict(self.structure)[degree]

    def __iter__(self):
        return iter(self.structure)

    def __mul__(self, fiber):
        return product(self.structure, fiber.structure)

    def __and__(self, fiber):
        out = []
        degrees_out = fiber.degrees
        for degree, dim in self:
            if degree in fiber.degrees:
                dim_out = fiber[degree]
                out.append((degree, dim, dim_out))
        return out

def get_tensor_device_and_dtype(features):
    first_tensor = next(iter(features.items()))[1]
    return first_tensor.device, first_tensor.dtype

# classes

class ResidualSE3(nn.Module):
    """ only support instance where both Fibers are identical """
    def forward(self, x, res):
        out = {}
        for degree, tensor in x.items():
            degree = str(degree)
            out[degree] = tensor
            if degree in res:
                out[degree] = out[degree] + res[degree]
        return out

class LinearSE3(nn.Module):
    def __init__(
        self,
        fiber_in,
        fiber_out
    ):
        super().__init__()
        self.weights = nn.ParameterDict()

        for (degree, dim_in, dim_out) in (fiber_in & fiber_out):
            key = str(degree)
            self.weights[key]  = nn.Parameter(torch.randn(dim_in, dim_out) / sqrt(dim_in))

    def forward(self, x):
        out = {}
        for degree, weight in self.weights.items():
            out[degree] = einsum('b n d m, d e -> b n e m', x[degree], weight)
        return out

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
        nonlin = nn.GELU(),
        eps = 1e-12
    ):
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
    """A tensor field network layer
    
    ConvSE3 stands for a Convolution SE(3)-equivariant layer. It is the 
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
        pool = True,
        edge_dim = 0,
        fourier_encode_dist = False,
        num_fourier_features = 4,
        splits = 4
    ):
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction

        self.num_fourier_features = num_fourier_features
        self.fourier_encode_dist = fourier_encode_dist

        # radial function will assume a dimension of at minimum 1, for the relative distance - extra fourier features must be added to the edge dimension
        edge_dim += (0 if not fourier_encode_dist else (num_fourier_features * 2))

        # Neighbor -> center weights
        self.kernel_unary = nn.ModuleDict()

        for (di, mi), (do, mo) in (self.fiber_in * self.fiber_out):
            self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, edge_dim = edge_dim, splits = splits)

        self.pool = pool

        # Center -> center weights
        if self_interaction:
            assert self.pool, 'must pool edges if followed with self interaction'
            self.self_interact = LinearSE3(fiber_in, fiber_out)
            self.self_interact_sum = ResidualSE3()

    def forward(
        self,
        inp,
        edge_info,
        rel_dist = None,
        basis = None
    ):
        neighbor_indices, neighbor_masks, edges = edge_info
        rel_dist = rearrange(rel_dist, 'b m n -> b m n ()')

        kernels = {}
        outputs = {}

        if self.fourier_encode_dist:
            rel_dist = fourier_encode(rel_dist[..., None], num_encodings = self.num_fourier_features)

        for (di, mi), (do, mo) in (self.fiber_in * self.fiber_out):
            etype = f'({di},{do})'
            kernel_fn = self.kernel_unary[etype]

            edge_features = torch.cat((rel_dist, edges), dim = -1) if exists(edges) else rel_dist
            kernels[etype] = kernel_fn(edge_features, basis = basis)
        
        for degree_out in self.fiber_out.degrees:
            output = 0
            degree_out_key = str(degree_out)

            for degree_in, m_in in self.fiber_in:
                x = inp[str(degree_in)]

                x = batched_index_select(x, neighbor_indices, dim = 1)
                x = x.view(*x.shape[:3], to_order(degree_in) * m_in, 1)

                etype = f'({degree_in},{degree_out})'
                kernel = kernels[etype]
                output = output + einsum('... o i, ... i c -> ... o c', kernel, x)

            if self.pool:
                output = masked_mean(output, neighbor_masks, dim = 2) if exists(neighbor_masks) else output.mean(dim = 2)

            leading_shape = x.shape[:2] if self.pool else x.shape[:3]
            output = output.view(*leading_shape, -1, to_order(degree_out))

            outputs[degree_out_key] = output

        if self.self_interaction:
            self_interact_out = self.self_interact(inp)
            outputs = self.self_interact_sum(outputs, self_interact_out)

        return outputs

class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""
    def __init__(
        self,
        num_freq,
        in_dim,
        out_dim,
        edge_dim = None,
        mid_dim = 128
    ):
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.edge_dim = default(edge_dim, 0)

        self.net = nn.Sequential(
            nn.Linear(self.edge_dim + 1, mid_dim),
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
        edge_dim = 0,
        splits = 4
    ):
        super().__init__()
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        self.num_freq = to_order(min(degree_in, degree_out))
        self.d_out = to_order(degree_out)
        self.edge_dim = edge_dim

        self.rp = RadialFunc(self.num_freq, nc_in, nc_out, edge_dim)

        self.splits = splits

    def forward(self, feat, basis):
        splits = self.splits
        R = self.rp(feat)
        B = basis[f'{self.degree_in},{self.degree_out}']

        out_shape = (*R.shape[:3], self.d_out * self.nc_out, -1)

        # torch.sum(R * B, dim = -1) is too memory intensive
        # needs to be chunked to reduce peak memory usage

        B = B.expand(-1, -1, -1, R.shape[3], -1, -1, -1, -1)
        R, B = map(lambda t: rearrange(t, 'b n h s ... -> (b n h s) ...').split(splits, dim = 0), (R, B))

        out = []
        for r_chunk, b_chunk in zip(R, B):
            out.append(torch.sum(r_chunk * b_chunk, dim = -1))

        out = torch.cat(out, dim = 0)

        # reshape and out
        return out.view(*out_shape)

# feed forwards

class FeedForwardSE3(nn.Module):
    def __init__(
        self,
        fiber,
        mult = 4
    ):
        super().__init__()
        self.fiber = fiber
        fiber_hidden = Fiber(list(map(lambda t: (t[0], t[1] * mult), fiber)))

        self.project_in  = LinearSE3(fiber, fiber_hidden)
        self.nonlin      = NormSE3(fiber_hidden)
        self.project_out = LinearSE3(fiber_hidden, fiber)

    def forward(self, features):
        outputs = self.project_in(features)
        outputs = self.nonlin(outputs)
        outputs = self.project_out(outputs)
        return outputs

class FeedForwardBlockSE3(nn.Module):
    def __init__(
        self,
        fiber,
    ):
        super().__init__()
        self.fiber = fiber
        self.prenorm = NormSE3(fiber)
        self.feedforward = FeedForwardSE3(fiber)
        self.residual = ResidualSE3()

    def forward(self, features):
        res = features
        out = self.prenorm(features)
        out = self.feedforward(out)
        return self.residual(out, res)

# attention

class AttentionSE3(nn.Module):
    def __init__(
        self,
        fiber,
        dim_head = 64,
        heads = 8,
        attend_self = False,
        edge_dim = None,
        fourier_encode_dist = False,
        rel_dist_num_fourier_features = 4,
        use_null_kv = False,
        splits = 4
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        hidden_fiber = Fiber(list(map(lambda t: (t[0], hidden_dim), fiber)))
        project_out = not (heads == 1 and len(fiber.dims) == 1 and dim_head == fiber.dims[0])

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = LinearSE3(fiber, hidden_fiber)
        self.to_k = ConvSE3(fiber, hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)
        self.to_v = ConvSE3(fiber, hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)
        self.to_out = LinearSE3(hidden_fiber, fiber) if project_out else nn.Identity()

        self.use_null_kv = use_null_kv
        if use_null_kv:
            self.null_keys = nn.ParameterDict()
            self.null_values = nn.ParameterDict()

            for degree in fiber.degrees:
                m = to_order(degree)
                degree_key = str(degree)
                self.null_keys[degree_key] = nn.Parameter(torch.zeros(heads, dim_head, m))
                self.null_values[degree_key] = nn.Parameter(torch.zeros(heads, dim_head, m))

        self.attend_self = attend_self
        if attend_self:
            self.to_self_k = LinearSE3(fiber, hidden_fiber)
            self.to_self_v = LinearSE3(fiber, hidden_fiber)

    def forward(self, features, edge_info, rel_dist, basis):
        h, attend_self = self.heads, self.attend_self
        device, dtype = get_tensor_device_and_dtype(features)
        _, neighbor_mask, edges = edge_info

        max_neg_value = -torch.finfo().max

        if exists(neighbor_mask):
            neighbor_mask = rearrange(neighbor_mask, 'b i j -> b () i j')

        queries = self.to_q(features)
        keys, values = self.to_k(features, edge_info, rel_dist, basis), self.to_v(features, edge_info, rel_dist, basis)

        if attend_self:
            self_keys, self_values = self.to_self_k(features), self.to_self_v(features)

        outputs = {}
        for degree in features.keys():
            q, k, v = map(lambda t: t[degree], (queries, keys, values))

            q = rearrange(q, 'b i (h d) m -> b h i d m', h = h)
            k, v = map(lambda t: rearrange(t, 'b i j (h d) m -> b h i j d m', h = h), (k, v))

            if self.use_null_kv:
                null_k, null_v = map(lambda t: t[degree], (self.null_keys, self.null_values))
                null_k, null_v = map(lambda t: repeat(t, 'h d m -> b h i () d m', b = q.shape[0], i = q.shape[2]), (null_k, null_v))
                k = torch.cat((null_k, k), dim = 3)
                v = torch.cat((null_v, v), dim = 3)

            if attend_self:
                self_k, self_v = map(lambda t: t[degree], (self_keys, self_values))
                self_k, self_v = map(lambda t: rearrange(t, 'b n (h d) m -> b h n () d m', h = h), (self_k, self_v))
                k = torch.cat((self_k, k), dim = 3)
                v = torch.cat((self_v, v), dim = 3)

            sim = einsum('b h i d m, b h i j d m -> b h i j', q, k) * self.scale

            if exists(neighbor_mask):
                num_left_pad = int(attend_self) + int(self.use_null_kv)
                mask = F.pad(neighbor_mask, (num_left_pad, 0), value = True)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim = -1)
            out = einsum('b h i j, b h i j d m -> b h i d m', attn, v)
            outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')

        return self.to_out(outputs)

class AttentionBlockSE3(nn.Module):
    def __init__(
        self,
        fiber,
        dim_head = 24,
        heads = 8,
        attend_self = False,
        edge_dim = None,
        use_null_kv = False,
        fourier_encode_dist = False,
        rel_dist_num_fourier_features = 4,
        splits = 4
    ):
        super().__init__()
        self.attn = AttentionSE3(fiber, heads = heads, dim_head = dim_head, attend_self = attend_self, edge_dim = edge_dim, use_null_kv = use_null_kv, rel_dist_num_fourier_features = rel_dist_num_fourier_features, splits = splits)
        self.prenorm = NormSE3(fiber)
        self.residual = ResidualSE3()

    def forward(self, features, edge_info, rel_dist, basis):
        res = features
        outputs = self.prenorm(features)
        outputs = self.attn(outputs, edge_info, rel_dist, basis)
        return self.residual(outputs, res)

# main class


class SE3Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 24,
        depth = 2,
        input_degrees = 1,
        num_degrees = 2,
        output_degrees = 1,
        valid_radius = 1e5,
        reduce_dim_out = False,
        num_tokens = None,
        num_edge_tokens = None,
        edge_dim = None,
        reversible = False,
        attend_self = True,
        use_null_kv = False,
        differentiable_coors = False,
        fourier_encode_dist = False,
        rel_dist_num_fourier_features = 4,
        num_neighbors = float('inf'),
        attend_sparse_neighbors = False,
        num_adj_degrees = None,
        adj_dim = 0,
        max_sparse_neighbors = float('inf'),
        dim_in = None,
        dim_out = None,
        norm_out = False,
        num_conv_layers = 0,
        causal = False,
        splits = 4
    ):
        super().__init__()
        dim_in = default(dim_in, dim)
        self.dim_in = cast_tuple(dim_in, input_degrees)

        self.dim = dim

        self.token_emb = None
        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        assert not (exists(num_edge_tokens) and not exists(edge_dim)), 'edge dimension (edge_dim) must be supplied if SE3 transformer is to have edge tokens'
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        self.has_edges = exists(edge_dim) and edge_dim > 0

        self.input_degrees = input_degrees
        self.num_degrees = num_degrees
        self.output_degrees = output_degrees

        # whether to differentiate through basis, needed for alphafold2

        self.differentiable_coors = differentiable_coors

        # neighbors hyperparameters

        self.valid_radius = valid_radius
        self.num_neighbors = num_neighbors

        # sparse neighbors, derived from adjacency matrix or edges being passed in

        self.attend_sparse_neighbors = attend_sparse_neighbors
        self.max_sparse_neighbors = max_sparse_neighbors

        # adjacent neighbor derivation and embed

        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'
        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None

        edge_dim = (edge_dim if self.has_edges else 0) + (adj_dim if exists(self.adj_emb) else 0)

        # define fibers and dimensionality

        dim_in = default(dim_in, dim)
        dim_out = default(dim_out, dim)

        fiber_in     = Fiber.create(input_degrees, dim_in)
        fiber_hidden = Fiber.create(num_degrees, dim)
        fiber_out    = Fiber.create(output_degrees, dim_out)

        conv_kwargs = dict(edge_dim = edge_dim, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)

        # causal

        assert not (causal and not attend_self), 'attending to self must be turned on if in autoregressive mode (for the first token)'
        self.causal = causal

        # main network

        self.conv_in  = ConvSE3(fiber_in, fiber_hidden, **conv_kwargs)

        # pre-convs

        self.convs = nn.ModuleList([])
        for _ in range(num_conv_layers):
            self.convs.append(nn.ModuleList([
                ConvSE3(fiber_hidden, fiber_hidden, **conv_kwargs),
                NormSE3(fiber_hidden)
            ]))

        # trunk

        layers = nn.ModuleList([])
        for _ in range(depth):
            layers.append(nn.ModuleList([
                AttentionBlockSE3(fiber_hidden, heads = heads, dim_head = dim_head, attend_self = attend_self, edge_dim = edge_dim, fourier_encode_dist = fourier_encode_dist, rel_dist_num_fourier_features = rel_dist_num_fourier_features, use_null_kv = use_null_kv, splits = splits),
                FeedForwardBlockSE3(fiber_hidden)
            ]))

        execution_class = ReversibleSequence if reversible else SequentialSequence
        self.net = execution_class(layers)

        # out

        self.conv_out = ConvSE3(fiber_hidden, fiber_out, **conv_kwargs)

        self.norm = NormSE3(fiber_out) if norm_out else nn.Identity()

        self.linear_out = LinearSE3(
            fiber_out,
            Fiber.create(output_degrees, 1)
        ) if reduce_dim_out else None

    def forward(
        self,
        feats,
        coors,
        mask = None,
        adj_mat = None,
        edges = None,
        return_type = None,
        return_pooled = False,
        neighbor_mask = None
    ):
        _mask = mask

        if self.output_degrees == 1:
            return_type = 0

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        assert not (self.attend_sparse_neighbors and not exists(adj_mat)), 'adjacency matrix (adjacency_mat) or edges (edges) must be passed in'
        assert not (self.has_edges and not exists(edges)), 'edge embedding (num_edge_tokens & edge_dim) must be supplied if one were to train on edge types'

        if torch.is_tensor(feats):
            feats = {'0': feats[..., None]}

        b, n, d, *_, device = *feats['0'].shape, feats['0'].device

        assert d == self.dim_in[0], f'feature dimension {d} must be equal to dimension given at init {self.dim_in[0]}'
        assert set(map(int, feats.keys())) == set(range(self.input_degrees)), f'input must have {self.input_degrees} degree'

        num_degrees, neighbors, max_sparse_neighbors, valid_radius = self.num_degrees, self.num_neighbors, self.max_sparse_neighbors, self.valid_radius

        assert self.attend_sparse_neighbors or neighbors > 0, 'you must either attend to sparsely bonded neighbors, or set number of locally attended neighbors to be greater than 0'

        # se3 transformer by default cannot have a node attend to itself

        exclude_self_mask = rearrange(~torch.eye(n, dtype = torch.bool, device = device), 'i j -> () i j')
        remove_self = lambda t: t.masked_select(exclude_self_mask).reshape(b, n, n - 1)
        get_max_value = lambda t: torch.finfo(t.dtype).max

        # create N-degrees adjacent matrix from 1st degree connections

        if exists(self.num_adj_degrees):
            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            adj_indices = adj_indices.masked_select(exclude_self_mask).reshape(b, n, n - 1)

        # calculate sparsely connected neighbors

        sparse_neighbor_mask = None
        num_sparse_neighbors = 0

        if self.attend_sparse_neighbors:
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if exists(adj_mat):
                if len(adj_mat.shape) == 2:
                    adj_mat = repeat(adj_mat, 'i j -> b i j', b = b)

            adj_mat = remove_self(adj_mat)

            adj_mat_values = adj_mat.float()
            adj_mat_max_neighbors = adj_mat_values.sum(dim = -1).max().item()

            if max_sparse_neighbors < adj_mat_max_neighbors:
                noise = torch.empty_like(adj_mat_values).uniform_(-0.01, 0.01)
                adj_mat_values += noise

            num_sparse_neighbors = int(min(max_sparse_neighbors, adj_mat_max_neighbors))
            values, indices = adj_mat_values.topk(num_sparse_neighbors, dim = -1)
            sparse_neighbor_mask = torch.zeros_like(adj_mat_values).scatter_(-1, indices, values)
            sparse_neighbor_mask = sparse_neighbor_mask > 0.5

        # exclude edge of token to itself

        indices = repeat(torch.arange(n, device = device), 'i -> b i j', b = b, j = n)
        rel_pos  = rearrange(coors, 'b n d -> b n () d') - rearrange(coors, 'b n d -> b () n d')

        indices = indices.masked_select(exclude_self_mask).reshape(b, n, n - 1)
        rel_pos = rel_pos.masked_select(exclude_self_mask[..., None]).reshape(b, n, n - 1, 3)

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = mask.masked_select(exclude_self_mask).reshape(b, n, n - 1)

        if exists(edges):
            if exists(self.edge_emb):
                edges = self.edge_emb(edges)

            edges = edges.masked_select(exclude_self_mask[..., None]).reshape(b, n, n - 1, -1)

        if exists(self.adj_emb):
            adj_emb = self.adj_emb(adj_indices)
            edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        rel_dist = rel_pos.norm(dim = -1)

        # rel_dist gets modified using adjacency or neighbor mask

        modified_rel_dist = rel_dist
        max_value = get_max_value(modified_rel_dist) # for masking out nodes from being considered as neighbors

        # neighbors

        if exists(neighbor_mask):
            neighbor_mask = remove_self(neighbor_mask)

            max_neighbors = neighbor_mask.sum(dim = -1).max().item()
            if max_neighbors > neighbors:
                print(f'neighbor_mask shows maximum number of neighbors as {max_neighbors} but specified number of neighbors is {neighbors}')

            modified_rel_dist.masked_fill_(~neighbor_mask, max_value)

        # use sparse neighbor mask to assign priority of bonded

        if exists(sparse_neighbor_mask):
            modified_rel_dist.masked_fill_(sparse_neighbor_mask, 0.)

        # mask out future nodes to high distance if causal turned on

        if self.causal:
            causal_mask = torch.ones(n, n - 1, device = device).triu().bool()
            modified_rel_dist.masked_fill_(causal_mask[None, ...], max_value)

        # if number of local neighbors by distance is set to 0, then only fetch the sparse neighbors defined by adjacency matrix

        if neighbors == 0:
            valid_radius = 0

        # get neighbors and neighbor mask, excluding self

        neighbors = int(min(neighbors, n - 1))
        total_neighbors = int(neighbors + num_sparse_neighbors)
        assert total_neighbors > 0, 'you must be fetching at least 1 neighbor'

        total_neighbors = int(min(total_neighbors, n - 1)) # make sure total neighbors does not exceed the length of the sequence itself

        dist_values, nearest_indices = modified_rel_dist.topk(total_neighbors, dim = -1, largest = False)
        neighbor_mask = dist_values <= valid_radius

        neighbor_rel_dist = batched_index_select(rel_dist, nearest_indices, dim = 2)
        neighbor_rel_pos = batched_index_select(rel_pos, nearest_indices, dim = 2)
        neighbor_indices = batched_index_select(indices, nearest_indices, dim = 2)

        if exists(mask):
            neighbor_mask = neighbor_mask & batched_index_select(mask, nearest_indices, dim = 2)

        if exists(edges):
            edges = batched_index_select(edges, nearest_indices, dim = 2)

        # calculate basis

        basis = get_basis(neighbor_rel_pos, num_degrees - 1, differentiable = self.differentiable_coors)

        # main logic

        edge_info = (neighbor_indices, neighbor_mask, edges)
        x = feats

        # project in

        x = self.conv_in(x, edge_info, rel_dist = neighbor_rel_dist, basis = basis)

        # preconvolution layers

        for conv, nonlin in self.convs:
            x = conv(x, edge_info, rel_dist = neighbor_rel_dist, basis = basis)
            x = nonlin(x)

        # transformer layers

        x = self.net(x, edge_info = edge_info, rel_dist = neighbor_rel_dist, basis = basis)

        # project out

        x = self.conv_out(x, edge_info, rel_dist = neighbor_rel_dist, basis = basis)

        # norm

        x = self.norm(x)

        # reduce dim if specified

        if exists(self.linear_out):
            x = self.linear_out(x)
            x = map_values(lambda t: t.squeeze(dim = 2), x)

        if return_pooled:
            mask_fn = (lambda t: masked_mean(t, _mask, dim = 1)) if exists(_mask) else (lambda t: t.mean(dim = 1))
            x = map_values(mask_fn, x)

        if '0' in x:
            x['0'] = x['0'].squeeze(dim = -1)

        if exists(return_type):
            return x[str(return_type)]

        return x
