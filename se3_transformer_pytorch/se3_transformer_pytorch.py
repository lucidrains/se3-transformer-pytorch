from math import sqrt
from itertools import product
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, einsum

from se3_transformer_pytorch.basis import get_basis
from se3_transformer_pytorch.utils import exists, default, uniq, map_values, batched_index_select, masked_mean, to_order, fourier_encode, cast_tuple, safe_cat, fast_split, rand_uniform, broadcat
from se3_transformer_pytorch.reversible import ReversibleSequence, SequentialSequence
from se3_transformer_pytorch.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

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
            structure = [FiberEl(degree, dim) for degree, dim in structure.items()]
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
        gated_scale = False,
        eps = 1e-12,
    ):
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.eps = eps

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for degree, chan in fiber:
            self.transform[str(degree)] = nn.ParameterDict({
                'scale': nn.Parameter(torch.ones(1, 1, chan)) if not gated_scale else None,
                'bias': nn.Parameter(rand_uniform((1, 1, chan), -1e-3, 1e-3)),
                'w_gate': nn.Parameter(rand_uniform((chan, chan), -1e-3, 1e-3)) if gated_scale else None
            })

    def forward(self, features):
        output = {}
        for degree, t in features.items():
            # Compute the norms and normalized features
            norm = t.norm(dim = -1, keepdim = True).clamp(min = self.eps)
            phase = t / norm

            # Transform on norms
            parameters = self.transform[degree]
            gate_weights, bias, scale = parameters['w_gate'], parameters['bias'], parameters['scale']

            transformed = rearrange(norm, '... () -> ...')

            if not exists(scale):
                scale = einsum('b n d, d e -> b n e', transformed, gate_weights)

            transformed = self.nonlin(transformed * scale + bias)
            transformed = rearrange(transformed, '... -> ... ()')

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

        self.splits = splits # for splitting the computation of kernel and basis, to reduce peak memory usage

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
        splits = self.splits
        neighbor_indices, neighbor_masks, edges = edge_info
        rel_dist = rearrange(rel_dist, 'b m n -> b m n ()')

        kernels = {}
        outputs = {}

        if self.fourier_encode_dist:
            rel_dist = fourier_encode(rel_dist[..., None], num_encodings = self.num_fourier_features)

        # split basis

        basis_keys = basis.keys()
        split_basis_values = list(zip(*list(map(lambda t: fast_split(t, splits, dim = 1), basis.values()))))
        split_basis = list(map(lambda v: dict(zip(basis_keys, v)), split_basis_values))

        # go through every permutation of input degree type to output degree type

        for degree_out in self.fiber_out.degrees:
            output = 0
            degree_out_key = str(degree_out)

            for degree_in, m_in in self.fiber_in:
                etype = f'({degree_in},{degree_out})'

                x = inp[str(degree_in)]

                x = batched_index_select(x, neighbor_indices, dim = 1)
                x = x.view(*x.shape[:3], to_order(degree_in) * m_in, 1)

                kernel_fn = self.kernel_unary[etype]
                edge_features = torch.cat((rel_dist, edges), dim = -1) if exists(edges) else rel_dist

                output_chunk = None
                split_x = fast_split(x, splits, dim = 1)
                split_edge_features = fast_split(edge_features, splits, dim = 1)

                # process input, edges, and basis in chunks along the sequence dimension

                for x_chunk, edge_features, basis in zip(split_x, split_edge_features, split_basis):
                    kernel = kernel_fn(edge_features, basis = basis)
                    chunk = einsum('... o i, ... i c -> ... o c', kernel, x_chunk)
                    output_chunk = safe_cat(output_chunk, chunk, dim = 1)

                output = output + output_chunk

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
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
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

        out = 0
        for i in range(R.shape[-1]):
            out += R[..., i] * B[..., i]

        out = rearrange(out, 'b n h s ... -> (b n h s) ...')

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
        norm_gated_scale = False
    ):
        super().__init__()
        self.fiber = fiber
        self.prenorm = NormSE3(fiber, gated_scale = norm_gated_scale)
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
        splits = 4,
        global_feats_dim = None,
        linear_proj_keys = False,
        tie_key_values = False
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        hidden_fiber = Fiber(list(map(lambda t: (t[0], hidden_dim), fiber)))
        project_out = not (heads == 1 and len(fiber.dims) == 1 and dim_head == fiber.dims[0])

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.linear_proj_keys = linear_proj_keys # whether to linearly project features for keys, rather than convolve with basis

        self.to_q = LinearSE3(fiber, hidden_fiber)
        self.to_v = ConvSE3(fiber, hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)

        assert not (linear_proj_keys and tie_key_values), 'you cannot do linear projection of keys and have shared key / values turned on at the same time'

        if linear_proj_keys:
            self.to_k = LinearSE3(fiber, hidden_fiber)
        elif not tie_key_values:
            self.to_k = ConvSE3(fiber, hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)
        else:
            self.to_k = None

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

        self.accept_global_feats = exists(global_feats_dim)
        if self.accept_global_feats:
            global_input_fiber = Fiber.create(1, global_feats_dim)
            global_output_fiber = Fiber.create(1, hidden_fiber[0])
            self.to_global_k = LinearSE3(global_input_fiber, global_output_fiber)
            self.to_global_v = LinearSE3(global_input_fiber, global_output_fiber)

    def forward(self, features, edge_info, rel_dist, basis, global_feats = None, pos_emb = None, mask = None):
        h, attend_self = self.heads, self.attend_self
        device, dtype = get_tensor_device_and_dtype(features)
        neighbor_indices, neighbor_mask, edges = edge_info

        if exists(neighbor_mask):
            neighbor_mask = rearrange(neighbor_mask, 'b i j -> b () i j')

        queries = self.to_q(features)
        values  = self.to_v(features, edge_info, rel_dist, basis)

        if self.linear_proj_keys:
            keys = self.to_k(features)
            keys = map_values(lambda val: batched_index_select(val, neighbor_indices, dim = 1), keys)
        elif not exists(self.to_k):
            keys = values
        else:
            keys = self.to_k(features, edge_info, rel_dist, basis)

        if attend_self:
            self_keys, self_values = self.to_self_k(features), self.to_self_v(features)

        if exists(global_feats):
            global_keys, global_values = self.to_global_k(global_feats), self.to_global_v(global_feats)

        outputs = {}
        for degree in features.keys():
            q, k, v = map(lambda t: t[degree], (queries, keys, values))

            q = rearrange(q, 'b i (h d) m -> b h i d m', h = h)
            k, v = map(lambda t: rearrange(t, 'b i j (h d) m -> b h i j d m', h = h), (k, v))

            if attend_self:
                self_k, self_v = map(lambda t: t[degree], (self_keys, self_values))
                self_k, self_v = map(lambda t: rearrange(t, 'b n (h d) m -> b h n () d m', h = h), (self_k, self_v))
                k = torch.cat((self_k, k), dim = 3)
                v = torch.cat((self_v, v), dim = 3)

            if exists(pos_emb) and degree == '0':
                query_pos_emb, key_pos_emb = pos_emb
                query_pos_emb = rearrange(query_pos_emb, 'b i d -> b () i d ()')
                key_pos_emb = rearrange(key_pos_emb, 'b i j d -> b () i j d ()')
                q = apply_rotary_pos_emb(q, query_pos_emb)
                k = apply_rotary_pos_emb(k, key_pos_emb)
                v = apply_rotary_pos_emb(v, key_pos_emb)

            if self.use_null_kv:
                null_k, null_v = map(lambda t: t[degree], (self.null_keys, self.null_values))
                null_k, null_v = map(lambda t: repeat(t, 'h d m -> b h i () d m', b = q.shape[0], i = q.shape[2]), (null_k, null_v))
                k = torch.cat((null_k, k), dim = 3)
                v = torch.cat((null_v, v), dim = 3)

            if exists(global_feats) and degree == '0':
                global_k, global_v = map(lambda t: t[degree], (global_keys, global_values))
                global_k, global_v = map(lambda t: repeat(t, 'b j (h d) m -> b h i j d m', h = h, i = k.shape[2]), (global_k, global_v))
                k = torch.cat((global_k, k), dim = 3)
                v = torch.cat((global_v, v), dim = 3)

            sim = einsum('b h i d m, b h i j d m -> b h i j', q, k) * self.scale

            if exists(neighbor_mask):
                num_left_pad = sim.shape[-1] - neighbor_mask.shape[-1]
                mask = F.pad(neighbor_mask, (num_left_pad, 0), value = True)
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)
            out = einsum('b h i j, b h i j d m -> b h i d m', attn, v)
            outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')

        return self.to_out(outputs)

# AttentionSE3, but with one key / value projection shared across all query heads
class OneHeadedKVAttentionSE3(nn.Module):
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
        splits = 4,
        global_feats_dim = None,
        linear_proj_keys = False,
        tie_key_values = False
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        hidden_fiber = Fiber(list(map(lambda t: (t[0], hidden_dim), fiber)))
        kv_hidden_fiber = Fiber(list(map(lambda t: (t[0], dim_head), fiber)))
        project_out = not (heads == 1 and len(fiber.dims) == 1 and dim_head == fiber.dims[0])

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.linear_proj_keys = linear_proj_keys # whether to linearly project features for keys, rather than convolve with basis

        self.to_q = LinearSE3(fiber, hidden_fiber)
        self.to_v = ConvSE3(fiber, kv_hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)

        assert not (linear_proj_keys and tie_key_values), 'you cannot do linear projection of keys and have shared key / values turned on at the same time'

        if linear_proj_keys:
            self.to_k = LinearSE3(fiber, kv_hidden_fiber)
        elif not tie_key_values:
            self.to_k = ConvSE3(fiber, kv_hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)
        else:
            self.to_k = None

        self.to_out = LinearSE3(hidden_fiber, fiber) if project_out else nn.Identity()

        self.use_null_kv = use_null_kv
        if use_null_kv:
            self.null_keys = nn.ParameterDict()
            self.null_values = nn.ParameterDict()

            for degree in fiber.degrees:
                m = to_order(degree)
                degree_key = str(degree)
                self.null_keys[degree_key] = nn.Parameter(torch.zeros(dim_head, m))
                self.null_values[degree_key] = nn.Parameter(torch.zeros(dim_head, m))

        self.attend_self = attend_self
        if attend_self:
            self.to_self_k = LinearSE3(fiber, kv_hidden_fiber)
            self.to_self_v = LinearSE3(fiber, kv_hidden_fiber)

        self.accept_global_feats = exists(global_feats_dim)
        if self.accept_global_feats:
            global_input_fiber = Fiber.create(1, global_feats_dim)
            global_output_fiber = Fiber.create(1, kv_hidden_fiber[0])
            self.to_global_k = LinearSE3(global_input_fiber, global_output_fiber)
            self.to_global_v = LinearSE3(global_input_fiber, global_output_fiber)

    def forward(self, features, edge_info, rel_dist, basis, global_feats = None, pos_emb = None, mask = None):
        h, attend_self = self.heads, self.attend_self
        device, dtype = get_tensor_device_and_dtype(features)
        neighbor_indices, neighbor_mask, edges = edge_info

        if exists(neighbor_mask):
            neighbor_mask = rearrange(neighbor_mask, 'b i j -> b () i j')

        queries = self.to_q(features)
        values  = self.to_v(features, edge_info, rel_dist, basis)

        if self.linear_proj_keys:
            keys = self.to_k(features)
            keys = map_values(lambda val: batched_index_select(val, neighbor_indices, dim = 1), keys)
        elif not exists(self.to_k):
            keys = values
        else:
            keys = self.to_k(features, edge_info, rel_dist, basis)

        if attend_self:
            self_keys, self_values = self.to_self_k(features), self.to_self_v(features)

        if exists(global_feats):
            global_keys, global_values = self.to_global_k(global_feats), self.to_global_v(global_feats)

        outputs = {}
        for degree in features.keys():
            q, k, v = map(lambda t: t[degree], (queries, keys, values))

            q = rearrange(q, 'b i (h d) m -> b h i d m', h = h)

            if attend_self:
                self_k, self_v = map(lambda t: t[degree], (self_keys, self_values))
                self_k, self_v = map(lambda t: rearrange(t, 'b n d m -> b n () d m'), (self_k, self_v))
                k = torch.cat((self_k, k), dim = 2)
                v = torch.cat((self_v, v), dim = 2)

            if exists(pos_emb) and degree == '0':
                query_pos_emb, key_pos_emb = pos_emb
                query_pos_emb = rearrange(query_pos_emb, 'b i d -> b () i d ()')
                key_pos_emb = rearrange(key_pos_emb, 'b i j d -> b i j d ()')
                q = apply_rotary_pos_emb(q, query_pos_emb)
                k = apply_rotary_pos_emb(k, key_pos_emb)
                v = apply_rotary_pos_emb(v, key_pos_emb)

            if self.use_null_kv:
                null_k, null_v = map(lambda t: t[degree], (self.null_keys, self.null_values))
                null_k, null_v = map(lambda t: repeat(t, 'd m -> b i () d m', b = q.shape[0], i = q.shape[2]), (null_k, null_v))
                k = torch.cat((null_k, k), dim = 2)
                v = torch.cat((null_v, v), dim = 2)

            if exists(global_feats) and degree == '0':
                global_k, global_v = map(lambda t: t[degree], (global_keys, global_values))
                global_k, global_v = map(lambda t: repeat(t, 'b j d m -> b i j d m', i = k.shape[1]), (global_k, global_v))
                k = torch.cat((global_k, k), dim = 2)
                v = torch.cat((global_v, v), dim = 2)

            sim = einsum('b h i d m, b i j d m -> b h i j', q, k) * self.scale

            if exists(neighbor_mask):
                num_left_pad = sim.shape[-1] - neighbor_mask.shape[-1]
                mask = F.pad(neighbor_mask, (num_left_pad, 0), value = True)
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)
            out = einsum('b h i j, b i j d m -> b h i d m', attn, v)
            outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')

        return self.to_out(outputs)

# global linear attention - only for type 0

class GlobalLinearAttention(nn.Module):
    def __init__(
        self,
        fiber,
        dim_head = 64,
        heads = 8,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_qkv = nn.Linear(fiber[0], inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, fiber[0])

    def forward(self, features, edge_info, rel_dist, basis, global_feats = None, pos_emb = None, mask = None):
        h = self.heads
        device, dtype = get_tensor_device_and_dtype(features)

        x = features['0'] # only working on type 0 features for global linear attention
        x = rearrange(x, '... () -> ...')

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n ()')
            k = k.masked_fill(~mask, -torch.finfo(k.dtype).max)
            v = v.masked_fill(~mask, 0.)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q *= self.scale

        context = einsum('b h n d, b h n e -> b h d e', k, v)
        attn_out = einsum('b h d e, b h n d -> b h n e', context, q)
        attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')
        attn_out = self.to_out(attn_out)

        out = map_values(lambda *args: 0, features)
        out['0'] = rearrange(attn_out, '... -> ... ()')
        return out

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
        splits = 4,
        global_feats_dim = False,
        linear_proj_keys = False,
        tie_key_values = False,
        attention_klass = AttentionSE3,
        norm_gated_scale = False
    ):
        super().__init__()
        self.attn = attention_klass(fiber, heads = heads, dim_head = dim_head, attend_self = attend_self, edge_dim = edge_dim, use_null_kv = use_null_kv, rel_dist_num_fourier_features = rel_dist_num_fourier_features, fourier_encode_dist =fourier_encode_dist, splits = splits, global_feats_dim = global_feats_dim, linear_proj_keys = linear_proj_keys, tie_key_values = tie_key_values)
        self.prenorm = NormSE3(fiber, gated_scale = norm_gated_scale)
        self.residual = ResidualSE3()

    def forward(self, features, edge_info, rel_dist, basis, global_feats = None, pos_emb = None, mask = None):
        res = features
        outputs = self.prenorm(features)
        outputs = self.attn(outputs, edge_info, rel_dist, basis, global_feats, pos_emb, mask)
        return self.residual(outputs, res)

# egnn

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

class HtypesNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8, scale_init = 1e-2, bias_init = 1e-2):
        super().__init__()
        self.eps = eps
        scale = torch.empty(1, 1, 1, dim, 1).fill_(scale_init)
        bias = torch.empty(1, 1, 1, dim, 1).fill_(bias_init)
        self.scale = nn.Parameter(scale)
        self.bias = nn.Parameter(bias)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * (norm * self.scale + self.bias)

class EGNN(nn.Module):
    def __init__(
        self,
        fiber,
        hidden_dim = 32,
        edge_dim = 0,
        init_eps = 1e-3,
        coor_weights_clamp_value = None
    ):
        super().__init__()
        self.fiber = fiber
        node_dim = fiber[0]

        htypes = list(filter(lambda t: t.degrees != 0, fiber))
        num_htypes = len(htypes)
        htype_dims = sum([fiberel.dim for fiberel in htypes])

        edge_input_dim = node_dim * 2 + htype_dims + edge_dim + 1

        self.node_norm = nn.LayerNorm(node_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            SiLU(),
            nn.Linear(edge_input_dim * 2, hidden_dim),
            SiLU()
        )

        self.htype_norms = nn.ModuleDict({})
        self.htype_gating = nn.ModuleDict({})

        for degree, dim in fiber:
            if degree == 0:
                continue
            self.htype_norms[str(degree)] = HtypesNorm(dim)
            self.htype_gating[str(degree)] = nn.Linear(node_dim, dim)

        self.htypes_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            SiLU(),
            nn.Linear(hidden_dim * 4, htype_dims)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim * 2),
            SiLU(),
            nn.Linear(node_dim * 2, node_dim)
        )

        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(
        self,
        features,
        edge_info,
        rel_dist,
        mask = None,
        **kwargs
    ):
        neighbor_indices, neighbor_masks, edges = edge_info

        mask = neighbor_masks

        # type 0 features

        nodes = features['0']
        nodes = rearrange(nodes, '... () -> ...')

        # higher types (htype)

        htypes = list(filter(lambda t: t[0] != '0', features.items()))
        htype_degrees = list(map(lambda t: t[0], htypes))
        htype_dims = list(map(lambda t: t[1].shape[-2], htypes))

        # prepare higher types

        rel_htypes = []
        rel_htypes_dists = []

        for degree, htype in htypes:
            rel_htype = rearrange(htype, 'b i d m -> b i () d m') - rearrange(htype, 'b j d m -> b () j d m')
            rel_htype_dist = rel_htype.norm(dim = -1)

            rel_htypes.append(rel_htype)
            rel_htypes_dists.append(rel_htype_dist)

        # prepare edges for edge MLP

        nodes_i = rearrange(nodes, 'b i d -> b i () d')
        nodes_j = batched_index_select(nodes, neighbor_indices, dim = 1)
        neighbor_higher_type_dists = map(lambda t: batched_index_select(t, neighbor_indices, dim = 2), rel_htypes_dists)
        coor_rel_dist = rearrange(rel_dist, 'b i j -> b i j ()')

        edge_mlp_inputs = broadcat((nodes_i, nodes_j, *neighbor_higher_type_dists, coor_rel_dist), dim = -1)

        if exists(edges):
            edge_mlp_inputs = torch.cat((edge_mlp_inputs, edges), dim = -1)

        # get intermediate representation

        m_ij = self.edge_mlp(edge_mlp_inputs)

        # to coordinates

        htype_weights = self.htypes_mlp(m_ij)

        if exists(self.coor_weights_clamp_value):
            clamp_value = self.coor_weights_clamp_value
            htype_weights.clamp_(min = -clamp_value, max = clamp_value)

        split_htype_weights = htype_weights.split(htype_dims, dim = -1)

        htype_updates = []

        if exists(mask):
            htype_mask = rearrange(mask, 'b i j -> b i j ()')
            htype_weights = htype_weights.masked_fill(~htype_mask, 0.)

        for degree, rel_htype, htype_weight in zip(htype_degrees, rel_htypes, split_htype_weights):
            normed_rel_htype = self.htype_norms[str(degree)](rel_htype)
            normed_rel_htype = batched_index_select(normed_rel_htype, neighbor_indices, dim = 2)

            htype_update = einsum('b i j d m, b i j d -> b i d m', normed_rel_htype, htype_weight)
            htype_updates.append(htype_update)

        # to nodes

        if exists(mask):
            m_ij_mask = rearrange(mask, '... -> ... ()')
            m_ij = m_ij.masked_fill(~m_ij_mask, 0.)

        m_i = m_ij.sum(dim = -2)

        normed_nodes = self.node_norm(nodes)
        node_mlp_input = torch.cat((normed_nodes, m_i), dim = -1)
        node_out = self.node_mlp(node_mlp_input) + nodes

        # update nodes

        features['0'] = rearrange(node_out, '... -> ... ()')

        # update higher types

        update_htype_dicts = dict(zip(htype_degrees, htype_updates))

        for degree, update_htype in update_htype_dicts.items():
            features[degree] = features[degree] + update_htype

        for degree in htype_degrees:
            gating = self.htype_gating[str(degree)](node_out).sigmoid()
            features[degree] = features[degree] * rearrange(gating, '... -> ... ()')

        return features

class EGnnNetwork(nn.Module):
    def __init__(
        self,
        *,
        fiber,
        depth,
        edge_dim = 0,
        hidden_dim = 32,
        coor_weights_clamp_value = None,
        feedforward = False
    ):
        super().__init__()
        self.fiber = fiber
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                EGNN(fiber = fiber, edge_dim = edge_dim, hidden_dim = hidden_dim, coor_weights_clamp_value = coor_weights_clamp_value),
                FeedForwardBlockSE3(fiber) if feedforward else None
            ]))

    def forward(
        self,
        features,
        edge_info,
        rel_dist,
        basis,
        global_feats = None,
        pos_emb = None,
        mask = None,
        **kwargs
    ):
        neighbor_indices, neighbor_masks, edges = edge_info
        device = neighbor_indices.device

        # modify neighbors to include self (since se3 transformer depends on removing attention to token self, but this does not apply for EGNN)

        self_indices = torch.arange(neighbor_indices.shape[1], device = device)
        self_indices = rearrange(self_indices, 'i -> () i ()')
        neighbor_indices = broadcat((self_indices, neighbor_indices), dim = -1)

        neighbor_masks = F.pad(neighbor_masks, (1, 0), value = True)
        rel_dist = F.pad(rel_dist, (1, 0), value = 0.)

        if exists(edges):
            edges = F.pad(edges, (0, 0, 1, 0), value = 0.)  # make edge of token to itself 0 for now

        edge_info = (neighbor_indices, neighbor_masks, edges)

        # go through layers

        for egnn, ff in self.layers:
            features = egnn(
                features,
                edge_info = edge_info,
                rel_dist = rel_dist,
                basis = basis,
                global_feats = global_feats,
                pos_emb = pos_emb,
                mask = mask,
                **kwargs
            )

            if exists(ff):
                features = ff(features)

        return features

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
        num_degrees = None,
        output_degrees = 1,
        valid_radius = 1e5,
        reduce_dim_out = False,
        num_tokens = None,
        num_positions = None,
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
        splits = 4,
        global_feats_dim = None,
        linear_proj_keys = False,
        one_headed_key_values = False,
        tie_key_values = False,
        rotary_position = False,
        rotary_rel_dist = False,
        global_linear_attn_every = 0,
        norm_gated_scale = False,
        use_egnn = False,
        egnn_hidden_dim = 32,
        egnn_weights_clamp_value = None,
        egnn_feedforward = False,
        hidden_fiber_dict = None,
        out_fiber_dict = None
    ):
        super().__init__()
        dim_in = default(dim_in, dim)
        self.dim_in = cast_tuple(dim_in, input_degrees)
        self.dim = dim

        # token embedding

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        # positional embedding

        self.num_positions = num_positions
        self.pos_emb = nn.Embedding(num_positions, dim) if exists(num_positions) else None

        self.rotary_rel_dist = rotary_rel_dist
        self.rotary_position = rotary_position

        self.rotary_pos_emb = None
        if rotary_position or rotary_rel_dist:
            num_rotaries = int(rotary_position) + int(rotary_rel_dist)
            self.rotary_pos_emb = SinusoidalEmbeddings(dim_head // num_rotaries)

        # edges

        assert not (exists(num_edge_tokens) and not exists(edge_dim)), 'edge dimension (edge_dim) must be supplied if SE3 transformer is to have edge tokens'

        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        self.has_edges = exists(edge_dim) and edge_dim > 0

        self.input_degrees = input_degrees

        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'

        self.num_degrees = num_degrees if exists(num_degrees) else (max(hidden_fiber_dict.keys()) + 1)

        output_degrees = output_degrees if not use_egnn else None
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

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None

        edge_dim = (edge_dim if self.has_edges else 0) + (adj_dim if exists(self.adj_emb) else 0)

        # define fibers and dimensionality

        dim_in = default(dim_in, dim)
        dim_out = default(dim_out, dim)

        assert exists(num_degrees) or exists(hidden_fiber_dict), 'either num_degrees or hidden_fiber_dict must be specified'

        fiber_in     = Fiber.create(input_degrees, dim_in)

        if exists(hidden_fiber_dict):
            fiber_hidden = Fiber(hidden_fiber_dict)
        elif exists(num_degrees):
            fiber_hidden = Fiber.create(num_degrees, dim)

        if exists(out_fiber_dict):
            fiber_out = Fiber(out_fiber_dict)
            self.output_degrees = max(out_fiber_dict.keys()) + 1
        elif exists(output_degrees):
            fiber_out = Fiber.create(output_degrees, dim_out)
        else:
            fiber_out = None

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
                NormSE3(fiber_hidden, gated_scale = norm_gated_scale)
            ]))

        # global features

        self.accept_global_feats = exists(global_feats_dim)
        assert not (reversible and self.accept_global_feats), 'reversibility and global features are not compatible'

        # trunk

        self.attend_self = attend_self

        default_attention_klass = OneHeadedKVAttentionSE3 if one_headed_key_values else AttentionSE3

        if use_egnn:
            self.net = EGnnNetwork(fiber = fiber_hidden, depth = depth, edge_dim = edge_dim, hidden_dim = egnn_hidden_dim, coor_weights_clamp_value = egnn_weights_clamp_value, feedforward = egnn_feedforward)
        else:
            layers = nn.ModuleList([])
            for ind in range(depth):
                use_global_linear_attn = global_linear_attn_every > 0 and (ind % global_linear_attn_every) == 0
                attention_klass = default_attention_klass if not use_global_linear_attn else GlobalLinearAttention

                layers.append(nn.ModuleList([
                    AttentionBlockSE3(fiber_hidden, heads = heads, dim_head = dim_head, attend_self = attend_self, edge_dim = edge_dim, fourier_encode_dist = fourier_encode_dist, rel_dist_num_fourier_features = rel_dist_num_fourier_features, use_null_kv = use_null_kv, splits = splits, global_feats_dim = global_feats_dim, linear_proj_keys = linear_proj_keys, attention_klass = attention_klass, tie_key_values = tie_key_values, norm_gated_scale = norm_gated_scale),
                    FeedForwardBlockSE3(fiber_hidden, norm_gated_scale = norm_gated_scale)
                ]))

            execution_class = ReversibleSequence if reversible else SequentialSequence
            self.net = execution_class(layers)

        # out

        self.conv_out = ConvSE3(fiber_hidden, fiber_out, **conv_kwargs) if exists(fiber_out) else None

        self.norm = NormSE3(fiber_out, gated_scale = norm_gated_scale, nonlin = nn.Identity()) if (norm_out or reversible) and exists(fiber_out) else nn.Identity()

        final_fiber = default(fiber_out, fiber_hidden)

        self.linear_out = LinearSE3(
            final_fiber,
            Fiber(list(map(lambda t: FiberEl(degrees = t[0], dim = 1), final_fiber)))
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
        neighbor_mask = None,
        global_feats = None
    ):
        assert not (self.accept_global_feats ^ exists(global_feats)), 'you cannot pass in global features unless you init the class correctly'

        _mask = mask

        if self.output_degrees == 1:
            return_type = 0

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        if exists(self.pos_emb):
            assert feats.shape[1] <= self.num_positions, 'feature sequence length must be less than the number of positions given at init'
            pos_emb = self.pos_emb(torch.arange(feats.shape[1], device = feats.device))
            feats += rearrange(pos_emb, 'n d -> () n d')

        assert not (self.attend_sparse_neighbors and not exists(adj_mat)), 'adjacency matrix (adjacency_mat) or edges (edges) must be passed in'
        assert not (self.has_edges and not exists(edges)), 'edge embedding (num_edge_tokens & edge_dim) must be supplied if one were to train on edge types'

        if torch.is_tensor(feats):
            feats = {'0': feats[..., None]}

        if torch.is_tensor(global_feats):
            global_feats = {'0': global_feats[..., None]}

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
                adj_indices = adj_indices.masked_fill(next_degree_mask, degree)
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

        indices = repeat(torch.arange(n, device = device), 'j -> b i j', b = b, i = n)
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

        modified_rel_dist = rel_dist.clone()
        max_value = get_max_value(modified_rel_dist) # for masking out nodes from being considered as neighbors

        # neighbors

        if exists(neighbor_mask):
            neighbor_mask = remove_self(neighbor_mask)

            max_neighbors = neighbor_mask.sum(dim = -1).max().item()
            if max_neighbors > neighbors:
                print(f'neighbor_mask shows maximum number of neighbors as {max_neighbors} but specified number of neighbors is {neighbors}')

            modified_rel_dist = modified_rel_dist.masked_fill(~neighbor_mask, max_value)

        # use sparse neighbor mask to assign priority of bonded

        if exists(sparse_neighbor_mask):
            modified_rel_dist = modified_rel_dist.masked_fill(sparse_neighbor_mask, 0.)

        # mask out future nodes to high distance if causal turned on

        if self.causal:
            causal_mask = torch.ones(n, n - 1, device = device).triu().bool()
            modified_rel_dist = modified_rel_dist.masked_fill(causal_mask[None, ...], max_value)

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

        # calculate rotary pos emb

        rotary_pos_emb = None
        rotary_query_pos_emb = None
        rotary_key_pos_emb = None

        if self.rotary_position:
            seq = torch.arange(n, device = device)
            seq_pos_emb = self.rotary_pos_emb(seq)
            self_indices = torch.arange(neighbor_indices.shape[1], device = device)
            self_indices = repeat(self_indices, 'i -> b i ()', b = b)
            neighbor_indices_with_self = torch.cat((self_indices, neighbor_indices), dim = 2)
            pos_emb = batched_index_select(seq_pos_emb, neighbor_indices_with_self, dim = 0)

            rotary_key_pos_emb = pos_emb
            rotary_query_pos_emb = repeat(seq_pos_emb, 'n d -> b n d', b = b)

        if self.rotary_rel_dist:
            neighbor_rel_dist_with_self = F.pad(neighbor_rel_dist, (1, 0), value = 0) * 1e2
            rel_dist_pos_emb = self.rotary_pos_emb(neighbor_rel_dist_with_self)
            rotary_key_pos_emb = safe_cat(rotary_key_pos_emb, rel_dist_pos_emb, dim = -1)

            query_dist = torch.zeros(n, device = device)
            query_pos_emb = self.rotary_pos_emb(query_dist)
            query_pos_emb = repeat(query_pos_emb, 'n d -> b n d', b = b)

            rotary_query_pos_emb = safe_cat(rotary_query_pos_emb, query_pos_emb, dim = -1)

        if exists(rotary_query_pos_emb) and exists(rotary_key_pos_emb):
            rotary_pos_emb = (rotary_query_pos_emb, rotary_key_pos_emb)

        # calculate basis

        basis = get_basis(neighbor_rel_pos, num_degrees - 1, differentiable = self.differentiable_coors)

        # main logic

        edge_info = (neighbor_indices, neighbor_mask, edges)
        x = feats

        # project in

        x = self.conv_in(x, edge_info, rel_dist = neighbor_rel_dist, basis = basis)

        # preconvolution layers

        for conv, nonlin in self.convs:
            x = nonlin(x)
            x = conv(x, edge_info, rel_dist = neighbor_rel_dist, basis = basis)

        # transformer layers

        x = self.net(x, edge_info = edge_info, rel_dist = neighbor_rel_dist, basis = basis, global_feats = global_feats, pos_emb = rotary_pos_emb, mask = _mask)

        # project out

        if exists(self.conv_out):
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
