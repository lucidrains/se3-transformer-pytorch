import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def safe_div(num, den, eps = 1e-8):
    res = num.div(den.clamp(min = eps))
    res.masked_fill_(den == 0, 0.)
    return res

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# swish activation fallback

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

# helper classes

class HtypesNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8, scale_init = 1e-2):
        super().__init__()
        self.eps = eps
        scale = torch.empty(1, 1, 1, dim, 1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * (norm * self.scale)

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

        self.htype_norms = nn.ModuleDict([])

        for degree, dim in fiber:
            if degree == 0:
                continue
            self.htype_norms[str(degree)] = HtypesNorm(dim)

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

        # type 0 features

        nodes = features['0']
        nodes = rearrange(nodes, '... () -> ...')

        # higher types (htype)

        htypes = list(filter(lambda t: t[0] != '0', features.items()))
        htype_degrees = list(map(lambda t: t[0], htypes))
        htype_dims = list(map(lambda t: t[1].shape[-2], htypes))

        # prepare mask

        if exists(mask):
            mask = batched_index_select(mask, neighbor_indices, dim = 1)

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

        return features

class EGnnNetwork(nn.Module):
    def __init__(
        self,
        *,
        fiber,
        depth,
        edge_dim = 0,
        hidden_dim = 32,
        coor_weights_clamp_value = None
    ):
        super().__init__()
        self.fiber = fiber
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EGNN(fiber = fiber, edge_dim = edge_dim, hidden_dim = hidden_dim, coor_weights_clamp_value = coor_weights_clamp_value))

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
        for layer in self.layers:
            features = layer(
                features,
                edge_info = edge_info,
                rel_dist = rel_dist,
                basis = basis,
                global_feats = global_feats,
                pos_emb = pos_emb,
                mask = mask,
                **kwargs
            )

        return features
