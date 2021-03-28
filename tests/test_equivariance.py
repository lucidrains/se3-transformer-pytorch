import torch
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot
from se3_transformer_pytorch.utils import torch_default_dtype

def test_transformer():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        num_degrees = 2,
        num_neighbors = 4,
        valid_radius = 10
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    out = model(feats, coors, mask, return_type = 0)
    assert out.shape == (1, 32, 64), 'output must be of the right shape'

def test_transformer_with_edges():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        num_degrees = 2,
        num_neighbors = 4,
        edge_dim = 4,
        num_edge_tokens = 4
    )

    feats = torch.randn(1, 32, 64)
    edges = torch.randint(0, 4, (1, 32))
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    out = model(feats, coors, mask, edges = edges, return_type = 0)
    assert out.shape == (1, 32, 64), 'output must be of the right shape'

def test_equivariance():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_neighbors = 4,
        num_degrees = 2,
        output_degrees = 2,
        fourier_encode_dist = True
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(15, 0, 45)
    out1 = model(feats, coors @ R, mask, return_type = 1)
    out2 = model(feats, coors, mask, return_type = 1) @ R

    diff = (out1 - out2).max()
    assert diff < 1e-4, 'is not equivariant'

@torch_default_dtype(torch.float64)
def test_equivariance_only_sparse_neighbors():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_degrees = 2,
        output_degrees = 2,
        num_neighbors = 0,
        attend_sparse_neighbors = True,
        num_adj_degrees = 2,
        adj_dim = 4
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    seq = torch.arange(32)
    adj_mat = (seq[:, None] >= (seq[None, :] - 1)) & (seq[:, None] <= (seq[None, :] + 1))

    R   = rot(15, 0, 45)
    out1 = model(feats, coors @ R, mask, adj_mat = adj_mat, return_type = 1)
    out2 = model(feats, coors, mask, adj_mat = adj_mat, return_type = 1) @ R

    diff = (out1 - out2).max()
    assert diff < 1e-4, 'is not equivariant'

def test_equivariance_with_reversible_network():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_neighbors = 4,
        num_degrees = 2,
        output_degrees = 2,
        reversible = True
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(15, 0, 45)
    out1 = model(feats, coors @ R, mask, return_type = 1)
    out2 = model(feats, coors, mask, return_type = 1) @ R

    diff = (out1 - out2).max()
    assert diff < 1e-4, 'is not equivariant'

def test_equivariance_with_type_one_input():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_neighbors = 4,
        num_degrees = 2,
        input_degrees = 2,
        output_degrees = 2
    )

    atom_features = torch.randn(1, 32, 64, 1)
    pred_coors = torch.randn(1, 32, 64, 3)

    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(15, 0, 45)
    out1 = model({'0': atom_features, '1': pred_coors @ R}, coors @ R, mask, return_type = 1)
    out2 = model({'0': atom_features, '1': pred_coors}, coors, mask, return_type = 1) @ R

    diff = (out1 - out2).max()
    assert diff < 1e-4, 'is not equivariant'
