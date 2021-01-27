## SE3 Transformer - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2006.10503">SE3-Transformers</a> for Equivariant Self-Attention, in Pytorch. May be needed for replicating Alphafold2 results and other drug discovery applications.

## Install

```bash
$ pip install se3-transformer-pytorch
```

## Usage

```python
import torch
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer

model = SE3Transformer(
    dim = 512,
    heads = 8,
    depth = 6,
    dim_head = 64,
    num_degrees = 4
)

feats = torch.randn(1, 1024, 512)
coors = torch.randn(1, 1024, 3)
mask  = torch.ones(1, 1024).bool()

out = model(feats, coors, mask) # (1, 1024, 512)
```

Example usage for Alphafold2

```python
import torch
from se3_transformer_pytorch import SE3Transformer

model = SE3Transformer(
    dim = 32,
    depth = 2,
    input_degrees = 2,
    num_degrees = 2,
    output_degrees = 2
)

atom_feats = torch.randn(2, 32, 32, 1)
pred_coors = torch.randn(2, 32, 32, 3)

# atom features are type 0, predicted coordinates are type 1
features = {'0': atom_feats, '1': pred_coors}
coors = torch.randn(2, 32, 3)
mask  = torch.ones(2, 32).bool()

refinement = model(features, coors, mask, return_type = 1) # (2, 32, 32, 3)
```

## Testing

```bash
$ python setup.py pytest
```

## Credit

This library is largely a port of <a href="https://github.com/FabianFuchsML/se3-transformer-public">Fabian's official repository</a>, but without the DGL library.

## Citations

```bibtex
@misc{fuchs2020se3transformers,
    title   = {SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks}, 
    author  = {Fabian B. Fuchs and Daniel E. Worrall and Volker Fischer and Max Welling},
    year    = {2020},
    eprint  = {2006.10503},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
