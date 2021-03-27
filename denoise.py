import torch
import torch.nn.functional as F
from torch.optim import Adam

from einops import rearrange, repeat

import sidechainnet as scn
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer

torch.set_default_dtype(torch.float64)

BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 16

def cycle(loader, len_thres = 500):
    while True:
        for data in loader:
            if data.seqs.shape[1] > len_thres:
                continue
            yield data

transformer = SE3Transformer(
    num_tokens = 24,
    dim = 8,
    dim_head = 8,
    heads = 2,
    depth = 2,
    attend_self = True,
    input_degrees = 1,
    output_degrees = 2,
    reduce_dim_out = True,
    differentiable_coors = True,
    num_neighbors = 0,
    attend_sparse_neighbors = True
)

data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False
)

dl = cycle(data['train'])
optim = Adam(transformer.parameters(), lr=1e-4)
transformer = transformer.cuda()

for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        seqs = seqs.cuda().argmax(dim = -1)
        coords = coords.cuda().type(torch.float64)
        masks = masks.cuda().bool()

        l = seqs.shape[1]
        coords = rearrange(coords, 'b (l s) c -> b l s c', s=14)

        # keep backbone coordinates

        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        seq = repeat(seqs, 'b n -> b (n c)', c = 3)
        masks = repeat(masks, 'b n -> b (n c)', c = 3)

        noised_coords = coords + torch.randn_like(coords).cuda()

        i = torch.arange(seq.shape[-1], device = seqs.device)
        adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))
        adj_mat = (adj_mat.float() @ adj_mat.float()) > 0  # get second degree neighbors

        out = transformer(
            seq,
            noised_coords,
            mask = masks,
            adj_mat = adj_mat,
            return_type = 1
        )

        denoised_coords = noised_coords + out

        loss = F.mse_loss(denoised_coords[masks], coords[masks]) 
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print('loss:', loss.item())
    optim.step()
    optim.zero_grad()
