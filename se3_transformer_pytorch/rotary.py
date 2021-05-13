import torch
from torch import nn, einsum
from einops import rearrange, repeat

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        freqs = t[..., None].float() * self.inv_freq[None, :]
        return repeat(freqs, '... d -> ... (d r)', r = 2)

def rotate_half(x):
    x = rearrange(x, '... (d j) m -> ... d j m', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -2)

def apply_rotary_pos_emb(t, freqs):
    rot_dim = freqs.shape[-2]
    t, t_pass = t[..., :rot_dim, :], t[..., rot_dim:, :]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -2)
