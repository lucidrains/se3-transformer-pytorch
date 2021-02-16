import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# helpers

def map_values(fn, x):
    out = {}
    for (k, v) in x.items():
        out[k] = fn(v)
    return out

def dict_chunk(x, chunks, dim):
    out1 = {}
    out2 = {}
    for (k, v) in x.items():
        c1, c2 = v.chunk(chunks, dim = dim)
        out1[k] = c1
        out2[k] = c2
    return out1, out2

def dict_sum(x, y):
    out = {}
    for k in x.keys():
        out[k] = x[k] + y[k]
    return out

def dict_subtract(x, y):
    out = {}
    for k in x.keys():
        out[k] = x[k] - y[k]
    return out

def dict_cat(x, y, dim):
    out = {}
    for k, v1 in x.items():
        v2 = y[k]
        out[k] = torch.cat((v1, v2), dim = dim)
    return out

def dict_set_(x, key, value):
    for k, v in x.items():
        setattr(v, key, value)

def dict_backwards_(outputs, grad_tensors):
    for k, v in outputs.items():
        torch.autograd.backward(v, grad_tensors[k], retain_graph = True)

def dict_del_(x):
    for k, v in x.items():
        del v
    del x

def values(d):
    return [v for _, v in d.items()]

# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, **kwargs):
        training = self.training
        x1, x2 = dict_chunk(x, 2, dim = -1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = dict_sum(x1, self.f(x2, record_rng = training, **kwargs))
            y2 = dict_sum(x2, self.g(y1, record_rng = training))

        return dict_cat(y1, y2, dim = -1)

    def backward_pass(self, y, dy, **kwargs):
        y1, y2 = dict_chunk(y, 2, dim = -1)
        dict_del_(y)

        dy1, dy2 = dict_chunk(dy, 2, dim = -1)
        dict_del_(dy)

        with torch.enable_grad():
            dict_set_(y1, 'requires_grad', True)
            gy1 = self.g(y1, set_rng = True)
            dict_backwards_(gy1, dy2)

        with torch.no_grad():
            x2 = dict_subtract(y2, gy1)
            dict_del_(y2)
            dict_del_(gy1)

            dx1 = dict_sum(dy1, map_values(lambda t: t.grad, y1))
            dict_del_(dy1)
            dict_set_(y1, 'grad', None)

        with torch.enable_grad():
            dict_set_(x2, 'requires_grad', True)
            fx2 = self.f(x2, set_rng = True, **kwargs)
            dict_backwards_(fx2, dx1)

        with torch.no_grad():
            x1 = dict_subtract(y1, fx2)
            dict_del_(y1)
            dict_del_(fx2)

            dx2 = dict_sum(dy2, map_values(lambda t: t.grad, x2))
            dict_del_(dy2)
            dict_set_(x2, 'grad', None)

            x2 = map_values(lambda t: t.detach(), x2)

            x = dict_cat(x1, x2, dim = -1)
            dx = dict_cat(dx1, dx2, dim = -1)

        return x, dx

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        input_keys = kwargs.pop('input_keys')
        split_dims = kwargs.pop('split_dims')
        input_values = x.split(split_dims, dim = -1)
        x = dict(zip(input_keys, input_values))

        ctx.kwargs = kwargs
        ctx.split_dims = split_dims
        ctx.input_keys = input_keys

        for block in blocks:
            x = block(x, **kwargs)

        ctx.y = map_values(lambda t: t.detach(), x)
        ctx.blocks = blocks

        x = torch.cat(values(x), dim = -1)
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        input_keys = ctx.input_keys
        split_dims = ctx.split_dims

        dy = dy.split(split_dims, dim = -1)
        dy = dict(zip(input_keys, dy))

        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)

        dy = torch.cat(values(dy), dim = -1)
        return dy, None, None

class SequentialSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, **kwargs):
        for (attn, ff) in self.blocks:
            x = attn(x, **kwargs)
            x = ff(x)
        return x

class ReversibleSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

    def forward(self, x, **kwargs):
        blocks = self.blocks

        x = map_values(lambda t: torch.cat((t, t), dim = -1), x)

        input_keys = x.keys()
        split_dims = tuple(map(lambda t: t.shape[-1], x.values()))
        block_kwargs = {'input_keys': input_keys, 'split_dims': split_dims, **kwargs}

        x = torch.cat(values(x), dim = -1)

        x = _ReversibleFunction.apply(x, blocks, block_kwargs)

        x = dict(zip(input_keys, x.split(split_dims, dim = -1)))
        x = map_values(lambda t: torch.stack(t.chunk(2, dim = -1)).mean(dim = 0), x)
        return x
