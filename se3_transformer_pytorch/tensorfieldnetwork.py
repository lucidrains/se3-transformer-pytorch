from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn, einsum

import numpy as np
from einops import rearrange, repeat

# constants

EPSILON = 1e-8

# helpers

# shifted soft plus nonlin
def ssp(x):
    return torch.log(0.5 * torch.exp(x) + 0.5)

def safe_div(num, den, eps = 1e-8):
    return num.div(den + eps)

def unit_vectors(v, dim = -1):
    return safe_div(v, v.norm(dim = dim, keepdims=True))

def get_eijk():
    """
    Constant Levi-Civita tensor

    Returns:
        tf.Tensor of shape [3, 3, 3]
    """
    eijk_ = np.zeros((3, 3, 3))
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return torch.from_numpy(eijk_).float()

class RotationEquivariantNonlinearity(nn.Module):
    def __init__(self, dim, nonlin = ssp):
        super().__init__()
        self.biases = nn.Parameter(torch.zeros(dim))
        self.nonlin = nonlin

    def forward(self, x):
        *_, representation_index, channels = x.shape

        if representation_index ==1:
            return self.nonlin(x)

        norm = x.norm(dim = -1)
        biases = rearrange(biases, 'c -> () () c')
        nonlin_out = self.nonlin(norm + biases)
        factor = safe_div(nonlin_out, norm)
        return x * factor[..., None]

class R(nn.Module):
    def __init__(
        self,
        nonlin = nn.ReLU(),
        hidden_dim = None,
        output_dim = 1
    ):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        self.w2 = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(output_dim))

        self.nonlin = nonlin

    def forward(self, x):
        x = self.b1 + torch.tensordot(x, self.w1, ((2,), (1,)))
        x = self.nonlin(x)
        x = self.b2 = + torch.tensordot(x, self.w2, ((2,), (1,)))
        return x

# layers

def Y_2(rij, eps = EPSILON):
    # rij : [N, N, 3]
    # x, y, z : [N, N]
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]
    r2 = (rij ** 2).sum(dim=-1).clamp_(min = eps)
    # return : [N, N, 5]
    output = torch.stack([ x * y / r2,
                           y * z / r2,
                           (-(x ** 2) - (y ** 2) + 2. * (z ** 2)) / (2 * sqrt(3) * r2),
                           z * x / r2,
                           ((x ** 2) - (y ** 2)) / (2. * r2)
                         ], dim = -1)
    return output


class F_0(nn.Module):
    def __init__(self, nonlin=nn.ReLU(), hidden_dim=None, output_dim=1):
        super().__init__()
        self.R = R(nonlin = nonlin, hidden_dim = hidden_dim, output_dim = output_dim)

    def forward(self, x):
        return self.R(x)[..., None]

class F_1(nn.Module):
    def __init__(self, nonlin=nn.ReLU(), hidden_dim=None, output_dim=1):
        super().__init__()
        self.R = R(nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, rij):
        radial = self.R(rij)
        dij = rij.norm(dim = -1)[..., None]
        radial.masked_fill_(dij < EPSILON, 0)
        return unit_vectors(rij)[..., None, :] * radial[..., None]

class F_2(nn.Module):
    def __init__(self, nonlin=nn.ReLU(), hidden_dim=None, output_dim=1):
        super().__init__()
        self.R = R(nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, rij):
        radial = self.R(rij)
        dij = rij.norm(dim = -1)[..., None]
        radial.masked_fill_(dij < EPSILON, 0)
        return Y_R(rij)[..., None, :] * masked_radial[..., None]

CLEBSH_GORDAN_EINSUM = 'i j k, a b f j, b f k -> a f i'

class Filter0(nn.Module):
    def __init__(self, nonlin=nn.ReLU(), hidden_dim=None, output_dim=1):
        super().__init__()
        self.F_0 = F_0(nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, layer_inputs, rbf_inputs):
        F_0_out = self.F_0(rbf_inputs)
        input_dim = layer_input.shape[-1]
        cg = torch.eye(input_dim)[..., None, :]
        return einsum(CLEBSH_GORDAN_EINSUM, cg, F_0_out, layer_input)

class Filter1Output0(nn.Module):
    def __init__(self, nonlin=nn.ReLU(), hidden_dim=None, output_dim=1):
        super().__init__()
        self.F_1(nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, layer_inputs, rbf_inputs, rij):
        assert layer_inputs.shape[-1] == 3, 'layer input must have dimension of 3'
        F_1_out = self.F_1(rbf_inputs, rij)
        cg = torch.eye(3)[None, ...]
        return einsum(CLEBSH_GORDAN_EINSUM, cg, F_1_out, layer_input)

class Filter1Output1(nn.Module):
    def __init__(self, nonlin=nn.ReLU(), hidden_dim=None, output_dim=1):
        super().__init__()
        self.F_1(nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, layer_inputs, rbf_inputs, rij):
        dim = layer_inputs.shape[-1]
        if dim == 1:
            cg = torch.eye(3)[..., None]
            return einsum(CLEBSH_GORDAN_EINSUM, cg, F_1_out, layer_input)
        elif dim == 3:
            return einsum(CLEBSH_GORDAN_EINSUM, get_eijk(), F_1_out, layer_input)
        else:
            raise NotImplementedError('other Ls not implemented')

class Filter2Output2(nn.Module):
    def __init__(self, nonlin=nn.ReLU(), hidden_dim=None, output_dim=1):
        super().__init__()
        self.F_2 = F_2(nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, layer_inputs, rbf_inputs, rij):
        F_2_out = self.F_2(rbf_inputs, rij)
        dim = layer_inputs.shape[-1]
        if dim == 1:
            cg = torch.eye(5)[..., None]
            einsum(CLEBSH_GORDAN_EINSUM, cg, F_2_out, layer_input)
        else:
            raise NotImplementedError('other Ls not implemented')

class SelfInteractionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias = False):
        super().__init__()
        self.w_si = nn.Parameter(torch.randn(output_dim, input_dim))
        self.b_si = nn.Parameter(torch.zeros(output_dim)) if bias else None

    def forward(self, inputs):
        out = einsum('a f i, g f -> a i g', inputs, self.w_si)
        out = rearrange(out, 'a i g -> a g i')
        if self.b_si is not None:
            out = out + self.b_si
        return out

class Conv(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.filter_0 = Filter0(output_dim = output_dim)
        self.filter_1_output_0 = Filter1Output0(output_dim = output_dim)
        self.filter_1_filter_1 = Filter1Output1(output_dim = output_dim)

    def forward(self, input_tensor_list, rbf, unit_vectors):
        output_tensor_list = {0: [], 1: []}
        for key, value in input_tensor_list.items():
            for i, tensor in enumerate(value):
                tensor_out = self.filter_0(tensor, rbf)
                m = 0 if tensor_out.shape[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)

                if key == 1:
                    tensor_out = self.filter_1_output_0(tensor, rbf, unit_vectors)
                    m = 0 if tensor_out.shape[-1] == 1 else 1
                    output_tensor_list[m].append(tensor_out)

                if key == 0 or key ==1:
                    tensor_out = self.filter_1_filter_1(tensor, rbf, unit_vectors)
                    m = 0 if tensor_out.shape[-1] == 1 else 1
                    output_tensor_list[m].append(tensor_out)

        return output_tensor_list

class SelfInteraction(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.self_interaction_with_biases = SelfInteractionLayer(output_dim = output_dim, bias = True)
        self.self_interaction_without_biases = SelfInteractionLayer(output_dim = output_dim, bias = False)

    def forward(self, x):
        output_tensor_list = {0: [], 1: []}
        for key, value in input_tensor_list.item():
            for i, tensor in enumerate(value):
                if key == 0:
                    tensor_out = self.self_interaction_with_biases(tensor)
                else:
                    tensor_out = self.self_interaction_without_biases(tensor)
                m = 0 if tensor_out.shape[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)

        return output_tensor_list

class Nonlinearity(nn.Module):
    def __init__(self, output_dim, nonlin = nn.ELU()):
        super().__init__()
        self.nonlin = nonlin
        self.rotation_equivariant_nonlin = self.RotationEquivariantNonlinearity(nonlin = nonlin, output_dim = output_dim)

    def forward(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key, value in input_tensor_list.items():
            for i, tensor in enumerate(value):
                tensor_out = self.rotation_equivariant_nonlin(tensor)
                m = 0 if tensor_out.shape[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)
        return output_tensor_list

class Concatenation(nn.Module):
    def forward(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key, value in input_tensor_list.items():
            output_tensor_list[key].append(torch.cat(values, dim = -2))
        return output_tensor_list

class TFN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
