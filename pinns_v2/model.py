import torch
import torch.nn as nn
import numpy as np
from pinns_v2.rff import GaussianEncoding
from collections import OrderedDict
import math

class MLP(nn.Module):
    def __init__(self, layers, activation_function, hard_constraint_fn=None, p_dropout=0.2, encoding=None) -> None:
        super(MLP, self).__init__()
        self.layers = layers
        self.activation = activation_function
        #self.encoding = encoding
        #if encoding != None:
            #encoding.setup(self)

        layer_list = list()        
        for i in range(len(self.layers)-2):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('dropout_%d' % i, nn.Dropout(p = p_dropout)))
        layer_list.append(('layer_%d' % (len(self.layers)-1), nn.Linear(self.layers[-2], self.layers[-1])))

        self.mlp = nn.Sequential(OrderedDict(layer_list))

        self.hard_constraint_fn = hard_constraint_fn

    def forward(self, x):
        orig_x = x
        #if self.encoding != None:
            #x = self.encoding(x)

        output = self.mlp(x)

        if self.hard_constraint_fn != None:
            output = self.hard_constraint_fn(orig_x, output)

        return output

class RWF(nn.Module): #Single layer
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None, m = 1.0, sd = 0.1):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        shape = (out_features, in_features)

        self.w = nn.init.xavier_normal_(torch.empty(shape, **factory_kwargs)) #we give a uninitialized tensor and use Xavier normal distribution to fill in_features (Glorot initialization)
        self.s = torch.randn(self.in_features)*sd + m #tensor with random numbers from a normal distribution with mean 'm' and standard deviation 'sd'
        self.s = torch.exp(self.s)
        self.v = self.w/self.s
        self.s = nn.parameter.Parameter(self.s)
        self.v = nn.parameter.Parameter(self.v)
        self.b = nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        k = self.s*self.v
        return nn.functional.linear(input, k, self.b)
    
class MLP_RWF(nn.Module):
    def __init__(self, layers, activation_function, hard_constraint_fn=None, p_dropout=0.2, encoding=None) -> None:
        super(MLP_RWF, self).__init__()
        self.layers = layers
        self.activation = activation_function
        #self.encoding = encoding
        #if encoding != None:
            #encoding.setup(self)

        layer_list = list()        
        for i in range(len(self.layers)-2):
            layer_list.append(
                ('layer_%d' % i, RWF(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('dropout_%d' % i, nn.Dropout(p = p_dropout)))
        layer_list.append(('layer_%d' % (len(self.layers)-1), RWF(self.layers[-2], self.layers[-1])))

        self.mlp = nn.Sequential(OrderedDict(layer_list))

        self.hard_constraint_fn = hard_constraint_fn

    def forward(self, x):
        orig_x = x
        #if self.encoding != None:
            #x = self.encoding(x)

        output = self.mlp(x)

        if self.hard_constraint_fn != None:
            output = self.hard_constraint_fn(orig_x, output)

        return output
    
class KAN(nn.Module): #Single layer
    def __init__(self, in_features: int, out_features: int, grid_size: int, spline_order: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        initial_v_grid = -1
        final_v_grid = 1
        sum_g_s = grid_size + spline_order

        self.w = nn.parameter.Parameter(torch.Tensor(out_features, in_features))
        self.spline_w = nn.parameter.Parameter(torch.Tensor(out_features, in_features, sum_g_s))

        m = (initial_v_grid - final_v_grid)/grid_size
        grid = (torch.arange(-spline_order, sum_g_s + 1) * m + initial_v_grid).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.resert_parameter()

    def spline(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        grid: torch.Tensor = (self.grid)
        b1 = (x >= grid[:, :-1])
        b2 = (x < grid[:, 1:])

        b = (b1 & b2).to(x.dtype)
        i = 1
        n = self.spline_order + 1
        for i in range(n):
            b = ((x - grid[:, : -(i + 1)]) 
                 / (grid[:, i:-1] - grid[:, : -(i + 1)]) 
                 * b[:, :, :-1]) 
            + (
                (grid[:, i + 1 :] - x) 
                / (grid[:, i + 1 :] - grid[:, 1:(-i)]) 
                * b[:, :, 1:])

        res = b.contiguous()
        
        return res
    
    def curve(self, x: torch.Tensor, y: torch.Tensor):
        k = self.spline(x)

        A = k.transpose(0, 1)
        B = y.transpose(0, 1)
        #least squares problem solution
        res = (torch.linalg.lstsq(A, B).solution).permute(2, 0, 1).contiguous()

        return res
        
class FactorizedModifiedLinear(RWF):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
    
    def forward(self, x, U , V):
        return torch.nn.functional.linear(torch.multiply(x, U) + torch.multiply((1-x), V), self.s*self.v, self.bias)
