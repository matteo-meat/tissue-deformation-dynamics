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
    def __init__(self, in_features: int, out_features: int, grid_size: int, spline_order: int, grid_range, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        initial_v_grid = grid_range[0]
        final_v_grid = grid_range[1]
        sum_g_s = grid_size + spline_order

        self.w = nn.parameter.Parameter(torch.Tensor(out_features, in_features))
        self.spline_w = nn.parameter.Parameter(torch.Tensor(out_features, in_features, sum_g_s))

        m = (final_v_grid - initial_v_grid)/grid_size
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
    
class KAN_NET(nn.Module):
    def __init__(self, layers_hidden, grid_size=5,
                    spline_order=3,
                    scale_noise=0.1,
                    scale_base=1.0,
                    scale_spline=1.0,
                    base_activation=torch.nn.SiLU,
                    grid_eps=0.02,
                    grid_range=[-1, 1],):
        super(KAN_NET, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KAN(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,  
                )
            )
        
        def forward(self, x: torch.Tensor, update_grid = False):
            for layer in self.layers:
                if update_grid:
                    layer.update_grid(x)
                x = layer(x)
            return x
        
        # TO CHECK IF NEEDED
        def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
            return sum(
                layer.regularization_loss(regularize_activation, regularize_entropy)
                for layer in self.layers
            )
