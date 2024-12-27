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
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None, m = 0.5, sd = 0.01):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        shape = (out_features, in_features)
        """""
        - Delete the storing of w, s and v which is redundant, this allow to save memeory, 
        - Change the initialization of s from in_feature to out_feature dimention because s represent 
        the scale factor for each neuron in the layer so it should be aligned with the number of output, also considering that
        in the equation 2.3, of the paper added on the readme, about the random weight factorization s should be a 1D vector 
        and it must be represented as diag(s^l) where l are the index of the input feature. So at the end if the rows of v are the 
        output neurons weight, s should moltiply each rows of v and for this reason should be of dimention [out_feature, 1] which 
        is the same thing, if we follow the paper strictly, of transform s as a diagonal matrix with zeros everywhere exept in 
        the diagonal, but I implemented the first approach (less multiplication).
        - Solved the problem of loss = NaN by inizializating b with zeros
        - Adding normalization of v to ensure stability. Also the paper enforce the separation between s and v, the first
        to control the magnitude and the second the directionality of the vector, so v should be normilized
        otherwise s is not fully control the magnitude
        - I found in the paper that they used different value of m and sd, using m = 1 for the Normal distribution mean 
        when passing on the exponential (the expected value) gives e^1 = 2.71 so initially v are smaller than the initial w so 
        larger update becouse the scaling factors are close to 2.71. So what I found is that the grater is m the larger is the 
        scaling factor and the update during training. It is interesting try different m around 1, I see in the paper (table 4) 
        that m = 2 is used for image regression which needs to scale the pixel more aggressively maybe? and for the Navier-Stokes
        use m = 0.5, maybe needs more precision initial condition and a less aggressive update approach.
        Of course this should be confirmed and It is interesting put this reasonings in the report aslo trying training with different 
        m values.

        """""


        w = nn.init.xavier_normal_(torch.empty(shape, **factory_kwargs)) #we give a uninitialized tensor and use Xavier normal distribution to fill in_features (Glorot initialization)
        s = torch.randn(out_features)*sd + m #tensor with random numbers from a normal distribution with mean 'm' and standard deviation 'sd'
        s = torch.exp(s)
        v = w/s[:, None]
        with torch.no_grad():
            v /= v.norm(dim=1, keepdim=True)

        self.s = nn.parameter.Parameter(s)
        self.v = nn.parameter.Parameter(v)

        self.b = nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs))
        nn.init.zeros_(self.b)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        k = self.s[:, None] * self.v
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
