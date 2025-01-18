import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self,
                 in_features: int, 
                 out_features: int, 
                 grid_size = 5, 
                 spline_order = 3,
                 scale_noise = 0.1,
                 scale_base = 1.0,
                 scale_spline = 1.0,
                 enable_standalone_scale_spline = True,
                 activation_function = torch.nn.SiLU,
                 grid_eps = 0.02,
                 grid_range = [-1, 1], 
                 device=None, 
                 dtype=None,
                ):
        super(KAN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.activation_function = activation_function()
        self.grid_eps = grid_eps
        
        initial_v_grid = grid_range[0]
        final_v_grid = grid_range[1]
        sum_g_s = grid_size + spline_order

        self.w = nn.parameter.Parameter(torch.Tensor(out_features, in_features))
        self.spline_w = nn.parameter.Parameter(torch.Tensor(out_features, in_features, sum_g_s))

        m = (final_v_grid - initial_v_grid)/grid_size
        grid = (torch.arange(-spline_order, sum_g_s + 1) * m + initial_v_grid).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.reset_parameters()
    
    def reset_parameters(self):
        
        scale = math.sqrt(5) * self.scale_base
        torch.nn.init.kaiming_uniform_(self.w, a=scale)

        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1/2
                ) * self.scale_noise / self.grid_size
            )
            self.spline_w.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a = math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):

        x = x.unsqueeze(-1)

        grid: torch.Tensor = (self.grid)

        b1 = (x >= grid[:, :-1])
        b2 = (x < grid[:, 1:])
        b = (b1 & b2).to(x.dtype)

        for i in range(1, self.spline_order + 1):

            left_b = ((x - grid[:, : -(i+1)]) /
                      (grid[:, i:-1] - grid[:, : -(i+1)])) * b[:, :, :-1]
            
            right_b = ((grid[:, i+1 :] - x) /
                       (grid[:, i+1 :] - grid[:, 1:-i])) * b[:, :, 1:]
            
            b = left_b + right_b

            # b = ((x - grid[:, : -(i + 1)]) 
            #      / (grid[:, i:-1] - grid[:, : -(i + 1)]) 
            #      * b[:, :, :-1]) 
            # + (
            #     (grid[:, i + 1 :] - x) 
            #     / (grid[:, i + 1 :] - grid[:, 1:(-i)]) 
            #     * b[:, :, 1:])
        
        return b.contiguous()
    
    def curve(self, x: torch.Tensor, y: torch.Tensor):

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        #least squares problem solution
        res = (torch.linalg.lstsq(A, B).solution).permute(2, 0, 1).contiguous()

        return res
    
    @property
    def scaled_spline_weight(self):
        
        if self.enable_standalone_scale_spline:
            return self.spline_w * self.spline_scaler.unsqueeze(-1)
        else:
            return self.spline_w
    
    def forward(self, x: torch.Tensor):

        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.activation_function(x), self.w)

        spline_basis = self.b_splines(x)
        batch_size = x.size(0)

        spline_basis_reshaped = spline_basis.view(batch_size, -1)
        spline_weights_reshaped = self.scaled_spline_weight.view(self.out_features, -1)

        spline_output = F.linear(
            spline_basis_reshaped, spline_weights_reshaped
        )
        # spline_output = F.linear(
        #     self.b_splines(x).view(x.size(0), -1),
        #     self.scaled_spline_weight.view(self.out_features, -1),
        # )
        
        output = (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)

        return output
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin = 0.01):
        # called only if update_grid = True, can skip for now
        pass

    def regularization_loss(self, regularize_activation = 1.0, regularize_entropy = 1.0):
        # maybe we can simply do L1 regularization, we don't need
        # this implementation
        pass
    
class KAN_NET(nn.Module):
    def __init__(self,
                 layers,
                 grid_size=5,
                 spline_order=3,
                 scale_noise=0.1,
                 scale_base=1.0,
                 scale_spline=1.0,
                 activation_function=torch.nn.SiLU,
                 hard_constraint_fn = None,
                 grid_eps=0.02,
                 grid_range=[-1, 1],
                ):
        
        super(KAN_NET, self).__init__()

        self.activation_function = activation_function
        self.hard_constraint_fn = hard_constraint_fn
        self.layers = nn.ModuleList()

        for i in range(len(layers)-1):
            self.layers.append(
                KAN(
                    layers[i],
                    layers[i+1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    activation_function=activation_function,
                    grid_eps=grid_eps,
                    grid_range=grid_range,  
                )
            )
        
    def forward(self, x: torch.Tensor, update_grid = False):
        orig_x = x
        for layer in self.layers:
            if update_grid:
                # keep this false for now
                layer.update_grid(x)
            x = layer(x)
        
        if self.hard_constraint_fn is not None:
            x = self.hard_constraint_fn(orig_x, x)
        
        return x
        
    # TO CHECK IF NEEDED
    # maybe we can simply do L1 regularization
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
