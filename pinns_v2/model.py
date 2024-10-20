import torch
import torch.nn as nn
import numpy as np
from pinns_v2.rff import GaussianEncoding
from collections import OrderedDict
import math


class ModifiedMLP(nn.Module):
    def __init__(self, layers, activation_function, hard_constraint_fn=None, p_dropout=0.2, encoding=None) -> None:
        super(ModifiedMLP, self).__init__()

        self.layers = layers
        self.activation = activation_function
        self.encoding = encoding
        if encoding != None:
            encoding.setup(self)
        
        self.U = torch.nn.Sequential(nn.Linear(self.layers[0], self.layers[1]), self.activation())
        self.V = torch.nn.Sequential(nn.Linear(self.layers[0], self.layers[1]), self.activation())

        layer_list = nn.ModuleList()        
        for i in range(0, len(self.layers)-2):
            layer_list.append(
                nn.Linear(layers[i], layers[i+1])
            )
            layer_list.append(self.activation())
            layer_list.append(Transformer())
            layer_list.append(nn.Dropout(p = p_dropout))
        self.hidden_layer = layer_list
        self.output_layer = nn.Linear(self.layers[-2], self.layers[-1])

        self.hard_constraint_fn = hard_constraint_fn
        

    def forward(self, x):
        orig_x = x
        if self.encoding != None:
            x = self.encoding(x)

        U = self.U(orig_x)
        V = self.V(orig_x)

        output = x
        for i in range(0, len(self.hidden_layer), 4):
            output = self.hidden_layer[i](output) #Linear
            output = self.hidden_layer[i+1](output) #Activation
            output = self.hidden_layer[i+2](output, U, V) #Transformer
            output = self.hidden_layer[i+3](output) #Dropout
        output = self.output_layer(output)

        if self.hard_constraint_fn != None:
            output = self.hard_constraint_fn(orig_x, output)

        return output

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
        self.b = nn.parameter.Parameter(torch.empty(out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        k = self.s*self.v
        return nn.functional.linear(input, k, self.bias)
    
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
        
class FactorizedModifiedLinear(RWF):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
    
    def forward(self, x, U , V):
        return torch.nn.functional.linear(torch.multiply(x, U) + torch.multiply((1-x), V), self.s*self.v, self.bias)

class Sin(nn.Module):
  def __init__(self):
    super(Sin, self).__init__()

  def forward(self, x):
    return torch.sin(x)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
    
    def forward(self, x, U, V):
        return torch.multiply(x, U) + torch.multiply(1-x, V)
        #return torch.nn.functional.linear(torch.multiply(x, U) + torch.multiply((1-x), V), self.weight, self.bias)
