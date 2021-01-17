import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from src.models.utils import get_activation
from typing import List


class Model(nn.Module):
    ''' Simple parametrized network '''
    
    def __init__(self, in_features: int, out_size: int, layers: List[int],
                 activation_name: str, activation_out: str):

        super().__init__()

        activation_method = get_activation(activation_name)
        activation_out = get_activation(activation_out)
        layerlist = []
        n_in = in_features

        for layer in layers:
            layerlist.append(nn.Linear(n_in, layer))
            layerlist.append(activation_method())
            n_in = layer
        layerlist.append(nn.Linear(layers[-1], out_size))
        layerlist.append(activation_out())

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class NeuronBud(nn.Module):
    ''' Static Bud model with two 3x3 hidden layers and given output size '''
    def __init__(self, size_out: int, activation_name: str):
        super().__init__()

        self.activation_method = get_activation(activation_name)
        self.weight = torch.ones(3, 1) * 1/3

        self.layers = nn.Sequential(
            nn.Linear(3, 3),
            self.activation_method(),
            nn.Linear(3, 3),
            self.activation_method(),
            nn.Linear(3, size_out),
            self.activation_method()
        )

    def forward(self, x):
        x = (self.weight * x).view(3)
        return self.layers(x)

class BuddingLayer(nn.Module):
    ''' Budding layer that stores and activates the buds for saturated neurons. '''
    def __init__(self, size_in: int, size_out: int, bias: bool = True):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.buds = nn.ModuleDict()
        self.weight = nn.Parameter(Tensor(size_out, size_in))
        if bias:
            self.bias = nn.Parameter(Tensor(size_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Weights and biases initialization same as in basic nn.Linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: Tensor, saturated: Tensor) -> Tensor:
        ''' saturated tensor should be provided from the previous layer

        '''
        if saturated.any().item() > 0:
            x_b = x * saturated.float()
            saturated_neurons = torch.nonzero(x_b)
            
            # Add new NeuronBuds if the number of buds has changed
            if saturated_neurons.size()[0] != len(self.buds):
                u_neurons = torch.unique(saturated_neurons).tolist()
                u_buds = [int(key) for key in self.buds.keys()]
                for new_bud in list(set(u_neurons) - set(u_buds)):
                    self.buds[str(new_bud)] = NeuronBud(self.size_out, 'relu')

            # Forward pass through buds
            u = [self.buds[key](x[int(key)]) for key in self.buds]
            u = torch.stack(u, dim=1)
            u = torch.sum(u, dim=1)

            # Apply linear transformation for non-saturated neurons
            x = x * (~saturated).float()
            
        x = F.linear(x, self.weight, self.bias)

        return x + u
