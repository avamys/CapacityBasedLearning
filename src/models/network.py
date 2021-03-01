import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from src.models.utils import get_activation
from typing import List
from collections import deque


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
        self.weight = torch.ones(1, 3) * 1/3

        self.layers = nn.Sequential(
            nn.Linear(3, 3),
            self.activation_method(),
            nn.Linear(3, 3),
            self.activation_method(),
            nn.Linear(3, size_out),
            self.activation_method()
        )

    def forward(self, x):
        x = (self.weight * x)
        return self.layers(x)

class BuddingLayer(nn.Module):
    ''' Budding layer that stores and activates the buds for saturated 
        neurons.
    '''
    def __init__(self, size_in: int, size_out: int, window_size: int, 
                 compute_lipschitz: bool = True, bias: bool = True):
        super().__init__()

        self.window_size = window_size
        self.weights_window = deque(maxlen=self.window_size)
        self.lipshitz_constants = deque(maxlen=self.window_size)
        self.saturated_neurons = torch.zeros(size_in, dtype=torch.bool)
        
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
        ''' Weights and biases initialization same as in basic nn.Linear '''
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def update_cache(self):
        ''' Adds weights to the cache '''
        self.weights_window.append(self.state_dict()['weight'].clone().detach())

    def get_lipschitz_constant(self):
        ''' Compute lipshitz constant for any new epoch '''
        current_weights = self.state_dict()['weight']

        for epoch_id, cached_weights in enumerate(self.weights_window):
            dist_func = torch.abs(torch.sum(cached_weights, dim=1) - torch.sum(current_weights, dim=1))
            dist_points = 1 if self.window_size - epoch_id == 1 else (self.window_size - epoch_id) - 1

            self.lipshitz_constants.append(dist_func / dist_points)

        return torch.max(torch.stack(list(self.lipshitz_constants), dim=1), dim=1)
    
    def forward(self, x: Tensor, saturated: Tensor) -> Tensor:
        ''' saturated tensor should be provided from the previous layer

        '''
        u = 0
        sat = torch.logical_or(self.saturated_neurons, saturated)
        self.update_cache()

        if sat.any().item() > 0:
            # Add new NeuronBuds if the number of buds has changed
            if torch.sum(sat) > torch.sum(self.saturated_neurons):
                u_neurons = torch.nonzero(sat).view(-1).tolist()
                u_buds = [int(key) for key in self.buds.keys()]
                
                for new_bud in list(set(u_neurons) - set(u_buds)):
                    self.buds[str(new_bud)] = NeuronBud(self.size_out, 'relu') # To parametrize

                self.saturated_neurons = sat

            # Forward pass through buds
            u = [self.buds[key](x[:, int(key)].view(-1,1)) for key in self.buds]
            u = torch.stack(u, dim=1)
            u = torch.sum(u, dim=1)

            # Apply linear transformation for non-saturated neurons
            x = x * (~self.saturated_neurons).float()
            
        x = F.linear(x, self.weight, self.bias)

        return x + u

class CapacityNet(nn.Module):
    def __init__(self, in_features: int, out_size: int, layers: List[int],
                 activation_name: str, activation_out: str, window_size):
        super().__init__()

        self.window_size = window_size

        # activation_method = get_activation(activation_name)
        # activation_out = get_activation(activation_out)
        n_in = in_features

        self.bl1 = BuddingLayer(n_in, 10, self.window_size)
        self.bl2 = BuddingLayer(10, 10, self.window_size)
        self.blout = BuddingLayer(10, out_size, self.window_size, False)

        self.sat0 = torch.zeros(n_in, dtype=torch.bool)

    def forward(self, x) -> Tensor:
        x = F.relu(self.bl1(x, self.sat0))
        saturation = self.bl1.get_lipschitz_constant()[0] > 0.1
        x = F.relu(self.bl2(x, saturation))
        saturation = self.bl2.get_lipschitz_constant()[0] > 0.01
        x = self.blout(x, saturation)

        return x
