import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from src.models.utils import get_activation
from typing import List, Tuple
from collections import deque
from scipy.special import binom

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
        self.lipschitz_size = int(binom(self.window_size+1, 2))
        self.weights_window = deque(maxlen=self.window_size)
        self.lipshitz_constants = deque(maxlen=self.lipschitz_size)
        self.saturated_neurons = torch.zeros(size_in, dtype=torch.bool)
        
        self.size_in = size_in
        self.size_out = size_out
        self.buds = nn.ModuleDict()
        self.weight = nn.Parameter(Tensor(size_out, size_in))
        if bias:
            self.bias = nn.Parameter(Tensor(size_out))
        else:
            self.register_parameter('bias', None)
        self.__reset_parameters()

    def __reset_parameters(self) -> None:
        ''' Weights and biases initialization same as in basic nn.Linear '''
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def __update_cache(self) -> None:
        ''' Adds weights to the cache '''
        current_weights = self.state_dict()['weight']

        self.weights_window.append(current_weights.clone().detach())

    def __update_lipschitz_cache(self) -> None:
        ''' Adds lipschitz constants computed for a current epoch's weights 
            to the cache.
        '''
        current_weights = self.state_dict()['weight']

        for epoch_id, cached_weights in enumerate(self.weights_window):
            dist_func = torch.abs(torch.sum(cached_weights, dim=1) - torch.sum(current_weights, dim=1))
            dist_points = self.window_size - epoch_id

            self.lipshitz_constants.append(dist_func / dist_points)

    def get_lipschitz_constant(self) -> Tensor:
        ''' Computes current best lipshitz constant or returns None if the 
            window is not full 
        '''

        if len(self.lipshitz_constants) == self.lipschitz_size:
            return torch.max(torch.stack(list(self.lipshitz_constants), dim=1), dim=1)[0]
        return None

    def forward(self, x: Tensor, saturated: Tensor = None) -> Tuple:
        ''' saturated tensor should be provided from the previous layer

        '''
        self.__update_lipschitz_cache() # update lipschitz deque with current weights
        self.__update_cache() # update weights deque with current weights

        best_lipschitz_constant = self.get_lipschitz_constant() 
        u = 0

        if saturated is not None:
            sat = torch.logical_or(self.saturated_neurons, saturated)

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

        return x + u, best_lipschitz_constant
