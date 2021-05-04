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


class LinearBud(nn.Module):
    ''' Static Bud model with two 3x3 hidden layers and given output size '''
    def __init__(self, size_in: int, size_out: int, layers: List[int], 
                 activation_name: str, activation_out: str):
        super().__init__()

        self.weight = torch.ones(1, size_in) * 1/size_in
        activation_method = get_activation(activation_name)
        activation_out = get_activation(activation_out)
        layerlist = []
        n_in = size_in

        for layer in layers:
            layerlist.append(nn.Linear(n_in, layer))
            layerlist.append(activation_method())
            n_in = layer
        layerlist.append(nn.Linear(layers[-1], size_out))
        layerlist.append(activation_out())

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x):
        x = (self.weight * x)
        return self.layers(x)


class NeuronBud(nn.Module):
    ''' Parametrized Bud model '''
    counter = 0
    def __init__(self, size_in: int, size_out: int, window_size: int, 
                 threshold: float, layers: List[int],
                 activation_name: str, activation_out: str, level: int = 0, parent: int = 0):
        super().__init__()

        self.window_size = window_size
        self.threshold = threshold
        self.activation = get_activation(activation_name)()
        self.activation_out = get_activation(activation_out)()
        
        self.level = level + 1
        self.parent_id = parent
        self.id = NeuronBud.counter
        NeuronBud.counter += 1

        self.weight = torch.ones(1, size_in) * 1/size_in
        n_in = size_in

        self.layerlist = nn.ModuleList()

        for layer_id, layer in enumerate(layers):
            self.layerlist.append(BuddingLayer(n_in, layer, window_size, level=self.level, idx=layer_id))
            n_in = layer
        self.layerlist.append(BuddingLayer(layers[-1], size_out, window_size, level=self.level, idx=len(layers)-1))

    def get_saturation(self, best_lipschitz):
        if best_lipschitz is not None:
            return best_lipschitz < self.threshold
        return None

    def get_n_buds(self):
        ''' Aggregate parameters of every layer '''
        n_buds = 0
        lipschitz_consts = dict()
        for idx, layer in enumerate(self.layerlist):
            n_buds += len(layer.buds)
            lips = layer.get_lipschitz_constant()
            if lips is not None:
                keys = [str(i) for i in range(lips.shape[0])]
                lipschitz_consts[f'lipschitz_{self.level}_{self.parent_id}_{self.id}_{idx}'] = dict(zip(keys, lips))
            lipschitz_consts.update(layer.lower_lipschitz)

        return n_buds, lipschitz_consts

    def forward(self, x, optim=None):
        # Distribution of input weights to the first layer
        x = (self.weight * x)

        # Default forward pass
        x, lip = self.layerlist[0].forward(x, optim=optim)
        for i, l in enumerate(self.layerlist[1:]):
            x = self.activation(x)
            saturation = self.get_saturation(lip)
            x, lip = self.layerlist[i+1].forward(x, saturation, optim)

        return self.activation_out(x)


class BuddingLayer(nn.Module):
    ''' Budding layer that stores and activates the buds for saturated 
        neurons.
    '''
    def __init__(self, size_in: int, size_out: int, window_size: int, 
                 compute_lipschitz: bool = True, bias: bool = True, level: int = 0, idx: int = 0):
        super().__init__()

        self.window_size = window_size
        self.lipschitz_size = int(binom(self.window_size+1, 2))
        self.weights_window = deque(maxlen=self.window_size)
        self.lipshitz_constants = deque(maxlen=self.lipschitz_size)
        self.saturated_neurons = torch.zeros(size_in, dtype=torch.bool)

        self.buds_optims = []
        
        self.size_in = size_in
        self.size_out = size_out
        self.buds = nn.ModuleDict()
        
        self.level = level
        self.id = idx
        self.n_buds = 0
        self.lower_lipschitz = dict()

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
            dist_func = torch.abs(torch.sum(cached_weights - current_weights, dim=1))
            dist_points = self.window_size - epoch_id

            self.lipshitz_constants.append(dist_func / dist_points)

    def get_lipschitz_constant(self) -> Tensor:
        ''' Computes current best lipshitz constant or returns None if the 
            window is not full 
        '''

        if len(self.lipshitz_constants) == self.lipschitz_size:
            return torch.max(torch.stack(list(self.lipshitz_constants), dim=1), dim=1)[0]
        return None

    def get_n_buds(self):
        ''' Sum up number of buds grown in lower levels '''
        sum_buds = 0
        for key in self.buds:
            lower_n_buds, buds_lipschitz = self.buds[key].get_n_buds()
            sum_buds += lower_n_buds
            self.lower_lipschitz.update(buds_lipschitz)

        self.n_buds = sum_buds

    def forward(self, x: Tensor, saturated: Tensor = None, optim=None) -> Tuple:
        ''' saturated tensor should be provided from the previous layer

        '''
        if self.training:
            self.__update_lipschitz_cache() # update lipschitz deque with current weights
            self.__update_cache() # update weights deque with current weights

        best_lipschitz_constant = self.get_lipschitz_constant() 
        u = 0

        if saturated is not None:
            if self.training:
                sat = torch.logical_or(self.saturated_neurons, saturated)
            else:
                # Growing buds is not allowed in evaluation mode
                sat = self.saturated_neurons

            if sat.any().item() > 0:
                # Add new NeuronBuds if the number of buds has changed
                if torch.sum(sat) > torch.sum(self.saturated_neurons):
                    u_neurons = torch.nonzero(sat).view(-1).tolist()
                    u_buds = [int(key) for key in self.buds.keys()]
                    
                    for new_bud in list(set(u_neurons) - set(u_buds)):
                        self.buds[str(new_bud)] = NeuronBud(
                            size_in=3,
                            size_out=self.size_out, 
                            window_size=self.window_size,
                            threshold=0.001, 
                            layers=(3,3), 
                            activation_name='tanh', 
                            activation_out='tanh', 
                            level=self.level,
                            parent=self.id)
                        
                        # Add new parameters to optimizer
                        if optim is not None:
                            optim.add_param_group({'params': self.buds[str(new_bud)].parameters()})
                    
                    self.saturated_neurons = sat

                # Forward pass through buds
                u = [self.buds[key].forward(x[:, int(key)].view(-1,1), optim) for key in self.buds]
                u = torch.stack(u, dim=1)
                u = torch.sum(u, dim=1)

                # Apply linear transformation for non-saturated neurons
                x = x * (~self.saturated_neurons).float()

            self.get_n_buds()
            
        x = F.linear(x, self.weight, self.bias)

        return x + u, best_lipschitz_constant


class CapacityModel(nn.Module):
    def __init__(self, size_in: int, size_out: int, window_size: int, 
                 threshold: float, layers: List[int],
                 activation_name: str):
        
        super().__init__()

        self.window_size = window_size
        self.threshold = threshold

        self.activation = get_activation(activation_name)()
        n_in = size_in
        self.layerlist = nn.ModuleList()

        for layer in layers:
            self.layerlist.append(BuddingLayer(n_in, layer, window_size))
            n_in = layer
        self.layerlist.append(BuddingLayer(layers[-1], size_out, window_size))

    def get_saturation(self, best_lipschitz):
        if best_lipschitz is not None:
            return best_lipschitz < self.threshold
        return None

    def get_n_buds(self):
        ''' Aggregate all information from lower levels' layers and buds
            (Used for logging)
        '''
        n_buds_core = 0
        n_buds_layers = 0
        lower_lipschitz = dict()
        for layer in self.layerlist:
            n_buds_core += len(layer.buds)
            n_buds_layers += layer.n_buds
            lower_lipschitz.update(layer.lower_lipschitz)
        return n_buds_core + n_buds_layers, lower_lipschitz

    def forward(self, x, optim=None):
        x, lip = self.layerlist[0].forward(x, optim=optim)
        for i, l in enumerate(self.layerlist[1:]):
            x = self.activation(x)
            saturation = self.get_saturation(lip)
            x, lip = self.layerlist[i+1].forward(x, saturation, optim)

        return x
