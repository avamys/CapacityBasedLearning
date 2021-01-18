import torch
import torch.nn as nn


def get_activation(activation: str):
    ''' Returns activation function given the corresponding name '''

    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'softmax': nn.Softmax
    }

    return activations[activation]

def calculate_lipschitz_constant(x1: torch.tensor, x2: torch.tensor, weights: torch.tensor):
    ''' Returns the value of Lipschitz constant for the given points '''

    dist_func = torch.abs(weights @ x2 - weights @ x1)
    dist_points = torch.norm(x2 - x1)
    return dist_func / dist_points
