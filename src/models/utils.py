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

def get_optimizer(optimizer: str):
    ''' Returns torch optimizer given the corresponding name '''

    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }

    return optimizers[optimizer]

def get_criterion(criterion: str):
    ''' Returns torch loss function given the corresponding name '''

    criterions = {
        'cross-entropy': nn.CrossEntropyLoss,
        'mse': nn.MSELoss
    }

    return criterions[criterion]