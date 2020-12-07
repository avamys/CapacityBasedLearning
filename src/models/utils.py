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
