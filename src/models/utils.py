import os
import random
import torch
import torch.nn as nn
import numpy as np
import torchmetrics.functional as metrics

from typing import Tuple


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

def get_metrics_dict(metric_list: Tuple[str]):
    ''' Returns metrics dict in format metric_name: metric_function '''

    classification_metrics = {
        'accuracy': metrics.accuracy,
        'average_precision': metrics.average_precision,
        'f1': metrics.f1,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'roc': metrics.roc
    }

    return {metric: classification_metrics[metric] for metric in metric_list}

def seed_everything(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
