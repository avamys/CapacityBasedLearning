import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing import Dict, Callable, Union


LossFunction = Callable[[Tensor, Tensor], Tensor]
Number = Union[int, float]
Metrics = Dict[str, Callable[[Tensor, Tensor], Number]]


def training_step(model: Module, criterion: LossFunction, 
                  optimizer: Optimizer, features: Tensor, target: Tensor, 
                  metrics: Metrics, forward_optim: bool = False):
    ''' Performs single training step, computes metrics and returns loss '''

    model.train()

    if forward_optim:
        target_pred = model.forward(features, optimizer)
    else:
        target_pred = model.forward(features)
        
    loss = criterion(target_pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    metrics(target_pred, target)

    return loss.item()

def validation_step(model: Module, criterion: LossFunction, features: Tensor, 
                    target: Tensor, metrics: Metrics):
    ''' Performs single validation step, computes metrics and returns loss '''

    model.eval()
    with torch.no_grad():
        target_val = model.forward(features)
        loss = criterion(target_val, target)

    metrics(target_val, target)

    return loss.item()
