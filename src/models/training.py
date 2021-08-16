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
                  metrics: Metrics, metric_kwargs: dict = {}, 
                  forward_optim: bool = False):
    ''' Performs single training step and returns loss and dict of computed metrics '''

    model.train()

    if forward_optim:
        target_pred = model.forward(features, optimizer)
    else:
        target_pred = model.forward(features)
        
    loss = criterion(target_pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    computed_metrics = dict()
    for metric, fnc in metrics.items():
        computed_metrics[metric] = fnc(target_pred, target, **metric_kwargs).item()

    return loss.item(), computed_metrics

def validation_step(model: Module, criterion: LossFunction, features: Tensor, 
                    target: Tensor, metrics: Metrics, 
                    metric_kwargs: dict = {}):
    ''' Performs single validation step and returns loss and dict of computed metrics '''

    model.eval()
    with torch.no_grad():
        target_val = model.forward(features)
        loss = criterion(target_val, target)

    computed_metrics = dict()
    for metric, fnc in metrics.items():
        computed_metrics[metric] = fnc(target_val, target, **metric_kwargs).item()

    return loss.item(), computed_metrics
