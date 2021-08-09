import os
import numpy as np
import torch
import ray

from ray import tune
from ray.tune.analysis import ExperimentAnalysis
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

from typing import Dict, List, Union, Callable, Optional

from src.models.network import Model, CapacityModel
from src.models.training import training_step, validation_step
from src.visualization.visualize import TBLogger
from src.models.utils import get_optimizer, get_criterion
from src.data.datasets import Dataset


LossFunction = Callable[[Tensor, Tensor], Tensor]
Config = Dict[str, Union[int, float, str, List[int]]]

class Configurator():
    def __init__(self, config: Config, dataset: Dataset) -> None:
        self.config = config
        self.name = self.config['name']
        self.parameters = self.config['parameters']
        self.local_dir = self.config['dir']

        training_params = config['training']
        self.criterion = get_criterion(training_params['criterion'])()
        self.epochs = training_params['epochs']
        self.dataset = dataset
        self.trainset, self.testset = self.dataset.split(training_params['test_size'])
        self.baseline = False
        self.model_init_params = []

    def _save_init_params(self, model: torch.nn.Module):
        self.model_init_params = []
        for layer in model.layerlist:
            weight, bias = layer.parameters()
            params = {'weight': weight.data.clone(), 'bias': bias.data.clone()}
            self.model_init_params.append(params)

    def _load_init_params(self, model: torch.nn.Module):
        for idx, layer in enumerate(model.layerlist):
            layer.weight.data = self.model_init_params[idx]['weight']
            if layer.bias is not None:
                layer.bias.data = self.model_init_params[idx]['bias']

    def train(self, config: Config, checkpoint_dir: Optional[str] = None) -> None:
        ''' Performs full training of a specified model in specified number of epochs '''

        writer = SummaryWriter(log_dir="custom_logs")

        train_dataloader = self.trainset.as_dataloader(batch_size=config['batch_size'])
        test_dataloader = self.testset.as_dataloader(batch_size=config['batch_size'])

        if not self.baseline:
            logger = TBLogger(writer)
            forward_optim = True
            model = CapacityModel(
                size_in=self.trainset.cols, 
                size_out=self.trainset.targets, 
                **config['model_params'])

        else:
            forward_optim = False
            model = Model(
                size_in=self.trainset.cols,
                size_out=self.trainset.targets, 
                **config['model_params'])

        writer.add_graph(model, self.trainset.X)

        optimizer = get_optimizer(config['optimizer'])(model.parameters(), lr=config['learning_rate'])

        for epoch in range(self.epochs):

            losses_train, losses_validate, accuracies = [], [], []
            for X, y in train_dataloader:
                # Training step
                loss_train = training_step(model, self.criterion, optimizer, X, y, forward_optim=forward_optim)
                losses_train.append(loss_train)
            
            if not self.baseline:
                model.update_budding_layers()
                # Logging
                logger.log_model_params(model, epoch)

            for X, y in test_dataloader:
                # Validation step
                loss_validate, accuracy = validation_step(model, self.criterion, X, y)
                losses_validate.append(loss_validate)
                accuracies.append(accuracy)

            tune.report(loss_train=np.mean(losses_train), loss_val=np.mean(losses_validate), accuracy=np.mean(accuracies))

        writer.close()

    def run(self) -> ExperimentAnalysis:
        self.baseline = False
        result = tune.run(self.train, config=self.parameters, name=self.name, verbose=0, local_dir=self.local_dir)
        return result

    def run_baseline(self) -> ExperimentAnalysis:
        self.baseline = True
        baseline_keys = ('layers', 'activation_name', 'activation_out')
        set_params = {key: self.parameters['model_params'][key] for key in baseline_keys}
        self.parameters['model_params'] = set_params
        name = f'{self.name}_baseline'
        result = tune.run(self.train, config=self.parameters, name=name, verbose=0, local_dir=self.local_dir)
        return result
