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
        self.train, self.test = self.dataset.split(training_params['test_size'])

        self.model_type = self.parameters['model']
        self.forward_optim = True if self.model_type == 'capacity' else False

    def get_model(self, model: str):
        models = {
            'capacity': CapacityModel,
            'benchmark': Model
        }
        return models[model]

    def train(self, config: Config, checkpoint_dir: Optional[str] = None) -> None:
        ''' Performs full training of a specified model in specified number of epochs '''

        writer = SummaryWriter(log_dir="custom_logs")
        logger = TBLogger(writer)

        train_dataloader = self.train.as_dataloader(batch_size=config['batch_size'])
        test_dataloader = self.test.as_dataloader(batch_size=config['batch_size'])

        model = self.get_model(self.model_type)(
            size_in=self.train.cols, 
            size_out=self.train.targets, 
            **config['model_params'])

        writer.add_graph(model, self.features)

        optimizer = get_optimizer(config['optimizer'])(model.parameters(), lr=config['learning_rate'])

        for epoch in range(self.epochs):

            losses_train, losses_validate, accuracies = [], [], []
            for X, y in train_dataloader:
                # Training step
                loss_train = training_step(model, self.criterion, optimizer, X, y, forward_optim=self.forward_optim)
                losses_train.append(loss_train)
            
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
        result = tune.run(self.train, config=self.parameters, name=self.name, verbose=3, local_dir=self.local_dir)
        return result
