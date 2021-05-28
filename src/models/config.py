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
from src.models.utils import get_optimizer


LossFunction = Callable[[Tensor, Tensor], Tensor]
Config = Dict[str, Union[int, float, str, List[int]]]

class Configurator():
    def __init__(self, config: Config, criterion: LossFunction, epochs: int,
                 features: Tensor, target: Tensor, features_val: Tensor, 
                 target_val: Tensor) -> None:
        self.config = config
        self.criterion = criterion
        self.epochs = epochs

        self.features = features
        self.target = target
        self.features_val = features_val
        self.target_val = target_val

        self.name = self.config['name']
        self.parameters = self.config['parameters']
        self.local_dir = self.config['dir']

    def train(self, config: Config, checkpoint_dir: Optional[str] = None) -> None:
        ''' Performs full training of a specified model in specified number of epochs '''

        writer = SummaryWriter(log_dir="custom_logs")
        logger = TBLogger(writer)

        losses_train, losses_validate = [], []

        model = CapacityModel(
            size_in=self.features.shape[1], 
            size_out=len(np.unique(self.target)), 
            window_size=config['window_size'], 
            threshold=config['threshold'], 
            layers=config['layers'], 
            activation_name=config['activation'],
            buds_params=config['buds_parameters']
        )

        writer.add_graph(model, self.features)

        optimizer = get_optimizer(config['optimizer'])(model.parameters(), lr=config['learning_rate'])

        for epoch in range(self.epochs):

            # Training step
            loss_train = training_step(model, self.criterion, optimizer, self.features, self.target, forward_optim=True)
            losses_train.append(loss_train)

            # Logging
            logger.log_model_params(model, epoch)

            # Validation step
            loss_validate, accuracy = validation_step(model, self.criterion, self.features_val, self.target_val)
            losses_validate.append(loss_validate)

            tune.report(loss_train=loss_train, loss_val=loss_validate, accuracy=accuracy)

        writer.close()

    def run(self) -> ExperimentAnalysis:
        result = tune.run(self.train, config=self.parameters, name=self.name, verbose=3, local_dir=self.local_dir)
        return result
