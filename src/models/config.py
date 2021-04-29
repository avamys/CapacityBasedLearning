import os
import numpy as np
import torch
import ray

from ray import tune
from torch.utils.tensorboard import SummaryWriter

from src.models.network import Model, CapacityModel
from src.models.training import training_step, validation_step


class Configurator():
    def __init__(self, config, criterion, epochs, features, target, features_val, target_val):
        self.config = config
        self.criterion = criterion
        self.epochs = epochs

        self.features = features
        self.target = target
        self.features_val = features_val
        self.target_val = target_val

        self.name = self.config['name']
        self.parameters = self.config['parameters']

    def train(self, config, checkpoint_dir=None):
        ''' Performs full training of a specified model in specified number of epochs '''

        writer = SummaryWriter(log_dir="custom_logs")

        losses_train, losses_validate = [], []

        model = CapacityModel(
            size_in=self.features.shape[1], 
            size_out=len(np.unique(self.target)), 
            window_size=config['window_size'], 
            threshold=config['threshold'], 
            layers=config['layers'], 
            activation_name=config['activation']
        )

        writer.add_graph(model, self.features)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['optim_lr'])
        for epoch in range(self.epochs):
            loss_train = training_step(model, self.criterion, optimizer, self.features, self.target)
            losses_train.append(loss_train)

            loss_validate, accuracy = validation_step(model, self.criterion, self.features_val, self.target_val)
            losses_validate.append(loss_validate)

            tune.report(loss_train=loss_train, loss_val=loss_validate, accuracy=accuracy)

        writer.close()

    def run(self):
        result = tune.run(self.train, config=self.parameters, name=self.name, verbose=3, local_dir='runs')
        return result
