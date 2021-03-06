import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from ray import tune

from network import Model
from training import *
from src.models.config import Configurator
from src.models.utils import get_criterion

import yaml
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_features', type=click.File('rb'))
@click.argument('input_target', type=click.File('rb'))
@click.argument('config_file', type=click.Path())
def main(input_features, input_target, config_file):
    """ Runs model training scripts to create and train models on prepared
        datasets (from ../processed) and save them in ../models.
    """
    logger = logging.getLogger(__name__)
    
    # Get reproducible results
    torch.manual_seed(42)

    # Load config dict for running experiment
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    training_params = config['training']

    logger.info('loading data')
    X = np.genfromtxt(input_features, delimiter=',')
    y = np.genfromtxt(input_target, delimiter=',')
    X_train, X_test, y_train, y_test = data_split(X, y, training_params['test_size'])

    logger.info('running')

    criterion = get_criterion(training_params['criterion'])()
    epochs = training_params['epochs']
    trainer = Configurator(config, criterion, epochs, X_train, y_train, X_test, y_test)
    analysis = trainer.run()

    print("Best config: ",analysis.get_best_config(metric="accuracy", mode="max"))

    # logger.info('saving trained model')
    # save_model(trained_model, 'model1.pth')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
