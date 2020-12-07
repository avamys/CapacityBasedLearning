import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from network import Model
from training import *

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_features', type=click.File('rb'))
@click.argument('input_target', type=click.File('rb'))
def main(input_features, input_target):
    """ Runs model training scripts to create and train models on prepared
        datasets (from ../processed) and save them in ../models.
    """
    logger = logging.getLogger(__name__)
    
    # Get reproducible results
    torch.manual_seed(42)

    writer = SummaryWriter()

    logger.info('loading data')
    X = np.genfromtxt(input_features, delimiter=',')
    y = np.genfromtxt(input_target, delimiter=',')
    X_train, X_test, y_train, y_test = data_split(X, y)

    logger.info('running')
    # Set up network and training parameters
    model = Model(X_train.shape[1], len(np.unique(y_train)), (100, 240, 10), 'relu', 'softmax')
    writer.add_graph(model, X_train)
    trained_model = train(model, nn.CrossEntropyLoss(), Adam(model.parameters(), lr=0.01),
                          300, X_train, y_train, X_test, y_test, writer)
    
    writer.close()

    logger.info('saving trained model')
    save_model(trained_model, 'model1.pth')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
