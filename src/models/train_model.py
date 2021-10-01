import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from ray import tune

from network import Model
from training import *
from src.models.config import Configurator
from src.models.utils import get_criterion, seed_everything
from src.data.datasets import Dataset

import yaml
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_data', type=click.Path())
@click.argument('config_file', type=click.Path())
@click.option('--baseline', '-b', is_flag=True)
@click.option('--project', '-p', type=str)
def main(input_data, config_file, baseline, project):
    """ Runs model training scripts to create and train models on prepared
        datasets (from ../processed) and save them in ../models.
    """
    logger = logging.getLogger(__name__)
    
    # Get reproducible results
    random_seed = 121
    seed_everything(random_seed)

    # Load config dict for running experiment
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.info('loading data')
    ds = Dataset.load(input_data)
    dsid = input_data.strip('/')[-1]

    logger.info('running')

    project = project if project else config['name']
    trainer = Configurator(
        config, ds, dataset_id=dsid, enable_wandb=True, random_state=random_seed, project=project)
    analysis = trainer.run(save_results=True)
    if baseline:
        analysis_b = trainer.run(baseline=True, save_results=True)
        print("Best config (baseline): ",analysis_b.get_best_config(metric="Accuracy_test", mode="max"))
    print("Best config: ",analysis.get_best_config(metric="Accuracy_test", mode="max"))

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
