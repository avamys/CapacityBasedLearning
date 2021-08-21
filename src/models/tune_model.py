from ray import tune
from src.models.config import Configurator
from src.data.datasets import Dataset, DatasetGenerator

import os
import yaml
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('config_file', type=click.Path())
@click.option('--dataset', default=None, type=click.Path())
@click.option('--generate', '-g', is_flag=True)
@click.option('--baseline', '-b', is_flag=True)
def main(config_file, dataset, all_datasets, baseline):
    """ Runs model training scripts to create and train models on prepared
        datasets (from ../processed) and save them in ../models.
    """
    logger = logging.getLogger(__name__)

    # Load config dict for running experiment
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.info('loading data')
    random_state = 121

    trainer = Configurator(config)

    if dataset:
        ds = Dataset.load(dataset)
        logger.info('running')
        trainer.load_new_dataset(ds)
        trainer.run(dataset=ds, save_results=True)
        if baseline:
            trainer.run(baseline=True, save_results=True)

    elif generate:
        dg = DatasetGenerator(**config['base_dataset'], random_state=random_state)
        path = dg.generate("data/processed", config)
        logger.info('all datasets created')

        for dataset in os.listdir(path):
            ds = Dataset.load(f"{path}/{dataset}")
            logger.info(f'running on {dataset}')
            analysis = trainer.run(dataset=ds, save_results=True)
            if baseline:
                analysis_b = trainer.run(baseline=True, save_results=True)
            print("Best config: ",analysis.get_best_config(metric="accuracy", mode="max"))
            print("Best config (baseline): ",analysis_b.get_best_config(metric="accuracy", mode="max"))

    logger.info('task finished')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
