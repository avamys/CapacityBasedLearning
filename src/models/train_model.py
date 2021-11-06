import os
import yaml
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.models.config import Configurator
from src.models.utils import seed_everything
from src.data.datasets import Dataset, DatasetGenerator


@click.command()
@click.argument('config_file', type=click.Path())
@click.option('--input_data', '-i', type=click.Path())
@click.option('--baseline', '-b', is_flag=True)
@click.option('--project', '-p', type=str)
@click.option('--generate', '-g', is_flag=True)
def main(config_file, input_data, baseline, project, generate):
    """ Runs model training scripts to create and train models on prepared
        datasets (from ../processed).
    """
    logger = logging.getLogger(__name__)
    
    # Get reproducible results
    random_seed = 121
    seed_everything(random_seed)

    # Load config dict for running experiment
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Run on single dataset
    if input_data:
        logger.info('loading data')
        ds = Dataset.load(input_data)
        dsid = input_data.strip('/')[-1]
        project = project if project else config['name']
        trainer = Configurator(config, ds, dataset_id=dsid, enable_wandb=True,
                               random_state=random_seed, project=project)

        logger.info('running')
        analysis = trainer.run(save_results=True)
        if baseline:
            analysis_b = trainer.run(baseline=True, save_results=True)
            print("Best config (baseline): ",analysis_b.get_best_config(metric="Accuracy_test", mode="max"))
        print("Best config: ",analysis.get_best_config(metric="Accuracy_test", mode="max"))
    
    # Run on generated datasets defined in config file
    if generate:
        project = project if project else 'capacity-dataset'
        trainer = Configurator(config, random_state=random_seed, 
                               project=project, enable_wandb=True)

        dg = DatasetGenerator(**config['base_dataset'], random_state=random_seed)
        path = dg.generate("data/processed", config)
        logger.info('all datasets created')

        for dataset in os.listdir(path):
            ds = Dataset.load(f"{path}/{dataset}")
            logger.info(f'running on {dataset}')
            if dataset != 'K0.csv':
                trainer.project = f'{project}-{dataset[3:5]}'
            else:
                trainer.project = f'{project}-base'
            analysis = trainer.run(dataset=ds, dataset_id=dataset)
            if baseline:
                analysis_b = trainer.run(baseline=True, save_results=True)
                print("Best config (baseline): ",analysis_b.get_best_config(metric="Accuracy_test", mode="max"))
            print("Best config: ",analysis.get_best_config(metric="Accuracy_test", mode="max"))

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
