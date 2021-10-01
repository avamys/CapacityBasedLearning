# -*- coding: utf-8 -*-
import click
import logging
import yaml
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from preprocessing import DataPreprocessor
from src.data.datasets import DatasetGenerator


@click.command()
@click.argument('dataset')
@click.argument('input_path', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.option('--config_file', '-c', type=click.Path())
@click.option('--generate', '-g', is_flag=True)
def main(dataset, input_path, output_filepath, config_file, generate):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    if generate:
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        random_state = 121
        dg = DatasetGenerator(**config['base_dataset'], random_state=random_state)
        ds = dg.get_base()
        ds.save("data/processed/K0.csv")
    else:
        logger.info('making final data sets from raw data')
        preprocessor = DataPreprocessor(dataset)

        logger.info('reading data file')
        X, y = preprocessor.read_data(input_path)

        logger.info('processing data')
        ds = preprocessor.preprocess_data(X, y)

        logger.info('saving processed data')
        save_dir = f'{output_filepath}/{preprocessor.dataset_name}.csv'
        ds.save(save_dir)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
