import os
import zipfile
import pandas as pd
import numpy as np
from urllib import request
from typing import Tuple, Callable, Union

Dataset = Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]


def get_url(dataset: str) -> str:

    urls = {
        'binary': "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        'multiclass': "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip",
        'extreme': "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
    }

    return urls[dataset]

def get_dataset(dataset: str, folder: str) -> None:
    url = get_url(dataset)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, dataset)

    if not os.path.exists(file_path):
        request.urlretrieve(url, file_path)
        if url.endswith(".zip"):
            file = zipfile.ZipFile(file_path)
            file.extractall(folder)

def get_preprocessing_function(dataset: str) -> Callable[[str], Dataset]:
    ''' Returns preprocessing function given the corresponding dataset name '''

    preprocessing_function = {
        'binary': adult,
        'multiclass': dry_bean,
        'extreme': letter_recognition
    }

    return preprocessing_function[dataset]

def get_range(range_min: Union[int, float], range_max: Union[int, float], 
              dist: Union[int, float]):
    ''' Simple inclusive range generator '''
    num = range_min
    while num <= range_max:
        yield num
        num += dist


def drugs(file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ''' Loads and does preliminary processing for UCI drugs dataset '''

    columns = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
               'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive',
               'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis',
               'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
               'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

    df = pd.read_csv(file, delimiter=',', names=columns)
    df.drop("ID", inplace=True, axis=1)

    # Index of first potential target class
    target_index = df.columns.to_list().index('Alcohol')
    cols_x = df.columns[:target_index]

    # Choose one of the classes as a target variable
    cols_y = df.columns[target_index]

    X, y = df.loc[:, cols_x], df.loc[:, cols_y]

    return X, y


def obesity(file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ''' Loads and does preliminary processing for UCI Obesity dataset '''

    df = pd.read_csv(file, delimiter=',')

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X, y


def mice(file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ''' Loads and does preliminary processing for UCI Mice dataset '''

    df = pd.read_excel(file)
    df.drop("MouseID", inplace=True, axis=1)

    na_cols = ["BAD_N", "BCL2_N", "H3AcK18_N", "EGR1_N", "H3MeK4_N"]
    df.drop(na_cols, axis=1, inplace=True)

    # There are 4 possible class columns - choose the most specific one
    cols_y = df.columns[-1]
    cols_X = df.columns[:-4]
    X, y = df.loc[:, cols_X], df.loc[:, cols_y]

    return X, y


def adult(root: str) -> Dataset:
    ''' Loads and does preliminary processing for UCI Adult dataset '''

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'income']
    path = f'{root}/binary'
    df = pd.read_csv(path, delimiter=',', names=columns)

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X, y

def letter_recognition(root: str) -> Dataset:

    columns = ['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar',
               'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
    path = f'{root}/extreme'
    df = pd.read_csv(path, delimiter=',', names=columns)

    X, y = df.iloc[:, 1:], df.iloc[:, 0]

    return X, y

def dry_bean(root: str) -> Dataset:
    path = f'{root}/DryBeanDataset/Dry_Bean_Dataset.xlsx'
    df = pd.read_excel(path)

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X, y

def anneal(file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ''' Loads and does preliminary processing for UCI Annealing dataset '''

    columns = ['family', 'product-type', 'steel', 'carbon', 'hardness', 
               'temper_rolling', 'condition', 'formability', 'strength', 
               'non-ageing', 'surface-finish', 'surface-quality', 'enamelability', 
               'bc', 'bf', 'bt', 'bw/me', 'bl', 'm', 'chrom', 'phos', 'cbond', 
               'marvi', 'exptl', 'ferro', 'corr', 'blue/bright/varn/clean', 
               'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width', 'len', 
               'oil', 'bore', 'packing', 'class']

    df = pd.read_csv(file, delimiter=',', names=columns)

    df.replace('?', np.nan, inplace=True)

    # Drop columns with majority of nan values
    drop = df.isna().sum() > 300
    to_drop = [df.columns[x] for x in range(len(drop)) if drop[x] == True]
    df.drop(to_drop, axis=1, inplace=True)

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X, y
