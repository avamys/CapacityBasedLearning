import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from typing import Tuple, Union

import src.data.utils as utils

Dataset = Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]

class DataPreprocessor:
    ''' Class for reading dataset file from a given path, processing it
        properly and saving to given path as a csv file.
    '''
    def __init__(self, name: str):
        self.dataset_name = name

    def read_data(self, path: str) -> Dataset:
        ''' Reads raw data into DataProcessor '''

        utils.get_dataset(self.dataset_name, path)

        preprocessing_function = utils.get_preprocessing_function(self.dataset_name)
        return preprocessing_function(path)

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Dataset:
        ''' Preprocess data using defined pipelines for categorical and
            numerical features
        '''

        # Pipeline for numeric columns
        numeric = X.loc[:, (X.dtypes == np.float64) | (X.dtypes == np.int64) ].columns
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        # Pipeline for categorical columns
        categorical = X.loc[:, X.dtypes == object].columns
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_pipeline, numeric),
                ('categorical', categorical_pipeline, categorical)])

        X = preprocessor.fit_transform(X)

        # Label encoding for potential target variables
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name='class')

        return X, y

    def write_data(self, path: str, X: pd.DataFrame, y: pd.Series) -> None:
        ''' Saves X and y as csv files in given directory '''

        # In case the ndarray was converted to sparse
        if type(X) != np.ndarray:
            X = X.toarray()

        np.savetxt(path+self.dataset_name+'_target'+'.csv', y, delimiter=",")
        np.savetxt(path+self.dataset_name+'_features'+'.csv', X, delimiter=",")
