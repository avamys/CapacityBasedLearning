import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import src.data.utils as utils


class DataPreprocessor:
    ''' Class for reading dataset file from a given path, processing it
        properly and saving to given path as a csv file.
    '''
    def __init__(self):
        self.X, self.y = pd.DataFrame(), pd.DataFrame()
        self.dataset_name = str()

    def read_data(self, path: str):
        ''' Reads raw data into DataProcessor '''

        # Pull out the dataset name from the path
        self.dataset_name = path.split('/')[-1].split('.')[0]
        preprocessing_function = utils.get_preprocessing_function(self.dataset_name)
        self.X, self.y = preprocessing_function(path)

    def preprocess_data(self):
        ''' Preprocess data using defined pipelines for categorical and
            numerical features
        '''
        if self.dataset_name == 'drugs':
            self.X = self.X.to_numpy()
        else:
            # Pipeline for numeric columns
            numeric = self.X.loc[:, (self.X.dtypes == np.float64) | (self.X.dtypes == np.int64) ].columns
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])

            # Pipeline for categorical columns
            categorical = self.X.loc[:, self.X.dtypes == object].columns
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numeric', numeric_pipeline, numeric),
                    ('categorical', categorical_pipeline, categorical)])

            self.X = preprocessor.fit_transform(self.X)

        # Label encoding for potential target variables
        le = LabelEncoder()
        self.y = pd.Series(le.fit_transform(self.y), name='class')

    def write_data(self, path: str):
        ''' Saves X and y as csv files in given directory '''

        # In case the ndarray was converted to sparse
        if type(self.X) != np.ndarray:
            self.X = self.X.toarray()

        np.savetxt(path+self.dataset_name+'_target'+'.csv', self.y, delimiter=",")
        np.savetxt(path+self.dataset_name+'_features'+'.csv', self.X, delimiter=",")
