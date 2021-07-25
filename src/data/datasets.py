import torch
import numpy as np
from typing import Union, List, Tuple

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class Dataset():
    def __init__(self, X: Union[np.ndarray, torch.Tensor] = None, 
                 y: Union[np.ndarray, torch.Tensor] = None) -> None:
        self.X = X
        self.y = y

    def save(path: str) -> None:
        concat = np.hstack([self.X, self.y.reshape((-1,1))])
        np.savetxt(path_x, concat, delimiter=",")

    @staticmethod
    def load(path: str) -> 'Dataset':
        concat = np.genfromtxt(path, delimiter=',')
        X = concat[:, :-1]
        y = concat[:, -1]
        return Dataset(X, y)

    def to_tensors(self) -> None:
        if not torch.is_tensor(self.X):
            self.X = torch.from_numpy(self.X)
            self.y = torch.from_numpy(self.y)

    def to_numpy(self) -> None:
        if torch.is_tensor(self.X):
            self.X = self.X.numpy()
            self.y = self.y.numpy()

    def split(self, test_size: float, random_state: int = None) -> Tuple['Dataset', 'Dataset']:
        ''' Performs train/test split and returns results as tenors '''
        self.to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
            test_size=test_size, random_state=random_state, shuffle=True)
        X_train = torch.Tensor(X_train)
        X_test = torch.Tensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        return Dataset(X_train, y_train), Dataset(X_test, y_test)

    def as_dataloader(self, batch_size: int = 64) -> DataLoader:
        self.to_tensors()
        torch_dataset = TensorDataset(self.X, self.y)
        torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
        return torch_dataloader
        

class DatasetGenerator():
    def __init__(self, n_samples: int, n_numerical: int, n_categorical: int, 
                 n_binary: int, noise: float, n_classes: int, 
                 random_state: int = None) -> None:
        self.base = {
            'n_samples': n_samples, 'n_numerical': n_numerical, 
            'n_categorical': n_categorical, 'n_binary': n_binary, 
            'noise': noise, 'n_classes': n_classes}
        self.ids = {key: i for i, key in enumerate(self.base.keys())}
        self.random_state = random_state
            
    def generate_dataset(
            self, n_samples: int, n_numerical: int, n_categorical: int, 
            n_binary: int, noise: float, n_classes: int,
            weights: List[float] = None, class_sep: float = 1.0, 
            random_state: int = None) -> Dataset:

        def get_binary_sep(col: np.ndarray):
            return (col.max() + col.min()) / 2
            
        def get_categorical_sep(col: np.ndarray, n_cat: int):
            return np.linspace(col.min(), col.max(), num=n_cat, endpoint=False)

        n_features = n_numerical + n_categorical + n_binary
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_features, 
            n_redundant=0, n_classes=n_classes, weights=weights, 
            flip_y=noise, class_sep=class_sep, random_state=random_state)

        for bin_idx in range(n_binary):
            col = -1 - bin_idx
            X[:, col] = X[:, col] > get_binary_sep(X[:, col])

        for cat_idx in range(n_categorical):
            col = -1 - n_binary - cat_idx
            X[:, col] = np.digitize(X[:, col], get_categorical_sep(X[:, col], 3)) - 1

        return Dataset(X, y)

    def get_base(self) -> Dataset:
        return self.generate_dataset(**self.base, random_state=self.random_state)

    def make_datasets(self, folder: str, feature: str, range_min: int, range_max: int, dist: int):
        values = np.linspace(range_min, range_max, dist)
        params = self.base
        feature_id = self.ids[feature]

        for value in values:
            params[feature] = value
            ds = self.generate_dataset(**params, random_state=self.random_state)
            ds.save(folder+f'K0_F{feature_id}_{value}.csv')
