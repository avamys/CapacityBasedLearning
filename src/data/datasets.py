import torch
import numpy as np
from typing import Union, List, Tuple

from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader


class Dataset():
    def __init__(self, X: Union[np.ndarray, torch.Tensor] = None, 
                 y: Union[np.ndarray, torch.Tensor] = None) -> None:
        
        self.X = X
        self.y = y

    def save(path_x: str, path_y: str) -> None:
        np.savetxt(path_x, self.X, delimiter=",")
        np.savetxt(path_y, self.y, delimiter=",")

    @staticmethod
    def load(path_x: str, path_y: str) -> Dataset:
        X = np.genfromtxt(path_x, delimiter=',')
        y = np.genfromtxt(path_y, delimiter=',')
        return Dataset(X, y)

    def split(self, test_size, random_state) -> Tuple[Dataset, Dataset]:
        ''' Performs train/test split and returns results as tenors '''

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
            test_size=test_size, random_state=random_state, shuffle=True)
        X_train = torch.Tensor(X_train)
        X_test = torch.Tensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        return Dataset(X_train, y_train), Dataset(X_test, y_test)

    def as_dataloader(self, batch_size):
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

    def make_datasets(self, feature: str, range_min: int, range_max: int, dist: int):
        values = np.linspace(range_min, range_max, dist)
        params = self.base

        for value in values:
            params[feature] = value
            ds = generate_dataset(**params, random_state=self.random_state)

