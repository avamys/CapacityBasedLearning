import numpy as np
from typing import List

from sklearn.datasets import make_classification

class DatasetGenerator():
    def __init__(self):
        pass
            
    def generate_dataset(
            self, n_samples: int, n_numerical: int, n_categorical: int, 
            n_binary: int, noise: float, n_classes: int,
            weights: List[float] = None, class_sep: float = 1.0, 
            random_state: int = None):

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

        return X, y
