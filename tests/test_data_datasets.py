import os
import torch
import unittest
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from src.data.datasets import Dataset, DatasetGenerator


class TestDataset(unittest.TestCase):
    def setUp(self):
        X = np.random.rand(5,3)
        y = np.array([1, 0, 2, 0, 1])
        self.ds = Dataset(X, y)

    def test_properties(self):
        self.assertEqual(self.ds.cols, 3)
        self.assertEqual(self.ds.rows, 5)
        self.assertEqual(self.ds.targets, 3)

    def test_conversions(self):
        self.ds.to_numpy()
        self.assertIsInstance(self.ds.X, np.ndarray)
        
        self.ds.to_tensors()
        self.assertIsInstance(self.ds.X, torch.Tensor)

        self.ds.to_numpy()
        self.assertIsInstance(self.ds.X, np.ndarray)
            
    def test_data_split(self):
        train, test = self.ds.split(test_size=0.2)
        
        self.assertIsInstance(train, Dataset)
        self.assertIsInstance(test, Dataset)
        self.assertIsInstance(train.X, torch.Tensor)
        self.assertIsInstance(test.X, torch.Tensor)
        self.assertIsInstance(train.y, torch.LongTensor)
        self.assertIsInstance(test.y, torch.LongTensor)
        self.assertEqual(test.rows, 1)
        self.assertEqual(train.rows, 4)

    def test_as_dataloader(self):
        dl = self.ds.as_dataloader(batch_size=2)

        self.assertIsInstance(dl, DataLoader)
        self.assertEqual(len(dl), 3)


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        base = {
            'n_samples': 1000, 'n_numerical': 10, 'n_categorical': 5, 
            'n_binary': 5, 'noise': 0.0, 'n_classes': 2}
        self.generator = DatasetGenerator(**base, random_state=121)

    def test_init(self):
        self.assertEqual(self.generator.ids['noise'], 4)
        self.assertEqual(self.generator.ids['n_classes'], 5)

    def test_generator_base(self):
        ds = self.generator.get_base()

        self.assertIsInstance(ds, Dataset)
        self.assertEqual(ds.X.shape[0], 1000)
        self.assertEqual(ds.y.shape[0], 1000)

    def test_generator(self):
        path = f"data/processed/test/"
        config = {
            'n_numerical': {'min': 10, 'max': 30, 'dist': 10}
        }
        for feature, params in config.items():
            self.generator.make_datasets(
                path, feature, params['min'], params['max'], params['dist'])

        created = os.listdir(path)
        self.assertEqual(len(created), 3)
        
        ds = Dataset.load(path+'K0_F1_10.csv')
        self.assertEqual(ds.cols, 20)
        ds = Dataset.load(path+'K0_F1_20.csv')
        self.assertEqual(ds.cols, 30)
        ds = Dataset.load(path+'K0_F1_30.csv')
        self.assertEqual(ds.cols, 40)


if __name__ == '__main__':
    unittest.main()
        