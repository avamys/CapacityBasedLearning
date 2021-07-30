import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch
from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.dp = DataPreprocessor('extreme')

    def test_read_data(self):
        X, y = self.dp.read_data("data/raw")
        
        self.assertEqual(self.dp.dataset_name, 'extreme')
        self.assertIsInstance(X, pd.pandas.core.frame.DataFrame)
        self.assertIsInstance(y, pd.pandas.core.frame.Series)

    def test_preprocess_data(self):
        X = pd.DataFrame([
            [0.0, 0.0, 'a', 'x'],
            [1.0, np.nan, 'b', np.nan],
            [2.0, 1.0, 'b', 'y']
            ])
        y = pd.Series(['a', 'b', 'b', 'a'])
        
        ds = self.dp.preprocess_data(X, y)

        self.assertIsNot(X, ds.X)
        self.assertIsNot(y, ds.y)
        self.assertFalse(np.isnan(ds.X).any())
        self.assertEqual(ds.X.shape[1], 7)
        self.assertTrue((ds.X[:, 2]==np.array([1,0,0])).all())
        self.assertTrue((ds.y == np.array([0,1,1,0])).all())

if __name__ == '__main__':
    unittest.main()
