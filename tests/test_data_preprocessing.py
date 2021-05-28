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
        
        X_p, y_p = self.dp.preprocess_data(X, y)

        self.assertIsNot(X, X_p)
        self.assertIsNot(y, y_p)
        self.assertFalse(np.isnan(X_p).any())
        self.assertEqual(X_p.shape[1], 7)
        self.assertTrue((X_p[:, 2]==np.array([1,0,0])).all())
        self.assertTrue((y_p == np.array([0,1,1,0])).all())

if __name__ == '__main__':
    unittest.main()
