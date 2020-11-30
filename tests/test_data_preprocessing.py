import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch
from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.dp = DataPreprocessor()

    @patch('src.data.preprocessing.utils')
    def test_read_data(self, mock_utils):
        mock_utils.get_preprocessing_function.return_value = lambda x: x.split('x')

        self.dp.read_data("data/x/drugs.csv")
        
        self.assertEqual(self.dp.dataset_name, 'drugs')
        self.assertEqual(self.dp.X, 'data/')
        self.assertEqual(self.dp.y, '/drugs.csv')

    def test_preprocess_data(self):
        self.dp.X = pd.DataFrame([
                [0.0, 0.0, 'a', 'x'],
                [1.0, np.nan, 'b', np.nan],
                [2.0, 1.0, 'b', 'y'],
            ])
        self.dp.y = pd.Series(['a', 'b', 'b', 'a'])
        
        self.dp.preprocess_data()

        self.assertFalse(np.isnan(self.dp.X).any())
        self.assertEqual(self.dp.X.shape[1], 7)
        self.assertTrue((self.dp.X[:, 2]==np.array([1,0,0])).all())
        self.assertTrue((self.dp.y == np.array([0,1,1,0])).all())

if __name__ == '__main__':
    unittest.main()
