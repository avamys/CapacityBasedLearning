import pandas as pd
import unittest
from unittest.mock import patch
from src.data.utils import *

class TestDataUtils(unittest.TestCase):
    def test_get_preprocessing_function(self):
        self.assertIs(get_preprocessing_function('drugs'), drugs)
        self.assertIs(get_preprocessing_function('obesity'), obesity)
        self.assertIs(get_preprocessing_function('mice'), mice)

    @patch('src.data.utils.pd')
    def test_anneal(self, mock_pd):
        mock_pd.read_csv.return_value = pd.DataFrame([['?','C','A',8,00,'?','S','?',000,
            '?','?','G','?','?','?','?','?','?','?','?','?','?','?','?','?','?','?','?',
            '?','?','?','COIL',0.700,0610.0,0000,'?',0000,'?','3'], ['?','C','R',00,00,
            '?','S','2',000,'?','?','E','?','?','?','?','?','?','?','?','?','?','?','?',
            '?','?','?','Y','?','?','?','COIL',3.200,0610.0,0000,'?',0000,'?','3']],
            columns=['family', 'product-type', 'steel', 'carbon', 'hardness', 
               'temper_rolling', 'condition', 'formability', 'strength', 
               'non-ageing', 'surface-finish', 'surface-quality', 'enamelability', 
               'bc', 'bf', 'bt', 'bw/me', 'bl', 'm', 'chrom', 'phos', 'cbond', 
               'marvi', 'exptl', 'ferro', 'corr', 'blue/bright/varn/clean', 
               'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width', 'len', 
               'oil', 'bore', 'packing', 'class'])
        X, y = anneal('placeholder')
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertTrue(X.isna().any().any())

    @patch('src.data.utils.pd')
    def test_adult(self, mock_pd):
        mock_pd.read_csv.return_value = pd.DataFrame([[
            39, 'State-gov', 77516, 'Bachelors', 13, 'Never-married', 'Adm-clerical', 
            'Not-in-family', 'White', 'Male', 2174, 0, 40, 'United-States', '<=50K'],
            [50, 'Self-emp-not-inc', 83311, 'Bachelors', 13, 'Married-civ-spouse', 
            'Exec-managerial', 'Husband', 'White', 'Male', 0, 0, 13, 'United-States', '<=50K']],
            columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'income'])
        X, y = adult('placeholder')
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(X.shape[1], 14)
        self.assertEqual(X.shape[0], 2)
        self.assertFalse(X.isna().any().any())
        self.assertFalse(y.isna().any().any())


if __name__ == '__main__':
    unittest.main()
