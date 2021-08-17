import unittest
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

from src.data.datasets import Dataset
from src.models.network import CapacityModel
from src.models.config import Configurator


class TestConfigurator(unittest.TestCase):
    def setUp(self):
        X = np.random.rand(256,5)
        y = np.random.randint(0, 2, 256)
        ds = Dataset(X, y)
        config_file = 'experiments/experiment_test.yaml'
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.trainer = Configurator(config, ds, verbosity=0)

    def test_init_params(self):
        params = self.trainer.parameters
        
        m1 = CapacityModel(size_in=5, size_out=1, **params['model_params'])
        m2 = CapacityModel(size_in=5, size_out=1, **params['model_params'])
        l1 = m1.layerlist[0]
        l2 = m2.layerlist[0]

        self.assertFalse(torch.equal(l1.weight, l2.weight))
        self.assertFalse(torch.equal(l1.bias, l2.bias))
        self.assertEqual(len(self.trainer.model_init_params), 0)

        self.trainer._save_init_params(m1)
        self.assertNotEqual(len(self.trainer.model_init_params), 0)
        
        self.trainer._load_init_params(m2)
        self.assertTrue(torch.equal(l1.weight, l2.weight))
        self.assertTrue(torch.equal(l1.bias, l2.bias))

    def test_run(self):
        self.trainer.epochs = 2
        results = self.trainer.run()
        best = results.get_best_config(metric="Accuracy_test", mode="max")

        self.assertTrue('layers' in best['model_params'])
        self.assertTrue('window_size' in best['model_params'])

    def test_run_baseline(self):
        self.trainer.epochs = 2
        results = self.trainer.run(baseline=True)
        best = results.get_best_config(metric="Accuracy_test", mode="max")

        self.assertTrue('layers' in best['model_params'])
        self.assertFalse('window_size' in best['model_params'])
        self.assertIn('window_size', self.trainer.parameters['model_params'].keys())
        self.assertIn('threshold', self.trainer.parameters['model_params'].keys())
        self.assertIn('layers', self.trainer.parameters['model_params'].keys())
