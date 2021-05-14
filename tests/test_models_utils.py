import torch.nn as nn
import unittest
from src.models.utils import *


class TestModelUtils(unittest.TestCase):
    def test_get_activation(self):
        self.assertIs(get_activation('relu'), nn.ReLU)
        self.assertIs(get_activation('sigmoid'), nn.Sigmoid)
        self.assertIs(get_activation('tanh'), nn.Tanh)
        self.assertIs(get_activation('softmax'), nn.Softmax)

    def test_get_optimizer(self):
        self.assertIs(get_optimizer('adam'), torch.optim.Adam)
        self.assertIs(get_optimizer('sgd'), torch.optim.SGD)

    def test_get_criterion(self):
        self.assertIs(get_criterion('cross-entropy'), nn.CrossEntropyLoss)
        self.assertIs(get_criterion('mse'), nn.MSELoss)


if __name__ == '__main__':
    unittest.main()
