import torch.nn as nn
import unittest
from src.models.utils import *


class TestModelUtils(unittest.TestCase):
    def test_get_activation(self):
        self.assertIs(get_activation('relu'), nn.ReLU)
        self.assertIs(get_activation('sigmoid'), nn.Sigmoid)
        self.assertIs(get_activation('tanh'), nn.Tanh)
        self.assertIs(get_activation('softmax'), nn.Softmax)


if __name__ == '__main__':
    unittest.main()
