import torch.nn as nn
import unittest
from src.models.utils import *


class TestModelUtils(unittest.TestCase):
    def test_get_activation(self):
        self.assertIs(get_activation('relu'), nn.ReLU)
        self.assertIs(get_activation('sigmoid'), nn.Sigmoid)
        self.assertIs(get_activation('tanh'), nn.Tanh)
        self.assertIs(get_activation('softmax'), nn.Softmax)

    def test_calculate_lipschitz_constant(self):
        w = torch.rand(2,2)
        k1 = calculate_lipschitz_constant(torch.tensor([0.1, 0.3]), torch.tensor([0.11, 0.31]), w)
        k2 = calculate_lipschitz_constant(torch.tensor([0.1, 0.3]), torch.tensor([0.09, 0.29]), w)

        self.assertAlmostEqual(k1[0].item(), k2[0].item(), 5)
        self.assertAlmostEqual(k1[1].item(), k2[1].item(), 5)
        self.assertEqual(k2.size()[0], 2)

if __name__ == '__main__':
    unittest.main()
