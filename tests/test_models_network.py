import numpy as np
import unittest
import torch
import torch.nn as nn

from src.models.network import NeuronBud, BuddingLayer

class TestNetwork(unittest.TestCase):
    def test_budding_layer(self):
        model = BuddingLayer(2, 3)
        x = torch.tensor([0.1, 0.1])
        saturated = torch.BoolTensor([True, False])
        
        out = model(x, saturated)
        
        self.assertEqual(len(model.buds), 1)
        self.assertEqual(list(model.buds.keys())[0], '0')
        self.assertEqual(out.size()[0], 3)

if __name__ == '__main__':
    unittest.main()