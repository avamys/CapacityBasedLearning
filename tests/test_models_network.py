import numpy as np
import unittest
import torch
import torch.nn as nn

from src.models.network import NeuronBud, BuddingLayer

class TestBuddingLayer(unittest.TestCase):
    def setUp(self):
        self.model = BuddingLayer(2, 3, 3)
        self.bud = BuddingLayer(2, 2, 2)

        self.bud.weight = nn.Parameter(torch.ones((2,2)))
        self.bud.update_cache()
        self.bud.weight = nn.Parameter(torch.ones((2,2)) * 2)
        self.bud.update_cache()

    def test_buds(self):
        x = torch.tensor([[0.1, 0.1], [1.0, 1.0]])
        saturated = torch.BoolTensor([True, False])

        self.assertEqual(len(self.model.buds), 0)
        
        out = self.model(x, saturated)
        
        self.assertEqual(len(self.model.buds), 1)
        self.assertEqual(list(self.model.buds.keys())[0], '0')
        self.assertEqual(out.size()[0], 2)
        self.assertEqual(out.size()[1], 3)

    def test_window_cap(self):
        self.assertIsNot(self.bud.weights_window[0], self.bud.weights_window[1])
        self.assertEqual(len(self.bud.weights_window), 2)
        self.bud.weight = nn.Parameter(torch.ones((2,2)))
        self.bud.update_cache()
        self.assertEqual(len(self.bud.weights_window), 2)

    def test_lipschitz(self):
        lip = self.bud.get_lipschitz_constant()[0]
        lip_self = self.bud.lipshitz_constants[-1]

        self.assertEqual(len(self.bud.lipshitz_constants), 2)
        self.assertTrue(torch.equal(lip_self, torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.equal(lip, torch.tensor([2.0, 2.0])))


if __name__ == '__main__':
    unittest.main()