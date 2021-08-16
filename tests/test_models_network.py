import numpy as np
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.network import NeuronBud, BuddingLayer

class TestBuddingLayer(unittest.TestCase):
    def setUp(self):
        self.params = {
            'learning_rate': 0.01,
            'size_in': 3,
            'threshold': 0.001,
            'decline': 1.0,
            'layers': [3,3],
            'activation': 'tanh'
        }
        self.bud = BuddingLayer(2, 2, 1, self.params)
        self.bud.weight = nn.Parameter(torch.ones((2,2)))
        self.bud.bias = nn.Parameter(torch.ones(2))

        self.x = torch.tensor([[0.1, 0.1], [1.0, 1.0]])
        self.saturated = torch.BoolTensor([False, True])
        self.optim = torch.optim.Adam(self.bud.parameters(), 0.001)

    def test_buds_creation(self):
        self.assertEqual(len(self.bud.buds), 0)
        y, lip = self.bud(self.x, self.saturated, self.optim)

        self.assertEqual(len(self.bud.buds), 1)
        self.assertEqual(list(self.bud.buds.keys())[0], '1')
        self.assertEqual(y.size()[0], 2)
        self.assertEqual(y.size()[1], 2)

    def test_basic_forward(self):
        y, lip = self.bud.forward(self.x)
        
        self.assertEqual(len(self.bud.weights_window), 0)
        self.bud.update_state()
        self.assertEqual(len(self.bud.weights_window), 1)
        self.assertEqual(len(self.bud.lipshitz_constants), 0)
        self.assertIsNone(lip)

        expected_y = F.linear(self.x, self.bud.weight, self.bud.bias)
        self.assertTrue(torch.equal(y, expected_y))

    def test_forward_with_bud(self):
        y, lip = self.bud.forward(self.x, self.saturated, self.optim)
        self.bud.eval()

        u = self.bud.buds['1'](self.x[:, 1].view(-1, 1), self.optim)
        zero_x = self.x * (~self.saturated).float()
        self.assertTrue(torch.equal(zero_x, torch.tensor([[0.1, 0], [1.0, 0]])))

        clear_x = F.linear(zero_x, self.bud.weight, self.bud.bias)
        expected_y = clear_x + u

        self.assertTrue(torch.equal(y, expected_y))

    def test_window_cap(self):
        model = BuddingLayer(2, 2, 2, self.params)
        model.weight = nn.Parameter(torch.ones((2,2)))
        self.assertEqual(len(model.weights_window), 0)

        model.update_state()
        y, lip = model.forward(self.x, self.saturated, self.optim)
        self.assertEqual(len(model.weights_window), 1)

        model.weight = nn.Parameter(torch.ones((2,2)) * 2)
        self.assertIsNot(model.weights_window[0], model.state_dict()['weight'])

        model.update_state()
        y, lip = model.forward(self.x, self.saturated, self.optim)
        self.assertEqual(len(model.weights_window), 2)
        self.assertIsNot(model.weights_window[0], model.weights_window[1])

    def test_lipschitz(self):
        lip = self.bud.get_lipschitz_constant()
        self.assertIsNone(lip)

        self.bud.update_state()
        y, lip = self.bud.forward(self.x, self.saturated, self.optim)
        self.bud.weight = nn.Parameter(torch.ones((2,2)) * 2)

        self.bud.update_state()
        y, lip = self.bud.forward(self.x, self.saturated, self.optim)

        self.assertEqual(len(self.bud.lipshitz_constants), 1)
        self.assertTrue(torch.equal(lip, torch.tensor([2.0, 2.0])))

class TestNeuronBud(unittest.TestCase):
    def setUp(self):
        params = {
            'size_in': 2,
            'threshold': 0.01,
            'decline': 1.0,
            'layers': [2,2],
            'activation': 'tanh'
        }
        self.model = NeuronBud(size_in=2, size_out=1, threshold=0.01, 
                               decline=1.0, layers=[2, 2], window_size=5,
                               activation_name='tanh', buds_params=params)

    def test_init(self):
        self.assertEqual(len(self.model.layerlist), 3)
        
        expected_weight = torch.tensor([0.5, 0.5]).view(1, -1)
        self.assertTrue(torch.equal(self.model.weight, expected_weight))

    def test_best_lipschitz(self):
        lipschitz = None
        saturation = self.model.get_saturation(lipschitz)
        self.assertIs(saturation, None)

        lipschitz = torch.tensor([1, 1, 0.01, 0.001])
        expected = torch.tensor([False, False, False, True])
        saturation = self.model.get_saturation(lipschitz)
        print(saturation)
        self.assertTrue(torch.equal(expected, saturation))

if __name__ == '__main__':
    unittest.main()