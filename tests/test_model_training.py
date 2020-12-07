import numpy as np
import unittest
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.training import *
from src.models.network import Model


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(5,3)
        self.y = np.array([1, 0, 2, 0, 1])
        self.model = Model(self.X.shape[1], 3, (6,4), 'relu', 'relu')

    def test_data_split(self):
        X_tr, X_t, y_tr, y_t = data_split(self.X, self.y)
        
        self.assertIsInstance(X_tr, torch.Tensor)
        self.assertIsInstance(X_t, torch.Tensor)
        self.assertIsInstance(y_tr, torch.LongTensor)
        self.assertIsInstance(y_t, torch.LongTensor)

    def test_calculate_accuracy(self):
        y_1 = torch.tensor([[0,1],[1,0],[0,1],[1,0],[1,0]])
        y_2 = torch.tensor([1,0,1,0,1])
        y_3 = torch.tensor([1,1,0,1,1])

        self.assertEqual(calculate_accuracy(y_1, y_2), 0.8)
        self.assertEqual(calculate_accuracy(y_1, y_3), 0.2)

    def test_training_step(self):
        X_t = torch.Tensor(self.X)
        y_t = torch.LongTensor(self.y)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=0.01)

        loss1 = training_step(self.model, criterion, optimizer, X_t, y_t)
        loss2 = training_step(self.model, criterion, optimizer, X_t, y_t)
        
        # self.assertNotAlmostEqual(loss1, loss2)
        self.assertNotEqual(loss1, loss2)

if __name__ == '__main__':
    unittest.main()
