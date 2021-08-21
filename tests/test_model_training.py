import numpy as np
import unittest
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.training import *
from src.models.network import Model, CapacityModel
from src.models.utils import get_metrics_dict

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(5,3)
        self.y = np.array([1, 0, 2, 0, 1])
        self.model = Model(self.X.shape[1], 3, (6,4), 'relu', 'relu')
        self.capmodel = CapacityModel(self.X.shape[1], 3, 3, 0.1, (10,10), 'relu', {})
        settings = {}
        self.metric = get_metrics_dict(['accuracy'], settings)[0]

    def test_calculate_accuracy(self):
        y_1 = torch.tensor([1,0,1,0,0])
        y_2 = torch.tensor([1,0,1,0,1])
        y_3 = torch.tensor([1,1,0,1,1])

        accuracy12 = self.metric(y_1, y_2)['Accuracy_train'].item()
        self.metric.reset()
        accuracy13 = self.metric(y_1, y_3)['Accuracy_train'].item()

        self.assertAlmostEqual(accuracy12, 0.8)
        self.assertAlmostEqual(accuracy13, 0.2)

    def test_training_step(self):
        X_t = torch.Tensor(self.X)
        y_t = torch.LongTensor(self.y)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.capmodel.parameters(), lr=0.01)

        loss1 = training_step(self.capmodel, criterion, optimizer, X_t, y_t, self.metric, True)
        loss2 = training_step(self.capmodel, criterion, optimizer, X_t, y_t, self.metric, True)
        
        self.assertNotEqual(loss1, loss2)

    def test_validation_step(self):
        X_t = torch.Tensor(self.X)
        y_t = torch.LongTensor(self.y)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=0.01)

        loss1 = validation_step(self.model, criterion, X_t, y_t, self.metric)
        loss2 = validation_step(self.model, criterion, X_t, y_t, self.metric)
        
        self.assertEqual(loss1, loss2)

if __name__ == '__main__':
    unittest.main()
