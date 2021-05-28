# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.network import BuddingLayer
from src.models.training import *

# %%
BATCH_SIZE = 64
EPOCHS = 50


# %%
class Net(nn.Module):
    def __init__(self, in_features: int, out_size: int):
        super().__init__()
        
        n_in = in_features

        self.bl1 = nn.Linear(n_in, 10)
        self.bl2 = nn.Linear(10, 9)
        self.blout = nn.Linear(9, out_size)
        
    def forward(self, x):
        x = F.relu(self.bl1(x))
        x = F.relu(self.bl2(x))
        x = self.blout(x)

        return x


# %%
class CapacityModel(nn.Module):
    def __init__(self, size_in: int, size_out: int, window_size: int, 
                 threshold: float, layers,
                 activation_name: str):
        
        super().__init__()

        self.window_size = window_size
        self.threshold = threshold

        self.activation = nn.ReLU()
        n_in = size_in
        self.layerlist = nn.ModuleList()

        for layer in layers:
            self.layerlist.append(BuddingLayer(n_in, layer, window_size))
            n_in = layer
        self.layerlist.append(BuddingLayer(layers[-1], size_out, window_size))

    def get_saturation(self, best_lipschitz):
        if best_lipschitz is not None:
            return best_lipschitz < self.threshold
        return None

    def forward(self, x, optim=None):
        x, lip = self.layerlist[0].forward(x, optim=optim)
        for i, l in enumerate(self.layerlist[1:]):
            x = self.activation(x)
            saturation = self.get_saturation(lip)
            x, lip = self.layerlist[i+1].forward(x, saturation, optim)

        return x


# %%
input_features = "../data/processed/obesity_features.csv"
input_target = "../data/processed/obesity_target.csv"

X = np.genfromtxt(input_features, delimiter=',')
y = np.genfromtxt(input_target, delimiter=',')
X_train, X_test, y_train, y_test = data_split(X, y)

# %%
trainloader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=True)

# %%
torch.manual_seed(101)
model = CapacityModel(
    X_train.shape[1], len(np.unique(y_train)), window_size=5, threshold=0.01, layers=[10, 9], activation_name='relu'
)
benchmark = Net(X_train.shape[1], len(np.unique(y_train)))

# %%
model

# %%
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)
bench_optim = Adam(benchmark.parameters(), lr=0.01)

# %%
losses = []
for epoch in range(EPOCHS):
    y_pred = model.forward(X_train, optimizer)
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    print(f"Epoch: {epoch+1}  Loss: {loss.item():10.3f}")
    print(f"SATURATION:\nBL2: {model.layerlist[1].saturated_neurons}\nBLout: {model.layerlist[2].saturated_neurons}")
    #print(model.state_dict())
    #print(model.get_extra_params())
    print("=========================================================================================\n")
    
    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()


# %%
plt.plot(losses)

# %%
losses = []
for epoch in range(EPOCHS):
    y_pred = benchmark.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    print(f"Epoch: {epoch+1}  Loss: {loss.item():10.3f}")
    print("=========================================================================================\n")
    
    bench_optim.zero_grad()
    loss.backward()
    bench_optim.step()

# %%
plt.plot(losses)

# %%
