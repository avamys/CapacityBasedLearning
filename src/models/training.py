import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.models.network import Model


def data_split(features, target, test_size=0.2):
    ''' Performs train/test split and returns results as tenors '''

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42, shuffle=True)
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test

def calculate_accuracy(target_predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ''' Calculates accuracy from the given tensors '''

    classes = torch.argmax(target_predicted, dim=1)
    accuracy = torch.mean((classes==target).float())

    return accuracy


def training_step(model, criterion, optimizer, features, target):
    ''' Performs single training step and returns loss '''

    model.train()
    target_pred = model.forward(features)
    loss = criterion(target_pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def validation_step(model, criterion, features, target):
    ''' Performs single validations step and returns loss and accuracy '''

    model.eval()
    with torch.no_grad():
        target_val = model.forward(features)
        loss = criterion(target_val, target)

    accuracy = calculate_accuracy(target_val, target)

    return loss.item(), accuracy.item()

def train(model, criterion, optimizer, epochs, features, target, features_val, target_val, writer):
    ''' Performs full training of a specified model in specified number of epochs '''

    losses_train, losses_validate = [], []
    
    for epoch in tqdm(range(epochs)):
        loss_train = training_step(model, criterion, optimizer, features, target)
        losses_train.append(loss_train)
        writer.add_scalar("Loss/train", loss_train, epoch)

        loss_validate, accuracy = validation_step(model, criterion, features_val, target_val)
        losses_validate.append(loss_validate)
        writer.add_scalar("Loss/test", loss_validate, epoch)
        writer.add_scalar('Accuracy', accuracy, epoch)

    return model

def save_model(model: Model, name: str):
    ''' Saves model with a given name to a proper folder '''

    torch.save(model.state_dict(), f'models/{name}')
