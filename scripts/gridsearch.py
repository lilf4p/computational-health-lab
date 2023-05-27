import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

import itertools
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from torch.utils.tensorboard import SummaryWriter


class LinNet(torch.nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x


# train model
def train(epoch, X_train, y_train, model, criterion, optimizer, device):
    model.to(device)
    model.train()
    epoch_loss = 0
    for x, l in zip(X_train, y_train):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        l = torch.tensor(l, dtype=torch.float32).to(device)
        out = model(x)
        loss = criterion(out, l)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    return epoch_loss / len(X_train)


def test(X_test, y_test, model, criterion, device):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for x, l in zip(X_test, y_test):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            l = torch.tensor(l, dtype=torch.float32).to(device)
            out = model(x)
            loss = criterion(out, l)
            epoch_loss += loss.item()
        return epoch_loss / len(X_test)


def trainer(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=500,
    learning_rate=1e-3,
    momentum=0.8,
    weight_decay=0,
):
    device = "cpu"
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    best_loss_val = np.inf
    best_loss_train = np.inf

    for epoch in range(epochs):
        train_loss = train(epoch, X_train, y_train, model, criterion, optimizer, device)
        val_loss = test(X_val, y_val, model, criterion, device)

        if val_loss < best_loss_val:
            best_loss_val = val_loss
            best_loss_train = train_loss

    return model, best_loss_train, best_loss_val, learning_rate, momentum, weight_decay


def process_input(args):
    best_loss_train = np.inf
    best_loss_val = np.inf

    X_train, y_train, X_val, y_val, params, res = args

    lr = params[0]
    mom = params[1]
    wd = params[2]

    input_shape = X_train.shape[1]
    model = LinNet(input_shape)

    model, loss_train, loss_val, lr, mom, wd = trainer(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=1000,
        learning_rate=lr,
        momentum=mom,
        weight_decay=wd,
    )

    if loss_val < best_loss_val:
        best_loss_val = loss_val
        best_loss_train = loss_train

    # print(f'eta: {lr:.4f}, alpha: {mom:.3f}, lambda: {wd:.7f} | Train Loss: {best_loss_train:.4f} | Val Loss: {loss_val:.4f}')
    # make a dict of the results
    return {
        "eta": lr,
        "alpha": mom,
        "lambda": wd,
        "train_loss": best_loss_train,
        "val_loss": loss_val,
    }


# TODO: take model class as input, make more general.
def grid_search(X_train, y_train, X_val, y_val, etas, momentums, lambdas, workers=6):
    # spawn new process for each hyperparameter combination, max 6 processes
    results = []
    prod = itertools.product(etas, momentums, lambdas)

    # add data to each process
    prod = [(X_train, y_train, X_val, y_val, p, results) for p in prod]

    with mp.Pool(processes=workers) as pool:
        waiter = pool.map_async(process_input, prod)
        res = waiter.get()

    return res
