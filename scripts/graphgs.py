import torch 
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np

import itertools
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

ADD_FEATURES = 7
STARTING_CHANNEL = 4


class GCN(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super(GCN, self).__init__()
        torch.manual_seed(7477)
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels*2*2)
        self.conv4 = GCNConv(hidden_channels*2*2, hidden_channels*2*2*2)

        self.lin1 = nn.Linear(self.conv4.out_channels + ADD_FEATURES, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 2)
        
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels*2)
        self.bn3 = BatchNorm(hidden_channels*2*2)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, data, edge_index, batch):
        # sequential model
        x = data.x
        x = F.gelu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.gelu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.gelu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.gelu(self.conv4(x, edge_index))

        x = global_mean_pool(x, batch) 
        # append data.y[1] at the end of x
        x = torch.cat((x, data.y[1]), 1)
    
        x = F.gelu(self.lin1(x))             # <-- x u features di pdata (sono 9)

        x = F.gelu(self.lin2(x))
        x = F.gelu(self.lin3(x))
        x = F.sigmoid(self.lin4(x))
        return x


def compute_accuracy(dataset, model, criterion, device='cpu'):
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    for data in dataset:
        data = data.to(device)
        y_true.append(torch.argmax(data.y[0]).item())
    
    for data in dataset:
        data = data.to(device)
        output = model(data, data.edge_index, data.batch)
        loss = criterion(output, data.y[0])
        output = torch.argmax(output).item()
        y_pred.append(output)
        losses.append(loss.item())
    
    return np.mean(losses)


def train(epoch, model, criterion, train_data, optimizer, device='cpu'):
    model.to(device)
    model.train()
    epoch_loss = 0
    
    for data in train_data: 
        data = data.to(device)
        out = model(data, data.edge_index, data.batch)    # Perform a single forward pass.
        loss = criterion(out, data.y[0])                  # Compute the loss.
        loss.backward()                                   # Derive gradients.
        optimizer.step()                                  # Update parameters based on gradients
        optimizer.zero_grad()                             # Clear gradients.
        epoch_loss += loss.item()

    #print(f'Epoch: {epoch:03d}, Loss: {epoch_loss/len(train_data):.4f}')
    return epoch_loss/len(train_data)

# define test loop
def test(loader, model, criterion, device='cpu'):
    return compute_accuracy(loader.dataset, model, criterion, device)


def trainer(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    momentum,
    weight_decay,
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
        train_loss = train(epoch, model, criterion, train_loader, optimizer, device)
        val_loss = test(val_loader, model, criterion, device)

        if val_loss < best_loss_val:
            best_loss_val = val_loss
            best_loss_train = train_loss

    return model, best_loss_train, best_loss_val, learning_rate, momentum, weight_decay


def process_input(args):

    best_loss_train = np.inf
    best_loss_val = np.inf

    train_loader, val_loader, params, res = args

    lr = params[0]
    mom = params[1]
    wd = params[2]

    model = GCN(hidden_channels=16)

    model, loss_train, loss_val, lr, mom, wd = trainer(
        model,
        train_loader,
        val_loader,
        epochs=1000,
        learning_rate=lr,
        momentum=mom,
        weight_decay=wd,
    )

    if loss_val < best_loss_val:
        best_loss_val = loss_val
        best_loss_train = loss_train

    print(f'eta: {lr:.4f}, alpha: {mom:.3f}, lambda: {wd:.7f} | Train Loss: {best_loss_train:.4f} | Val Loss: {loss_val:.4f}')
    # make a dict of the results
    return {
        "eta": lr,
        "alpha": mom,
        "lambda": wd,
        "train_loss": best_loss_train,
        "val_loss": loss_val,
    }


# TODO: take model class as input, make more general.
def grid_search(train_loader, val_loader, etas, momentums, lambdas, workers=6):
    # spawn new process for each hyperparameter combination, max 6 processes
    results = []
    prod = itertools.product(etas, momentums, lambdas)

    # add data to each process
    prod = [(train_loader, val_loader, p, results) for p in prod]
    print(prod)
    with mp.Pool(processes=workers) as pool:
        waiter = pool.map_async(process_input, prod)
        res = waiter.get()

    return res


