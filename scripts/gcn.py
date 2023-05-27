import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np

from torch_geometric.nn import GCNConv, BatchNorm,  global_mean_pool
from sklearn.metrics import accuracy_score, f1_score

# define classifier
ADD_FEATURES = 8
DEVICE = 'cpu'


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.001, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.val_loss_min = np.Inf
        self.tr_loss_min = np.Inf

    def __call__(self, epoch, tr_loss, val_loss):

        # if val loss is not decreased within delta, increase counter

        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.tr_loss_min = tr_loss
            self.counter = 0
        elif val_loss > (self.val_loss_min + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print('-----------------------')
                print(f'EARLY STOPPING \tBest train loss: {self.tr_loss_min:.4f}\tBest val loss: {self.val_loss_min:.4f}')
                print('-----------------------')
                return True


# actually the final model
class GCN(nn.Module):
    """Graph Convolutional Network
    Args:
        hidden_channels (int): number of hidden channels

    Returns:
        x (tensor): output of the model

    """
    def __init__(self, hidden_channels) -> None:
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, hidden_channels)

        self.lin1 = nn.Linear(self.conv1.out_channels + ADD_FEATURES, 32)
        self.lin2 = nn.Linear(32, 2)
  
        self.bn1 = BatchNorm(hidden_channels)

    def forward(self, data, edge_index, batch):
        """Forward pass of the model
        Args:
            data (torch_geometric.data): data
            edge_index (torch.tensor): edge index
            batch (torch.tensor): batch
            
        Returns:
            x (tensor): output of the model
        """
        # sequential model
        x = data.x
        x = F.gelu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = global_mean_pool(x, batch) 
        # append data.y[1] at the end of x
        x = torch.cat((x, data.y[1]), 1)
        x = F.relu(self.lin1(x))             # <-- x u features di pdata (sono 8)
        x = F.sigmoid(self.lin2(x))
        return x
    

def compute_accuracy(dataset, model: nn.Module, criterion):
    """Compute accuracy of the model
    Args:
        dataset (torch_geometric.data): dataset
        model (nn.Module): model
        criterion (nn.Module): loss function
    Returns:
        accuracy (float): accuracy of the model
    """

    model.eval()
    y_true = []
    y_pred = []
    losses = []
    for data in dataset:
        data = data.to(DEVICE)
        y_true.append(torch.argmax(data.y[0]).item())
    
    for data in dataset:
        data = data.to(DEVICE)
        output = model(data, data.edge_index, data.batch)
        loss = criterion(output, data.y[0])
        output = torch.argmax(output).item()
        y_pred.append(output)
        losses.append(loss.item())
    
    return f1_score(y_true, y_pred), np.mean(losses)


def train(epoch, model, criterion, optimizer, train_loader):
    """Train the model
    Args:
        epoch (int): current epoch
        model (nn.Module): model
        criterion (nn.Module): loss function
        optimizer (torch.optim): optimizer
        train_loader (torch_geometric.data.DataLoader): train loader
    Returns:
        epoch_loss (float): loss of the current epoch        
    """
    model.to(DEVICE)
    model.train()
    epoch_loss = 0
    
    for data in train_loader: 
        data = data.to(DEVICE)
        out = model(data, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y[0])                  # Compute the loss.
        loss.backward()                                   # Derive gradients.
        optimizer.step()                                  # Update parameters based on gradients
        optimizer.zero_grad()                             # Clear gradients.
        epoch_loss += loss.item()

    return epoch_loss/len(train_loader)

# define test loop
def test(loader, model, criterion):
    """
    Args:
        loader (torch_geometric.data.DataLoader): loader
        model (nn.Module): model
        criterion (nn.Module): loss function
    Returns:
        accuracy (float): accuracy of the model
    """
    return compute_accuracy(loader.dataset, model, criterion)


def train_loop(model, criterion, optimizer, train_loader=None, val_loader=None, epochs=100, early_stopping=None, verbose=True, min_loss=None, min_acc=None):
    """Train the model
    Args:
        model (nn.Module): model
        criterion (nn.Module): loss function
        optimizer (torch.optim): optimizer
        train_loader (torch_geometric.data.DataLoader): train loader
        val_loader (torch_geometric.data.DataLoader): validation loader
        epochs (int): number of epochs
        early_stopping (EarlyStopping): early stopping object
        verbose (bool): print results
        min_loss (float): minimum loss to stop training
        min_acc (float): minimum accuracy to stop training
    Returns:
        epoch_loss (float): loss of the current epoch
    """
    val_losses = []
    train_losses = []

    for epoch in range(epochs):
        
        loss = train(epoch, model, criterion, optimizer, train_loader)
 
        tr_acc, tr_loss   = test(train_loader, model, criterion)
        val_acc, val_loss = test(val_loader, model, criterion)
        val_losses.append(val_loss)
        train_losses.append(tr_loss)


        if verbose:
            print(f'Epoch: {epoch:03d}, Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if min_loss is not None and tr_loss <= min_loss:
            print(f'Min Loss reached: Epoch: {epoch:03d}, Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}')
            break

        if min_acc is not None and tr_acc >= min_acc:
            print(f'Min Acc reached: Epoch: {epoch:03d}, Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}')
            break
           
        if early_stopping is not None and early_stopping(epoch, tr_loss, val_loss):
            break


# wrap model in a sklearn classifier
class GCNClassifier(nn.Module):
    def __init__(self, model, criterion, optimizer, epochs=100, early_stopping=None, verbose=True, min_loss=None, min_acc=None):
        super(GCNClassifier, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.min_loss = min_loss
        self.min_acc = min_acc

    def fit(self, train_loader, val_loader=None):
        train_loop(self.model, self.criterion, self.optimizer, train_loader, val_loader, self.epochs, self.early_stopping, self.verbose, self.min_loss, self.min_acc)

    def predict(self, loader):
        return compute_accuracy(loader.dataset, self.model, self.criterion)

    def predict_proba(self, loader):
        self.model.eval()
        y_pred = []
        for data in loader:
            data = data.to(DEVICE)
            output = self.model(data, data.edge_index, data.batch)
            output = output.detach().numpy()
            y_pred.append(output)
        return np.concatenate(y_pred, axis=0)

    def score(self, loader):
        return self.predict(loader)[0]

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()