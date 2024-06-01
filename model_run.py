"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn import metrics
def crossEntropy():
    pass
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()
    pred_list = []
    target_list = []
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
       
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        train_loss += loss.item() 

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        pred_list.append(y_pred_class.cpu().detach().numpy())
        target_list.append(y.cpu().detach().numpy())
    
    all_pred = np.concatenate(np.array(pred_list), axis=0)
    all_target = np.concatenate(np.array(target_list), axis=0)
    # all_pred = torch.cat(pred_list, dim=0)
    # all_target = torch.cat(target_list, dim=0)
    # loss = loss_fn(all_pred, all_target)
    train_loss = train_loss / len(dataloader)
    acc = metrics.accuracy_score(all_target, all_pred)
    auc = metrics.roc_auc_score(all_target, all_pred)
    return train_loss, acc, auc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 
    pred_list = []
    target_list = []
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            
            pred_list.append(test_pred_labels.cpu().detach().numpy())
            target_list.append(y)
        all_pred = np.concatenate(np.array(pred_list), axis=0)
        all_target = np.concatenate(np.array(target_list), axis=0)
        # all_pred = torch.cat(pred_list, dim=0)
        # all_target = torch.cat(target_list, dim=0)
        # loss = loss_fn(all_pred, all_target)
        test_loss = test_loss / len(dataloader)
        acc = metrics.accuracy_score(all_target, all_pred)
        auc = metrics.roc_auc_score(all_target, all_pred)
        return loss, acc, auc