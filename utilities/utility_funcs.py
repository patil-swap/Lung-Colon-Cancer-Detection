import os

import numpy as np
import matplotlib.pyplot as plt
import torch


def get_model_name(name, batch_size, learning_rate, epoch):
    """Return a checkpoint filename without directory."""
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(
        name,
        batch_size,
        learning_rate,
        epoch,
    )
    return path


def normalize_label(labels):
    """Normalize binary labels to float 0/1.

    Handles the case where a mini-batch contains only one class, avoiding
    division by zero.
    """
    max_val = torch.max(labels)
    min_val = torch.min(labels)

    if max_val == min_val:
        normalized = torch.zeros_like(labels, dtype=torch.float32)
        if max_val != 0:
            normalized.fill_(1.0)
        return normalized

    normalized = (labels - min_val).float() / (max_val - min_val)
    return normalized


def evaluate(net, loader, criterion, use_cuda=False):
    """Evaluate a binary model on a validation loader.

    Returns:
        err: Average classification error over the loader.
        loss: Average loss over the loader.
    """
    net.eval()

    if use_cuda and torch.cuda.is_available():
        net = net.cuda()

    device = next(net.parameters()).device

    total_loss = 0.0
    total_err = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            labels = normalize_label(labels)
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1)

            outputs = net(inputs).view(-1)
            loss = criterion(outputs, labels)

            preds = (outputs > 0.0).long()
            total_err += (preds != labels.long()).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    net.train()

    if total_samples == 0:
        return 0.0, 0.0

    err = total_err / total_samples
    loss = total_loss / total_samples
    return err, loss


def plot_training_curve(path):
    """Plot training/validation error and loss curves from CSV files."""
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))

    n = len(train_err)

    plt.title("Train vs Validation Error")
    plt.plot(range(1, n + 1), train_err, label="Train")
    plt.plot(range(1, n + 1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc="best")
    plt.show()

    plt.title("Train vs Validation Loss")
    plt.plot(range(1, n + 1), train_loss, label="Train")
    plt.plot(range(1, n + 1), val_loss, label="Validation")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
