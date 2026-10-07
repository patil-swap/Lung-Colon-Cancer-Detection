import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utilities import utility_funcs


def train_net(
    net,
    train_loader,
    val_loader,
    batch_size=150,
    learning_rate=0.005,
    num_epochs=6,
    use_cuda=False,
    checkpoint_dir="checkpoint_files",
    optimizer_name="adamw",
    weight_decay=1e-4,
    pos_weight=None,
    early_stopping_patience=5,
    seed=42,
):
    """Train a binary CNN and save checkpoints plus training curves.

    This version intentionally does not change model architecture. It improves
    training stability and evaluation correctness.

    Args:
        net: PyTorch model with a scalar binary output.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        batch_size: Used for checkpoint naming.
        learning_rate: Used for checkpoint naming and optimizer.
        num_epochs: Maximum number of epochs.
        use_cuda: Use CUDA when available.
        checkpoint_dir: Directory for saved checkpoints and CSVs.
        optimizer_name: One of "adamw", "adam", or "sgd".
        weight_decay: Weight decay applied to optimizer.
        pos_weight: Optional positive-class weight for BCEWithLogitsLoss.
        early_stopping_patience: Stop after this many epochs without val error
            improvement. Pass None to disable.
        seed: Random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    net = net.to(device)

    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            net.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = getattr(net, "name", net.__class__.__name__)
    base_name = f"model_{model_name}_bs{batch_size}_lr{learning_rate}"
    best_model_path = os.path.join(checkpoint_dir, f"{base_name}_best")

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    best_val_err = float("inf")
    epochs_no_improve = 0
    actual_epochs = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        net.train()

        total_train_loss = 0.0
        total_train_err = 0.0
        total_samples = 0

        for imgs, labels in train_loader:
            labels = utility_funcs.normalize_label(labels)
            imgs = imgs.to(device)
            labels = labels.to(device).float().view(-1)

            optimizer.zero_grad()

            outputs = net(imgs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.0).long()
            total_train_err += (preds != labels.long()).sum().item()
            total_train_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        if total_samples > 0:
            train_err[epoch] = total_train_err / total_samples
            train_loss[epoch] = total_train_loss / total_samples
        else:
            train_err[epoch] = 0.0
            train_loss[epoch] = 0.0

        val_err[epoch], val_loss[epoch] = utility_funcs.evaluate(
            net,
            val_loader,
            criterion,
            use_cuda=use_cuda,
        )

        scheduler.step(val_loss[epoch])

        print(
            (
                "Epoch {}: Train err: {:.4f}, Train loss: {:.4f} | "
                "Validation err: {:.4f}, Validation loss: {:.4f}"
            ).format(
                epoch + 1,
                train_err[epoch],
                train_loss[epoch],
                val_err[epoch],
                val_loss[epoch],
            )
        )

        epoch_model_path = os.path.join(checkpoint_dir, f"{base_name}_epoch{epoch}")
        torch.save(net.state_dict(), epoch_model_path)

        actual_epochs = epoch + 1

        if val_err[epoch] < best_val_err:
            best_val_err = val_err[epoch]
            epochs_no_improve = 0
            torch.save(net.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path}")
        else:
            epochs_no_improve += 1

        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    elapsed_time = time.time() - start_time
    print("Finished Training")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")

    train_err = train_err[:actual_epochs]
    train_loss = train_loss[:actual_epochs]
    val_err = val_err[:actual_epochs]
    val_loss = val_loss[:actual_epochs]

    np.savetxt(os.path.join(checkpoint_dir, f"{base_name}_train_err.csv"), train_err)
    np.savetxt(os.path.join(checkpoint_dir, f"{base_name}_train_loss.csv"), train_loss)
    np.savetxt(os.path.join(checkpoint_dir, f"{base_name}_val_err.csv"), val_err)
    np.savetxt(os.path.join(checkpoint_dir, f"{base_name}_val_loss.csv"), val_loss)

    print(f"Best validation error: {best_val_err:.4f}")
    print(f"Best model saved at: {best_model_path}")
