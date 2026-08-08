"""
Training loop.

A single, condition-agnostic training function used identically for every
experimental condition. Which manifest/dataset is passed in is the ONLY thing
that should change between baseline / cleaned / noise_corrected / dcai_improved
runs — this function's internals must not be modified per-condition.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    """Tracks per-epoch metrics for later plotting / comparison across conditions."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str = "cuda",
    early_stopping_patience: int = 5,
) -> tuple[nn.Module, TrainingHistory]:
    """Train a model with early stopping on validation F1.

    Args:
        model: Model to train (from src.models.architectures.build_model).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Maximum number of training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay (L2 regularization).
        device: "cuda" or "cpu".
        early_stopping_patience: Stop if val_f1 doesn't improve for this many epochs.

    Returns:
        Tuple of (trained model with best weights loaded, TrainingHistory).
    """
    from src.models.evaluate import evaluate_model  # local import avoids circular dependency

    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = TrainingHistory()
    patience_counter = 0
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        epoch_loss /= len(train_loader.dataset)
        scheduler.step()

        val_metrics = evaluate_model(model, val_loader, device=device)
        history.train_loss.append(epoch_loss)
        history.val_loss.append(val_metrics["loss"])
        history.val_f1.append(val_metrics["f1"])

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f",
            epoch + 1, epochs, epoch_loss, val_metrics["loss"], val_metrics["f1"],
        )

        if val_metrics["f1"] > history.best_val_f1:
            history.best_val_f1 = val_metrics["f1"]
            history.best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history
