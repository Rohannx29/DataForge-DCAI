"""
Model evaluation.

Computes the metric suite defined in configs/model_config.yaml. F1/AUROC/Recall
are prioritized over raw accuracy throughout this project because the datasets
are imbalanced (defects are the minority class) — see docs/architecture.md.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str = "cuda") -> dict[str, float]:
    """Run inference over a DataLoader and compute the standard metric suite.

    Args:
        model: Trained (or in-training) model.
        data_loader: DataLoader to evaluate on (val or test split).
        device: "cuda" or "cpu".

    Returns:
        Dictionary with keys: loss, accuracy, precision, recall, f1, auroc.
    """
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # positive-class probability (binary case)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = {
        "loss": total_loss / len(data_loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }

    try:
        metrics["auroc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Occurs if a batch/split contains only one class — log and continue.
        logger.warning("AUROC undefined for this evaluation set (single-class batch)")
        metrics["auroc"] = float("nan")

    return metrics


def get_predicted_probabilities(model: nn.Module, data_loader: DataLoader, device: str = "cuda") -> np.ndarray:
    """Return full predicted-probability matrix, used by cleanlab noise detection.

    Args:
        model: Trained model.
        data_loader: DataLoader to run inference on (should NOT be shuffled).
        device: "cuda" or "cpu".

    Returns:
        Array of shape (n_samples, n_classes) with softmax probabilities.
    """
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)
