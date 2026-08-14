"""
Label noise detection.

Wraps `cleanlab`'s confident-learning approach to identify samples whose
assigned label likely does not match the true class, using out-of-sample
predicted probabilities from cross-validation. This is the technical core
of the "noise_corrected" experimental condition.

Reference: Northcutt, Jiang, Chuang (2021) "Confident Learning: Estimating
Uncertainty in Dataset Labels", JAIR.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils.logger import get_logger

logger = get_logger(__name__)

_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def detect_label_issues(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    return_indices_ranked_by: str = "self_confidence",
    n_jobs: int = 1,
) -> np.ndarray:
    """Identify indices of likely mislabeled samples using confident learning.

    Args:
        labels: Array of shape (n_samples,) with given (possibly noisy) labels.
        pred_probs: Array of shape (n_samples, n_classes) with OUT-OF-SAMPLE
            predicted probabilities (e.g. from k-fold cross-validation —
            using in-sample probabilities will bias results).
        return_indices_ranked_by: Ranking method for returned issue indices.
            Options: "self_confidence", "normalized_margin", "confidence_weighted_entropy".
        n_jobs: Number of parallel processes cleanlab uses internally.
            FORCED TO 1 by default — cleanlab's multiprocessing triggers a
            known Windows + Python 3.12 + torch._dynamo import bug (subprocess
            re-imports of torch cause a runaway inspect.signature() recursion,
            manifesting as MemoryError). Single-process is slightly slower but
            stable on Windows. Safe to raise on Linux/Mac if needed.

    Returns:
        Array of sample indices (positional, into the arrays passed in) flagged
        as likely mislabeled, ranked by estimated severity (most likely issue first).
    """
    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by=return_indices_ranked_by,
        n_jobs=n_jobs,
    )
    logger.info(
        "Flagged %d / %d samples (%.2f%%) as likely label issues",
        len(issue_indices), len(labels), 100 * len(issue_indices) / len(labels),
    )
    return issue_indices


def get_out_of_sample_predictions(
    manifest: pd.DataFrame,
    model_config: dict,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute out-of-sample predicted probabilities via k-fold cross-validation.

    Runs ONLY over the manifest's 'train' split rows — val/test are held out
    of this process entirely, since they must remain trustworthy ground truth
    for evaluation, not subject to noise correction.

    NOTE ON COST: this trains n_folds separate models from scratch (same
    architecture/hyperparameters as the main training config), so it costs
    roughly n_folds x a normal training run. On the Casting dataset with a
    GPU, expect several minutes total, not seconds.

    Args:
        manifest: Full manifest DataFrame with 'image_path', 'label', 'split' columns.
        model_config: Merged config dict (base_config + model_config) — reuses
            the SAME architecture/hyperparameters as the main training pipeline,
            per the project's controlled-experiment principle.
        n_folds: Number of cross-validation folds.
        seed: Random seed for fold splitting and model init.

    Returns:
        Tuple of (original_indices, pred_probs):
            original_indices: positional indices into the train-split subset
                of `manifest` (i.e. index into manifest[manifest.split=='train']
                .reset_index(drop=True)), in the SAME order as pred_probs.
            pred_probs: array of shape (n_train_samples, n_classes) with
                out-of-sample predicted probabilities.
    """
    from src.models.architectures import build_model
    from src.models.evaluate import get_predicted_probabilities
    from src.models.train import train_model
    from src.data.dataset import DefectDataset

    train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
    labels = train_rows["label"].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oos_pred_probs = np.zeros((len(train_rows), model_config["model"]["num_classes"]))

    for fold_idx, (fold_train_idx, fold_holdout_idx) in enumerate(skf.split(train_rows, labels)):
        logger.info("OOS prediction fold %d/%d (holdout size=%d)", fold_idx + 1, n_folds, len(fold_holdout_idx))

        fold_train_df = train_rows.iloc[fold_train_idx].assign(split="train")
        fold_holdout_df = train_rows.iloc[fold_holdout_idx].assign(split="val")
        fold_manifest = pd.concat([fold_train_df, fold_holdout_df]).reset_index(drop=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            fold_manifest.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            fold_train_dataset = DefectDataset(tmp_path, split="train", transform=_IMAGENET_TRANSFORM)
            fold_holdout_dataset = DefectDataset(tmp_path, split="val", transform=_IMAGENET_TRANSFORM)

            fold_train_loader = DataLoader(fold_train_dataset, batch_size=model_config["training"]["batch_size"], shuffle=True)
            fold_holdout_loader = DataLoader(fold_holdout_dataset, batch_size=model_config["training"]["batch_size"], shuffle=False)

            model = build_model(
                architecture=model_config["model"]["architecture"],
                num_classes=model_config["model"]["num_classes"],
                pretrained=model_config["model"]["pretrained"],
                freeze_backbone=model_config["model"]["freeze_backbone"],
                dropout=model_config["model"]["dropout"],
            )

            model, _ = train_model(
                model=model,
                train_loader=fold_train_loader,
                val_loader=fold_holdout_loader,
                epochs=model_config["training"]["epochs"],
                learning_rate=model_config["training"]["learning_rate"],
                weight_decay=model_config["training"]["weight_decay"],
                device=model_config["project"]["device"],
                early_stopping_patience=model_config["training"]["early_stopping_patience"],
            )

            fold_probs = get_predicted_probabilities(model, fold_holdout_loader, device=model_config["project"]["device"])
            oos_pred_probs[fold_holdout_idx] = fold_probs
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    original_indices = np.arange(len(train_rows))
    return original_indices, oos_pred_probs


def inject_synthetic_label_noise(
    manifest: pd.DataFrame,
    noise_fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Deliberately corrupt a fraction of TRAIN-split labels to test noise-detection recall/precision.

    Only touches 'train' split rows — val/test remain untouched so evaluation
    stays trustworthy. Used to validate that detect_label_issues() actually
    finds injected errors before trusting it on real (unknown ground-truth)
    label noise.

    Args:
        manifest: Original manifest with 'label' and 'split' columns.
        noise_fraction: Fraction of TRAIN samples to relabel with a random wrong class.
        seed: Random seed for reproducibility.

    Returns:
        Manifest with train-split labels corrupted and an added
        'is_synthetically_noisy' boolean column for later evaluation of
        detection performance.
    """
    rng = np.random.RandomState(seed)
    manifest = manifest.copy()
    manifest["is_synthetically_noisy"] = False

    train_mask = manifest["split"] == "train"
    train_indices = manifest[train_mask].index

    n_noisy = int(len(train_indices) * noise_fraction)
    noisy_idx = rng.choice(train_indices, size=n_noisy, replace=False)
    classes = manifest["label"].unique()

    for idx in noisy_idx:
        true_label = manifest.at[idx, "label"]
        wrong_choices = [c for c in classes if c != true_label]
        manifest.at[idx, "label"] = rng.choice(wrong_choices)
        manifest.at[idx, "is_synthetically_noisy"] = True

    logger.info("Injected synthetic noise into %d / %d train samples", n_noisy, len(train_indices))
    return manifest