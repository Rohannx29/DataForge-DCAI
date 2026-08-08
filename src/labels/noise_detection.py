"""
Label noise detection.

Wraps `cleanlab`'s confident-learning approach to identify samples whose
assigned label likely does not match the true class, using out-of-sample
predicted probabilities from cross-validation. This is the technical core
of the "noise_corrected" experimental condition.

Reference: Northcutt, Jiang, Chuang (2021) "Confident Learning: Estimating
Uncertainty in Dataset Labels", JAIR.
"""
import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues

from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_label_issues(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    return_indices_ranked_by: str = "self_confidence",
) -> np.ndarray:
    """Identify indices of likely mislabeled samples using confident learning.

    Args:
        labels: Array of shape (n_samples,) with given (possibly noisy) labels.
        pred_probs: Array of shape (n_samples, n_classes) with OUT-OF-SAMPLE
            predicted probabilities (e.g. from k-fold cross-validation —
            using in-sample probabilities will bias results).
        return_indices_ranked_by: Ranking method for returned issue indices.
            Options: "self_confidence", "normalized_margin", "confidence_weighted_entropy".

    Returns:
        Array of sample indices flagged as likely mislabeled, ranked by
        estimated severity (most likely issue first).
    """
    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by=return_indices_ranked_by,
    )
    logger.info(
        "Flagged %d / %d samples (%.2f%%) as likely label issues",
        len(issue_indices), len(labels), 100 * len(issue_indices) / len(labels),
    )
    return issue_indices


def get_out_of_sample_predictions(model_fn, manifest: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    """Compute out-of-sample predicted probabilities via k-fold cross-validation.

    Required input for detect_label_issues() — using probabilities from a
    model trained on the same data it predicts would bias noise detection
    toward finding fewer issues than actually exist.

    Args:
        model_fn: Callable that returns a fresh, untrained model instance
            compatible with the training pipeline in src/models/train.py.
        manifest: Training manifest with 'image_path' and 'label' columns.
        n_folds: Number of cross-validation folds.

    Returns:
        Array of shape (n_samples, n_classes) with out-of-sample predicted probabilities.
    """
    # TODO: implement k-fold training loop using src.models.train.train_model,
    # collecting held-out predictions for each fold's validation split.
    raise NotImplementedError(
        "Implement k-fold OOS prediction collection once src/models/train.py baseline is complete."
    )


def inject_synthetic_label_noise(
    manifest: pd.DataFrame,
    noise_fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Deliberately corrupt a fraction of labels to test noise-detection recall/precision.

    Used to validate that detect_label_issues() actually finds injected errors
    before trusting it on real (unknown ground-truth) label noise.

    Args:
        manifest: Original manifest with a 'label' column.
        noise_fraction: Fraction of samples to relabel with a random wrong class.
        seed: Random seed for reproducibility.

    Returns:
        Manifest with a 'label' column corrupted and an added 'is_synthetically_noisy'
        boolean column for later evaluation of detection performance.
    """
    rng = np.random.RandomState(seed)
    manifest = manifest.copy()
    manifest["is_synthetically_noisy"] = False

    n_noisy = int(len(manifest) * noise_fraction)
    noisy_idx = rng.choice(manifest.index, size=n_noisy, replace=False)
    classes = manifest["label"].unique()

    for idx in noisy_idx:
        true_label = manifest.at[idx, "label"]
        wrong_choices = [c for c in classes if c != true_label]
        manifest.at[idx, "label"] = rng.choice(wrong_choices)
        manifest.at[idx, "is_synthetically_noisy"] = True

    logger.info("Injected synthetic noise into %d / %d samples", n_noisy, len(manifest))
    return manifest
