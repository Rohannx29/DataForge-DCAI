"""
Class balance analysis and data-level rebalancing strategies.

Provides diagnostics (imbalance ratio, minority class identification) that
feed into src/dcai/augmentation.py, and simple resampling utilities as a
baseline comparison against targeted augmentation.
"""
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_imbalance_ratio(manifest: pd.DataFrame) -> dict:
    """Compute class counts and imbalance ratio for a manifest.

    Args:
        manifest: DataFrame with a 'label' column.

    Returns:
        Dict with 'counts' (Series), 'majority_class', 'minority_class', 'ratio'.
    """
    counts = manifest["label"].value_counts()
    result = {
        "counts": counts,
        "majority_class": counts.idxmax(),
        "minority_class": counts.idxmin(),
        "ratio": counts.max() / counts.min(),
    }
    logger.info(
        "Class imbalance ratio: %.2f (majority=%s: %d, minority=%s: %d)",
        result["ratio"], result["majority_class"], counts.max(), result["minority_class"], counts.min(),
    )
    return result


def naive_oversample(manifest: pd.DataFrame, target_label: int, target_count: int, seed: int = 42) -> pd.DataFrame:
    """Duplicate minority-class rows (naive oversampling) as a comparison baseline.

    NOTE: This is intentionally the "weak baseline" data-level intervention —
    the project's DCAI contribution is showing that TARGETED AUGMENTATION
    (src/dcai/augmentation.py) outperforms this naive approach, not just that
    any oversampling helps.

    Args:
        manifest: Training manifest.
        target_label: Class to oversample.
        target_count: Desired sample count for the class after oversampling.
        seed: Random seed for reproducible sampling.

    Returns:
        Manifest with duplicated minority-class rows appended.
    """
    minority_rows = manifest[manifest["label"] == target_label]
    n_needed = max(0, target_count - len(minority_rows))

    if n_needed == 0:
        return manifest

    duplicated = minority_rows.sample(n=n_needed, replace=True, random_state=seed)
    logger.info("Naively duplicated %d rows for class %s", n_needed, target_label)
    return pd.concat([manifest, duplicated]).reset_index(drop=True)
