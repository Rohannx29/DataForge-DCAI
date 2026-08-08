"""
Outlier / anomalous-sample detection.

Identifies technically valid but statistically anomalous images (e.g. wrong
lighting conditions, camera artifacts, out-of-distribution backgrounds) that
are valid files with correct labels but may still harm training if not
reviewed. Distinct from src/data/validation.py (corrupt files) and
src/labels/noise_detection.py (wrong labels).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_feature_outliers(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    contamination: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Flag samples as feature-space outliers using Isolation Forest.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim) — penultimate-layer
            model features, or simple pixel-statistics features as a lightweight
            fallback.
        manifest: Manifest DataFrame aligned row-for-row with embeddings.
        contamination: Expected proportion of outliers in the dataset.
        seed: Random seed for reproducibility.

    Returns:
        Manifest with an added boolean 'is_outlier' column.
    """
    detector = IsolationForest(contamination=contamination, random_state=seed)
    outlier_flags = detector.fit_predict(embeddings) == -1  # -1 indicates outlier

    manifest = manifest.copy()
    manifest["is_outlier"] = outlier_flags
    logger.info("Flagged %d / %d samples as feature-space outliers", outlier_flags.sum(), len(manifest))

    return manifest
