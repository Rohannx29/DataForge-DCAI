"""
Label noise correction.

Applies corrective actions to samples flagged by noise_detection.py: either
removal or relabeling. Produces the manifest used by the 'noise_corrected'
experimental condition.
"""
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def remove_flagged_samples(manifest: pd.DataFrame, issue_indices) -> pd.DataFrame:
    """Drop samples flagged as likely mislabeled.

    The simpler and generally recommended correction strategy: removing
    ambiguous samples is safer than guessing a "corrected" label, since an
    incorrect relabel introduces a different kind of noise. Document this
    trade-off explicitly in the report's Methodology section.

    Args:
        manifest: Full manifest DataFrame.
        issue_indices: Indices flagged by detect_label_issues().

    Returns:
        Manifest with flagged rows removed.
    """
    before = len(manifest)
    cleaned = manifest.drop(index=issue_indices).reset_index(drop=True)
    logger.info("Removed %d flagged samples (%d -> %d)", before - len(cleaned), before, len(cleaned))
    return cleaned


def relabel_flagged_samples(
    manifest: pd.DataFrame,
    issue_indices,
    pred_probs,
) -> pd.DataFrame:
    """Relabel flagged samples with the model's most confident predicted class.

    Riskier than removal — only recommended when pred_probs confidence for the
    flagged sample's top predicted class is very high (e.g. > 0.9), otherwise
    prefer remove_flagged_samples(). Always report how many samples were
    relabeled vs. removed in the final report for transparency.

    Args:
        manifest: Full manifest DataFrame.
        issue_indices: Indices flagged by detect_label_issues().
        pred_probs: Out-of-sample predicted probabilities, shape (n_samples, n_classes).

    Returns:
        Manifest with flagged rows relabeled to argmax(pred_probs).
    """
    import numpy as np

    manifest = manifest.copy()
    new_labels = np.argmax(pred_probs[issue_indices], axis=1)
    manifest.loc[issue_indices, "label"] = new_labels
    logger.info("Relabeled %d flagged samples", len(issue_indices))
    return manifest
