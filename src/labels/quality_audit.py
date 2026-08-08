"""
Label quality auditing.

Computes descriptive label-quality metrics used both to decide WHERE to apply
noise correction, and to report on dataset quality in the paper/report
(e.g. "X% of labels flagged as likely noisy, concentrated in class Y").
"""
from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LabelQualityReport:
    """Summary statistics describing label quality for a dataset."""

    n_samples: int
    class_counts: pd.Series
    imbalance_ratio: float          # majority_count / minority_count
    n_flagged_noisy: int | None = None    # populated after noise_detection.py runs
    flagged_fraction: float | None = None

    def summary(self) -> str:
        lines = [
            f"Samples: {self.n_samples}",
            f"Class distribution:\n{self.class_counts.to_string()}",
            f"Imbalance ratio (majority:minority): {self.imbalance_ratio:.2f}",
        ]
        if self.n_flagged_noisy is not None:
            lines.append(
                f"Flagged as likely noisy: {self.n_flagged_noisy} ({self.flagged_fraction:.2%})"
            )
        return "\n".join(lines)


def audit_label_distribution(manifest: pd.DataFrame) -> LabelQualityReport:
    """Compute class distribution and imbalance ratio for a manifest.

    Args:
        manifest: DataFrame with a 'label' column.

    Returns:
        LabelQualityReport with distribution statistics.
    """
    class_counts = manifest["label"].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()

    report = LabelQualityReport(
        n_samples=len(manifest),
        class_counts=class_counts,
        imbalance_ratio=imbalance_ratio,
    )
    logger.info(report.summary())
    return report


def compute_inter_annotator_agreement(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """Compute Cohen's Kappa between two sets of annotations for the same samples.

    Relevant if a subset of the dataset is independently re-labeled to estimate
    original annotation quality (recommended for at least a 200-sample audit
    subset — see docs/dataset_description.md).

    Args:
        labels_a: First annotator's labels, indexed identically to labels_b.
        labels_b: Second annotator's labels.

    Returns:
        Cohen's Kappa score (-1 to 1; >0.8 is considered near-perfect agreement).
    """
    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(labels_a, labels_b)
    logger.info("Inter-annotator Cohen's Kappa: %.3f", kappa)
    return kappa
