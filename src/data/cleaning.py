"""
Data cleaning.

Applies fixes identified by src/data/validation.py: removes corrupt files,
resolves duplicates, standardizes image formats. Produces the "cleaned"
experimental condition manifest, distinct from raw and from noise_corrected
(which additionally fixes LABEL errors, not just file-level errors).
"""
from pathlib import Path

import pandas as pd

from src.data.validation import ValidationReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


def remove_corrupt_files(manifest: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """Drop rows referencing corrupt/unreadable images.

    Args:
        manifest: DataFrame with an 'image_path' column.
        report: ValidationReport produced by validate_image_integrity().

    Returns:
        Filtered manifest with corrupt entries removed.
    """
    before = len(manifest)
    manifest = manifest[~manifest["image_path"].isin(report.corrupt_files + report.unreadable_files)]
    logger.info("Removed %d corrupt/unreadable entries (%d -> %d)", before - len(manifest), before, len(manifest))
    return manifest.reset_index(drop=True)


def resolve_duplicates(manifest: pd.DataFrame, duplicate_groups: list[list[str]], strategy: str = "keep_first") -> pd.DataFrame:
    """Remove duplicate images, keeping one representative per group.

    Args:
        manifest: DataFrame with an 'image_path' column.
        duplicate_groups: Groups of duplicate file paths from find_duplicate_images().
        strategy: How to choose which duplicate to keep. Currently only "keep_first".

    Returns:
        Manifest with duplicates resolved.
    """
    if strategy != "keep_first":
        raise NotImplementedError(f"Duplicate resolution strategy '{strategy}' not implemented")

    paths_to_drop = []
    for group in duplicate_groups:
        paths_to_drop.extend(group[1:])  # keep the first, drop the rest

    before = len(manifest)
    manifest = manifest[~manifest["image_path"].isin(paths_to_drop)]
    logger.info("Removed %d duplicate entries (%d -> %d)", before - len(manifest), before, len(manifest))
    return manifest.reset_index(drop=True)


def save_cleaned_manifest(manifest: pd.DataFrame, output_path: str | Path) -> None:
    """Persist the cleaned manifest — this becomes the input for the 'cleaned' condition.

    Args:
        manifest: Cleaned DataFrame.
        output_path: Destination CSV path, e.g. data/processed/casting/manifest_cleaned.csv
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    logger.info("Saved cleaned manifest (%d rows) to %s", len(manifest), output_path)
