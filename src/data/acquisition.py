"""
Dataset acquisition.

Handles downloading and organizing raw datasets into data/raw/<dataset_name>/.
Kept separate from preprocessing: acquisition should be idempotent and never
modify pixel data or labels — only fetch and arrange files on disk.
"""
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_casting_dataset(dest_dir: str | Path = "data/raw/casting") -> None:
    """Download/extract the Casting Product Image Dataset.

    NOTE: Kaggle datasets require API credentials (~/.kaggle/kaggle.json).
    This function wraps the kaggle CLI; see docs/setup.md for credential setup.

    Args:
        dest_dir: Destination directory for the raw dataset.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("TODO: implement Kaggle API download for casting dataset -> %s", dest_dir)
    raise NotImplementedError(
        "Implement via kaggle API: "
        "kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product"
    )


def download_mvtec_ad(dest_dir: str | Path = "data/raw/mvtec_ad") -> None:
    """Download/extract the MVTec Anomaly Detection dataset.

    NOTE: MVTec AD requires accepting a license agreement on the official page
    before download; this cannot be fully automated. See docs/setup.md.

    Args:
        dest_dir: Destination directory for the raw dataset.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("TODO: manual download required — see docs/setup.md for MVTec AD instructions")
    raise NotImplementedError("MVTec AD requires manual download due to licensing terms.")


def verify_download_integrity(dataset_dir: str | Path, expected_min_files: int) -> bool:
    """Sanity-check that a downloaded dataset directory looks complete.

    Args:
        dataset_dir: Directory to check.
        expected_min_files: Minimum number of files expected (rough sanity check).

    Returns:
        True if the directory contains at least expected_min_files files.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return False
    file_count = sum(1 for _ in dataset_dir.rglob("*") if _.is_file())
    logger.info("Found %d files in %s (expected >= %d)", file_count, dataset_dir, expected_min_files)
    return file_count >= expected_min_files
