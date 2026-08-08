"""
Dataset acquisition.

Handles downloading and organizing raw datasets into data/raw/<dataset_name>/.
Kept separate from preprocessing: acquisition should be idempotent and never
modify pixel data or labels — only fetch and arrange files on disk.
"""
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


CASTING_KAGGLE_HANDLE = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
CASTING_EXPECTED_MIN_FILES = 6000  # dataset has ~7,000 images; used as a download sanity check


def download_casting_dataset(dest_dir: str | Path = "data/raw/casting") -> Path:
    """Download the Casting Product Image Dataset via kagglehub and copy it into dest_dir.

    Requires Kaggle API credentials at ~/.kaggle/kaggle.json (see docs/setup.md).
    kagglehub caches the download outside the project (~/.cache/kagglehub) and
    is idempotent — re-running this is safe and won't re-download if already cached.

    Args:
        dest_dir: Destination directory to copy the dataset into (data/raw/casting).

    Returns:
        Path to dest_dir containing the dataset.

    Raises:
        RuntimeError: If the download completes but the file count looks wrong
            (likely a partial/corrupt download) — fails loudly rather than
            letting a bad download silently propagate into EDA/training.
    """
    import shutil

    import kagglehub

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading '%s' via kagglehub...", CASTING_KAGGLE_HANDLE)
    cached_path = Path(kagglehub.dataset_download(CASTING_KAGGLE_HANDLE))
    logger.info("kagglehub cached download at: %s", cached_path)

    # kagglehub downloads to its own cache dir; copy into our project-local data/raw/
    # so the rest of the pipeline (configs point at data/raw/casting) works unchanged.
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info("dest_dir already populated — skipping copy (idempotent)")
    else:
        shutil.copytree(cached_path, dest_dir, dirs_exist_ok=True)
        logger.info("Copied dataset into project at %s", dest_dir)

    if not verify_download_integrity(dest_dir, expected_min_files=CASTING_EXPECTED_MIN_FILES):
        raise RuntimeError(
            f"Download integrity check failed for {dest_dir} — expected at least "
            f"{CASTING_EXPECTED_MIN_FILES} files. The download may be partial or corrupt. "
            f"Try deleting {dest_dir} and re-running."
        )

    logger.info("Casting dataset ready at %s", dest_dir)
    return dest_dir


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