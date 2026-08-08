"""
Dataset validation.

Checks structural/technical data quality issues BEFORE any modeling begins:
corrupt images, wrong dimensions, unreadable files, duplicate images, missing
labels. This is distinct from label-quality auditing (src/labels/) which
assumes images are technically valid and instead questions whether the
assigned class is correct.
"""
from dataclasses import dataclass, field
from pathlib import Path

import hashlib

from PIL import Image, UnidentifiedImageError

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Summary of dataset validation results."""

    total_files: int = 0
    corrupt_files: list[str] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.corrupt_files or self.unreadable_files)

    def summary(self) -> str:
        return (
            f"Validated {self.total_files} files | "
            f"Corrupt: {len(self.corrupt_files)} | "
            f"Duplicate groups: {len(self.duplicate_groups)} | "
            f"Unreadable: {len(self.unreadable_files)}"
        )


def validate_image_integrity(image_dir: str | Path) -> ValidationReport:
    """Scan a directory of images for corruption/unreadability.

    Args:
        image_dir: Directory containing images to validate (recursive).

    Returns:
        A ValidationReport summarizing findings.
    """
    image_dir = Path(image_dir)
    report = ValidationReport()

    image_paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    report.total_files = len(image_paths)

    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as e:
            logger.warning("Corrupt/unreadable image: %s (%s)", path, e)
            report.corrupt_files.append(str(path))

    return report


def find_duplicate_images(image_dir: str | Path) -> list[list[str]]:
    """Find exact-duplicate images via content hashing.

    Duplicate defect images across train/test splits would cause data leakage
    and inflate reported metrics — this check is run before any split occurs.

    Args:
        image_dir: Directory containing images to check.

    Returns:
        List of duplicate groups, each a list of file paths sharing a hash.
    """
    image_dir = Path(image_dir)
    hashes: dict[str, list[str]] = {}

    for path in image_dir.rglob("*"):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()
        hashes.setdefault(file_hash, []).append(str(path))

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    if duplicate_groups:
        logger.warning("Found %d duplicate image groups", len(duplicate_groups))

    return duplicate_groups
