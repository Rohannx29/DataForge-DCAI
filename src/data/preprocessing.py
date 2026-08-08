"""
Data preprocessing.

Builds the initial manifest (image_path, label, split) from a raw directory
structure, and applies standard preprocessing (resize, normalization stats
computation). This is condition-agnostic — it runs once on raw data; the
label-quality and DCAI stages operate on the manifest it produces.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_manifest_from_directory(
    raw_dir: str | Path,
    class_map: dict[str, int],
) -> pd.DataFrame:
    """Build an (image_path, label) manifest from a class-per-subdirectory layout.

    Expects: raw_dir/<class_name>/*.jpg

    Args:
        raw_dir: Root directory containing one subdirectory per class.
        class_map: Mapping from subdirectory name to integer label.

    Returns:
        DataFrame with columns [image_path, label].
    """
    raw_dir = Path(raw_dir)
    records = []

    for class_name, label in class_map.items():
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            logger.warning("Expected class directory not found: %s", class_dir)
            continue
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                records.append({"image_path": str(img_path), "label": label})

    manifest = pd.DataFrame(records)
    logger.info("Built manifest with %d entries across %d classes", len(manifest), len(class_map))
    return manifest


def stratified_split(
    manifest: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Add a stratified 'split' column to the manifest, preserving class ratios.

    Args:
        manifest: DataFrame with a 'label' column.
        train_frac, val_frac, test_frac: Split proportions, must sum to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        Manifest with an added 'split' column ('train'/'val'/'test').
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    train_df, temp_df = train_test_split(
        manifest, train_size=train_frac, stratify=manifest["label"], random_state=seed
    )
    relative_val = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, train_size=relative_val, stratify=temp_df["label"], random_state=seed
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    result = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df)
    )
    return result


def compute_normalization_stats(manifest: pd.DataFrame, sample_size: int = 500) -> tuple[list[float], list[float]]:
    """Compute per-channel mean/std for normalization from a sample of training images.

    Args:
        manifest: Manifest DataFrame with 'image_path' and 'split' columns.
        sample_size: Number of training images to sample for the estimate.

    Returns:
        (mean, std) as 3-element lists (RGB).
    """
    train_paths = manifest[manifest["split"] == "train"]["image_path"].sample(
        n=min(sample_size, (manifest["split"] == "train").sum()), random_state=42
    )

    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    n_pixels = 0

    for path in train_paths:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
        n_pixels += img.shape[0] * img.shape[1]

    mean = (pixel_sum / n_pixels).tolist()
    std = np.sqrt(pixel_sq_sum / n_pixels - np.array(mean) ** 2).tolist()

    logger.info("Computed normalization stats -> mean=%s, std=%s", mean, std)
    return mean, std
