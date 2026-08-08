"""
Targeted data augmentation.

Unlike blanket augmentation applied uniformly to all classes, this module
applies augmentation SELECTIVELY to under-represented / rare defect classes,
which is the data-centric argument: fix the imbalance in the data, don't just
reweight the loss function.
"""
import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_defect_augmentation_pipeline(image_size: tuple[int, int] = (224, 224)) -> A.Compose:
    """Augmentation pipeline for defect images.

    Transforms chosen to be defect-preserving (i.e. they must not alter or
    remove the actual defect region): geometric transforms and lighting
    variation are safe; aggressive cropping is avoided since it risks
    cropping out small defects entirely.

    Args:
        image_size: Target (height, width) for resizing.

    Returns:
        An albumentations.Compose augmentation pipeline.
    """
    return A.Compose([
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.Normalize(),
    ])


def augment_minority_class(
    manifest: pd.DataFrame,
    minority_label: int,
    target_count: int,
    output_dir: str,
    image_size: tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """Generate augmented copies of minority-class images to reach a target count.

    Args:
        manifest: Training manifest with 'image_path' and 'label' columns.
        minority_label: The class label to augment.
        target_count: Desired number of samples for this class after augmentation.
        output_dir: Directory to save augmented images.
        image_size: Target image size for augmentation.

    Returns:
        Manifest with new rows appended for the augmented images.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minority_rows = manifest[manifest["label"] == minority_label]
    current_count = len(minority_rows)
    n_needed = max(0, target_count - current_count)

    if n_needed == 0:
        logger.info("Class %s already at target count (%d); no augmentation needed", minority_label, current_count)
        return manifest

    pipeline = get_defect_augmentation_pipeline(image_size)
    new_rows = []
    rng = np.random.RandomState(42)

    for i in range(n_needed):
        source_row = minority_rows.iloc[rng.randint(len(minority_rows))]
        image = np.array(Image.open(source_row["image_path"]).convert("RGB"))
        augmented = pipeline(image=image)["image"]

        out_path = output_dir / f"aug_{minority_label}_{i}.png"
        Image.fromarray((augmented * 255).astype(np.uint8)).save(out_path)
        new_rows.append({"image_path": str(out_path), "label": minority_label, "split": "train"})

    logger.info("Generated %d augmented samples for class %s", n_needed, minority_label)
    return pd.concat([manifest, pd.DataFrame(new_rows)]).reset_index(drop=True)
