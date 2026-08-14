"""
MVTec AD-specific manifest building.

MVTec AD's directory layout differs fundamentally from a standard
classification dataset: train/ contains ONLY 'good' images (it was designed
for unsupervised anomaly detection, not supervised classification), and all
defective images live under test/<defect_type>/, split across multiple named
defect subtypes per category. This module pools all of that into a single
binary (good/defective) manifest per category, matching the classification
setup used elsewhere in this project — see docs/dataset_description.md for
the methodology note on this adaptation.
"""
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_manifest_from_mvtec_category(raw_dir: str | Path, category: str) -> pd.DataFrame:
    """Build a binary classification manifest for one MVTec AD category.

    Pools train/good/, test/good/, and all test/<defect_type>/ images into a
    single manifest with binary labels. Preserves the specific defect_type
    and, for defective images, the path to its pixel-level ground truth mask
    (unused for classification, kept for a future segmentation stretch goal).

    Args:
        raw_dir: Root MVTec AD directory (e.g. "data/raw/mvtec_ad").
        category: Category name (e.g. "bottle", "metal_nut", "screw").

    Returns:
        DataFrame with columns: [image_path, label, defect_type, category, mask_path].
        label: 0 = good, 1 = defective.
        mask_path: None for good images; path to ground_truth mask for defective ones.
    """
    category_dir = Path(raw_dir) / category
    records = []

    # Pool 'good' images from BOTH train/ and test/ splits — MVTec AD's own
    # train/test split is for anomaly detection, not used here; our own
    # stratified_split() will create the real train/val/test partition.
    for good_dir in [category_dir / "train" / "good", category_dir / "test" / "good"]:
        if not good_dir.exists():
            logger.warning("Expected directory not found: %s", good_dir)
            continue
        for img_path in good_dir.glob("*.png"):
            records.append({
                "image_path": str(img_path),
                "label": 0,
                "defect_type": "good",
                "category": category,
                "mask_path": None,
            })

    # Pool all defective images across every defect_type subdirectory under test/.
    test_dir = category_dir / "test"
    if test_dir.exists():
        defect_type_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
        for defect_dir in defect_type_dirs:
            defect_type = defect_dir.name
            mask_dir = category_dir / "ground_truth" / defect_type

            for img_path in defect_dir.glob("*.png"):
                mask_path = mask_dir / f"{img_path.stem}_mask.png"
                records.append({
                    "image_path": str(img_path),
                    "label": 1,
                    "defect_type": defect_type,
                    "category": category,
                    "mask_path": str(mask_path) if mask_path.exists() else None,
                })

    manifest = pd.DataFrame(records)
    if manifest.empty:
        logger.warning("No images found for category '%s' at %s", category, category_dir)
        return manifest

    class_counts = manifest["label"].value_counts()
    logger.info(
        "Category '%s': %d total (%d good, %d defective across %d defect types)",
        category, len(manifest), class_counts.get(0, 0), class_counts.get(1, 0),
        manifest[manifest["label"] == 1]["defect_type"].nunique(),
    )

    return manifest


def build_combined_manifest(raw_dir: str | Path, categories: list[str]) -> pd.DataFrame:
    """Build and concatenate manifests across multiple MVTec AD categories.

    Args:
        raw_dir: Root MVTec AD directory.
        categories: List of category names to include (e.g. from configs/dataset_mvtec.yaml).

    Returns:
        Combined DataFrame across all categories, with a 'category' column
        preserved so per-category analysis remains possible downstream.
    """
    all_manifests = [build_manifest_from_mvtec_category(raw_dir, cat) for cat in categories]
    all_manifests = [m for m in all_manifests if not m.empty]

    if not all_manifests:
        raise ValueError(f"No data found for any category in {categories} under {raw_dir}")

    combined = pd.concat(all_manifests, ignore_index=True)
    logger.info("Combined manifest: %d total images across %d categories", len(combined), len(categories))
    return combined


def stratified_split_per_category(
    manifest: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply stratified_split independently WITHIN each category, then recombine.

    Splitting per-category (rather than on the pooled dataframe directly)
    guarantees each category is proportionally represented in every split —
    important since categories differ in size and defect-type diversity, and
    we don't want e.g. 'screw' (5 defect types) to accidentally dominate the
    test set while 'bottle' underrepresents it.

    Args:
        manifest: Combined manifest with 'label' and 'category' columns.
        train_frac, val_frac, test_frac: Split proportions, must sum to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        Manifest with an added 'split' column, category-balanced across splits.
    """
    from src.data.preprocessing import stratified_split

    per_category_splits = []
    for category in manifest["category"].unique():
        category_df = manifest[manifest["category"] == category].reset_index(drop=True)
        split_df = stratified_split(category_df, train_frac, val_frac, test_frac, seed=seed)
        per_category_splits.append(split_df)

    return pd.concat(per_category_splits, ignore_index=True)