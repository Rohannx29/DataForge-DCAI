"""
PyTorch Dataset classes.

Defines the dataset abstraction used identically across all four experimental
conditions. The DEFECT LABELS a given instance returns depend on which
label-quality stage (raw / cleaned / noise_corrected) produced its manifest
file — the Dataset class itself does not change.
"""
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DefectDataset(Dataset):
    """Generic image classification dataset driven by a manifest CSV.

    The manifest (produced by src/data/preprocessing.py or src/labels/*.py)
    contains columns: [image_path, label, split]. Using a manifest file rather
    than scanning directories directly means every experimental condition
    (baseline/cleaned/noise_corrected/dcai_improved) can point to a different
    manifest while reusing this exact same Dataset implementation.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        transform: Callable | None = None,
    ) -> None:
        """
        Args:
            manifest_path: Path to a CSV with columns [image_path, label, split].
            split: Which split to load ("train", "val", "test").
            transform: Optional albumentations/torchvision transform pipeline.
        """
        manifest = pd.read_csv(manifest_path)
        self.data = manifest[manifest["split"] == split].reset_index(drop=True)
        self.transform = transform

        if self.data.empty:
            raise ValueError(f"No rows found for split='{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image=image)["image"] if hasattr(self.transform, "processors") \
                else self.transform(image)

        return image, label

    def class_distribution(self) -> pd.Series:
        """Return label counts — used by src/dcai/class_balance.py to detect imbalance."""
        return self.data["label"].value_counts()
