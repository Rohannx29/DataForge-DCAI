"""
Exploratory data analysis plots.

Generates the standard EDA figures referenced in the report's Dataset
Description section: class distribution, sample image grids, image
dimension/quality distributions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


def plot_class_distribution(manifest: pd.DataFrame, save_path: str | Path | None = None) -> None:
    """Bar plot of class counts, annotated with the imbalance ratio.

    Args:
        manifest: DataFrame with a 'label' column.
        save_path: If provided, saves the figure instead of/in addition to displaying it.
    """
    counts = manifest["label"].value_counts().sort_index()
    imbalance_ratio = counts.max() / counts.min()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_title(f"Class Distribution (Imbalance Ratio: {imbalance_ratio:.2f})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample Count")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(manifest: pd.DataFrame, n_samples_per_class: int = 4, save_path: str | Path | None = None) -> None:
    """Display a grid of sample images per class for visual sanity-checking.

    Args:
        manifest: DataFrame with 'image_path' and 'label' columns.
        n_samples_per_class: Number of example images to show per class.
        save_path: If provided, saves the figure.
    """
    classes = sorted(manifest["label"].unique())
    fig, axes = plt.subplots(len(classes), n_samples_per_class, figsize=(n_samples_per_class * 2, len(classes) * 2))

    for row, label in enumerate(classes):
        samples = manifest[manifest["label"] == label].sample(n=min(n_samples_per_class, (manifest["label"] == label).sum()))
        for col, (_, sample) in enumerate(samples.iterrows()):
            ax = axes[row, col] if len(classes) > 1 else axes[col]
            img = Image.open(sample["image_path"])
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"Class {label}", fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
