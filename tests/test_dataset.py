"""Unit tests for src.data.dataset.DefectDataset."""
import pandas as pd
import pytest

from src.data.dataset import DefectDataset


@pytest.fixture
def sample_manifest(tmp_path):
    """Create a minimal manifest CSV with tiny dummy images for testing."""
    from PIL import Image

    rows = []
    for i in range(4):
        img_path = tmp_path / f"img_{i}.png"
        Image.new("RGB", (32, 32), color=(i * 10, i * 10, i * 10)).save(img_path)
        rows.append({
            "image_path": str(img_path),
            "label": i % 2,
            "split": "train" if i < 3 else "val",
        })

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_dataset_length(sample_manifest):
    dataset = DefectDataset(sample_manifest, split="train")
    assert len(dataset) == 3


def test_dataset_getitem_returns_image_and_label(sample_manifest):
    dataset = DefectDataset(sample_manifest, split="train")
    image, label = dataset[0]
    assert label in (0, 1)
    assert image is not None


def test_dataset_raises_on_empty_split(sample_manifest):
    with pytest.raises(ValueError):
        DefectDataset(sample_manifest, split="test")  # no 'test' rows in fixture


def test_class_distribution(sample_manifest):
    dataset = DefectDataset(sample_manifest, split="train")
    distribution = dataset.class_distribution()
    assert distribution.sum() == 3
