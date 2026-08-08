"""Unit tests for src.data.cleaning."""
import pandas as pd

from src.data.cleaning import resolve_duplicates
from src.data.validation import ValidationReport
from src.data.cleaning import remove_corrupt_files


def test_remove_corrupt_files():
    manifest = pd.DataFrame({
        "image_path": ["a.png", "b.png", "c.png"],
        "label": [0, 1, 0],
    })
    report = ValidationReport(corrupt_files=["b.png"])

    result = remove_corrupt_files(manifest, report)

    assert len(result) == 2
    assert "b.png" not in result["image_path"].values


def test_resolve_duplicates_keeps_first():
    manifest = pd.DataFrame({
        "image_path": ["a.png", "b.png", "c.png", "d.png"],
        "label": [0, 0, 1, 1],
    })
    duplicate_groups = [["a.png", "b.png"]]

    result = resolve_duplicates(manifest, duplicate_groups)

    assert len(result) == 3
    assert "a.png" in result["image_path"].values
    assert "b.png" not in result["image_path"].values
