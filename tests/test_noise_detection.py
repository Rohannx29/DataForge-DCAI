"""Unit tests for src.labels.noise_detection."""
import numpy as np
import pandas as pd

from src.labels.noise_detection import detect_label_issues, inject_synthetic_label_noise


def test_inject_synthetic_label_noise_marks_correct_fraction():
    manifest = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(100)],
        "label": [0] * 50 + [1] * 50,
    })

    noisy_manifest = inject_synthetic_label_noise(manifest, noise_fraction=0.1, seed=42)

    assert noisy_manifest["is_synthetically_noisy"].sum() == 10


def test_detect_label_issues_finds_obvious_noise():
    # Construct a case where predicted probabilities strongly disagree with given labels.
    labels = np.array([0, 0, 0, 1, 1, 1])
    pred_probs = np.array([
        [0.9, 0.1],
        [0.85, 0.15],
        [0.1, 0.9],   # given label 0, but model is confident it's class 1 -> likely noisy
        [0.1, 0.9],
        [0.15, 0.85],
        [0.9, 0.1],   # given label 1, but model is confident it's class 0 -> likely noisy
    ])

    issue_indices = detect_label_issues(labels, pred_probs)

    assert 2 in issue_indices
    assert 5 in issue_indices
