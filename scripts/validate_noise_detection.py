#!/usr/bin/env python
"""
Entry point: validate the label-noise detection pipeline using synthetic noise.

Casting's real labels appear near-perfectly clean (baseline model reached
test F1 = 1.0 — see docs/dataset_description.md), leaving no real noise to
detect. This script instead injects a KNOWN amount of synthetic label noise,
runs cleanlab-based detection, and reports precision/recall of the detector
against the known ground truth — validating the noise-detection machinery
itself ahead of applying it to MVTec AD in Phase 2, where results are
expected to matter more.

Usage:
    python scripts/validate_noise_detection.py \
        --manifest data/processed/casting/manifest_cleaned.csv \
        --noise-fraction 0.10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.labels.noise_detection import (
    detect_label_issues,
    get_out_of_sample_predictions,
    inject_synthetic_label_noise,
)
from src.utils.config import load_config, merge_configs
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate noise detection via synthetic noise injection")
    parser.add_argument("--manifest", required=True, help="Path to a cleaned manifest CSV")
    parser.add_argument("--noise-fraction", type=float, default=0.10, help="Fraction of train labels to corrupt")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--base-config", default="configs/base_config.yaml")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    model_cfg = load_config(args.model_config)
    config = merge_configs(base_cfg, model_cfg)
    set_seed(config["project"]["seed"])

    manifest = pd.read_csv(args.manifest)
    noisy_manifest = inject_synthetic_label_noise(manifest, noise_fraction=args.noise_fraction, seed=config["project"]["seed"])

    logger.info("Running %d-fold cross-validation to get out-of-sample predictions (this will take several minutes)...", args.n_folds)
    train_indices, pred_probs = get_out_of_sample_predictions(noisy_manifest, config, n_folds=args.n_folds, seed=config["project"]["seed"])

    train_rows = noisy_manifest[noisy_manifest["split"] == "train"].reset_index(drop=True)
    labels = train_rows["label"].values
    true_noisy_mask = train_rows["is_synthetically_noisy"].values

    flagged_indices = detect_label_issues(labels=labels, pred_probs=pred_probs)
    flagged_mask = np.zeros(len(train_rows), dtype=bool)
    flagged_mask[flagged_indices] = True

    true_positives = np.sum(flagged_mask & true_noisy_mask)
    false_positives = np.sum(flagged_mask & ~true_noisy_mask)
    false_negatives = np.sum(~flagged_mask & true_noisy_mask)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    logger.info(
        "Noise detection validation | injected=%d flagged=%d | TP=%d FP=%d FN=%d | precision=%.3f recall=%.3f",
        true_noisy_mask.sum(), flagged_mask.sum(), true_positives, false_positives, false_negatives, precision, recall,
    )


if __name__ == "__main__":
    main()