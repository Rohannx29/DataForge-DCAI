#!/usr/bin/env python
"""
Entry point: label quality audit and noise detection on REAL (non-synthetic) labels.

Produces the 'noise_corrected' condition manifest. Note: on Casting, this is
expected to flag very few (possibly zero) real issues, given baseline test
F1 = 1.0 — see docs/dataset_description.md. Use scripts/validate_noise_detection.py
to validate the detector itself via synthetic noise.

Usage:
    python scripts/run_label_audit.py --config configs/dataset_casting.yaml --manifest data/processed/casting/manifest_cleaned.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.labels.noise_correction import remove_flagged_samples
from src.labels.noise_detection import detect_label_issues, get_out_of_sample_predictions
from src.labels.quality_audit import audit_label_distribution
from src.utils.config import load_config, merge_configs
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit label quality and detect noisy labels")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    parser.add_argument("--manifest", required=True, help="Path to cleaned manifest CSV")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--base-config", default="configs/base_config.yaml")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    model_cfg = load_config(args.model_config)
    config = merge_configs(base_cfg, model_cfg)
    set_seed(config["project"]["seed"])

    manifest = pd.read_csv(args.manifest)

    report = audit_label_distribution(manifest)
    logger.info("\n%s", report.summary())

    logger.info("Computing out-of-sample predictions for confident learning (%d folds, this will take a while)...", args.n_folds)
    train_indices, pred_probs = get_out_of_sample_predictions(manifest, config, n_folds=args.n_folds, seed=config["project"]["seed"])

    train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
    issue_indices = detect_label_issues(labels=train_rows["label"].values, pred_probs=pred_probs)

    corrected_train = remove_flagged_samples(train_rows, issue_indices)
    non_train_rows = manifest[manifest["split"] != "train"]
    corrected = pd.concat([corrected_train, non_train_rows]).reset_index(drop=True)

    output_path = args.manifest.replace("manifest_cleaned.csv", "manifest_noise_corrected.csv")
    corrected.to_csv(output_path, index=False)
    logger.info("Noise-corrected manifest saved to %s (%d train rows remaining, was %d)", output_path, len(corrected_train), len(train_rows))


if __name__ == "__main__":
    main()