#!/usr/bin/env python
"""
Entry point: label quality audit and noise detection.

Usage:
    python scripts/run_label_audit.py --config configs/dataset_casting.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.labels.noise_correction import remove_flagged_samples
from src.labels.noise_detection import detect_label_issues, get_out_of_sample_predictions
from src.labels.quality_audit import audit_label_distribution
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit label quality and detect noisy labels")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    parser.add_argument("--manifest", required=True, help="Path to cleaned manifest CSV")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest)

    report = audit_label_distribution(manifest)
    logger.info("\n%s", report.summary())

    logger.info("Computing out-of-sample predictions for confident learning...")
    pred_probs = get_out_of_sample_predictions(model_fn=None, manifest=manifest)  # TODO: wire up model_fn

    issue_indices = detect_label_issues(labels=manifest["label"].values, pred_probs=pred_probs)

    corrected = remove_flagged_samples(manifest, issue_indices)
    output_path = args.manifest.replace("manifest_cleaned.csv", "manifest_noise_corrected.csv")
    corrected.to_csv(output_path, index=False)
    logger.info("Noise-corrected manifest saved to %s", output_path)


if __name__ == "__main__":
    main()