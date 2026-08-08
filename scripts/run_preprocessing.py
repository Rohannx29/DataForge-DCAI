#!/usr/bin/env python
"""
Entry point: final preprocessing (normalization stats) before training.

Usage:
    python scripts/run_preprocessing.py --manifest data/processed/casting/manifest_cleaned.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.preprocessing import compute_normalization_stats
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute preprocessing statistics for a manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--output", default=None, help="Path to save normalization stats JSON")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    mean, std = compute_normalization_stats(manifest)

    stats = {"mean": mean, "std": std}
    output_path = args.output or args.manifest.replace(".csv", "_norm_stats.json")

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Normalization stats saved to %s", output_path)


if __name__ == "__main__":
    main()