#!/usr/bin/env python
"""
Entry point: dataset validation (corrupt files, duplicates).

Usage:
    python scripts/run_validation.py --config configs/dataset_casting.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.validation import find_duplicate_images, validate_image_integrity
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw dataset integrity")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = config["dataset"]["raw_dir"]

    report = validate_image_integrity(raw_dir)
    logger.info(report.summary())

    duplicates = find_duplicate_images(raw_dir)
    logger.info("Found %d duplicate groups", len(duplicates))

    if not report.is_clean:
        logger.warning("Dataset has integrity issues — run scripts/run_cleaning.py before proceeding")


if __name__ == "__main__":
    main()