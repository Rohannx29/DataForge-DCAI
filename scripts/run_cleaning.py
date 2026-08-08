#!/usr/bin/env python
"""
Entry point: data cleaning (produces the 'cleaned' experimental condition manifest).

Usage:
    python scripts/run_cleaning.py --config configs/dataset_casting.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaning import remove_corrupt_files, resolve_duplicates, save_cleaned_manifest
from src.data.preprocessing import build_manifest_from_directory, stratified_split
from src.data.validation import find_duplicate_images, validate_image_integrity
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw dataset -> 'cleaned' condition manifest")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]

    class_map = {name: idx for idx, name in enumerate(dataset_cfg.get("classes", []))}
    manifest = build_manifest_from_directory(dataset_cfg["raw_dir"], class_map)

    validation_report = validate_image_integrity(dataset_cfg["raw_dir"])
    manifest = remove_corrupt_files(manifest, validation_report)

    duplicate_groups = find_duplicate_images(dataset_cfg["raw_dir"])
    manifest = resolve_duplicates(manifest, duplicate_groups)

    split_cfg = config["split"]
    manifest = stratified_split(
        manifest,
        train_frac=split_cfg["train"],
        val_frac=split_cfg["val"],
        test_frac=split_cfg["test"],
    )

    output_path = f"{dataset_cfg['processed_dir']}/manifest_cleaned.csv"
    save_cleaned_manifest(manifest, output_path)


if __name__ == "__main__":
    main()