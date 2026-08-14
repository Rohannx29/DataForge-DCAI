#!/usr/bin/env python
"""
Entry point: build the RAW 'baseline' condition manifest — no cleaning,
no label correction, no DCAI interventions applied.

This is intentionally the simplest script in the pipeline: it exists so the
'baseline' experimental condition has a real, distinct manifest to train on,
separate from 'cleaned' (which is produced by run_cleaning.py). Even when a
dataset turns out to have zero corrupt/duplicate files (as Casting did),
this script's output remains the reference point every other condition is
compared against.

Usage:
    python scripts/build_baseline_manifest.py --config configs/dataset_casting.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import build_manifest_from_directory, stratified_split
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the raw 'baseline' condition manifest")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    split_cfg = config["split"]

    class_map = {name: idx for idx, name in enumerate(dataset_cfg.get("classes", []))}
    manifest = build_manifest_from_directory(dataset_cfg["raw_dir"], class_map)

    manifest = stratified_split(
        manifest,
        train_frac=split_cfg["train"],
        val_frac=split_cfg["val"],
        test_frac=split_cfg["test"],
    )

    output_dir = Path(dataset_cfg["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest_baseline.csv"
    manifest.to_csv(output_path, index=False)

    logger.info("Baseline manifest saved (%d rows, no cleaning applied) to %s", len(manifest), output_path)


if __name__ == "__main__":
    main()