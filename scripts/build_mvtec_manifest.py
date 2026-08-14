#!/usr/bin/env python
"""
Entry point: build the combined, split MVTec AD manifest across all configured
categories. Produces the 'baseline' condition manifest for Phase 2 — parallel
to scripts/build_baseline_manifest.py for the Casting dataset.

Usage:
    python scripts/build_mvtec_manifest.py --config configs/dataset_mvtec.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mvtec_preprocessing import build_combined_manifest, stratified_split_per_category
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the combined MVTec AD baseline manifest")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    split_cfg = config["split"]

    manifest = build_combined_manifest(dataset_cfg["raw_dir"], dataset_cfg["categories"])

    manifest = stratified_split_per_category(
        manifest,
        train_frac=split_cfg["train"],
        val_frac=split_cfg["val"],
        test_frac=split_cfg["test"],
    )

    split_counts = manifest.groupby(["category", "split"]).size().unstack(fill_value=0)
    logger.info("Split breakdown by category:\n%s", split_counts.to_string())

    output_dir = Path(dataset_cfg["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest_baseline.csv"
    manifest.to_csv(output_path, index=False)

    logger.info("MVTec baseline manifest saved (%d rows) to %s", len(manifest), output_path)


if __name__ == "__main__":
    main()