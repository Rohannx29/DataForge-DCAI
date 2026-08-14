#!/usr/bin/env python
"""
Entry point: exploratory data analysis for the MVTec AD combined manifest.

Unlike scripts/run_eda.py (built for Casting's flat two-class layout), this
reads the manifest already built by build_mvtec_manifest.py directly, since
MVTec's raw directory structure (train/good + test/<many defect types> +
ground_truth masks) isn't something a generic directory-scanning EDA script
should re-parse. Reports both overall and per-category class distribution,
plus defect-type diversity — new axes Casting's single-category EDA never needed.

Usage:
    python scripts/run_mvtec_eda.py --manifest data/processed/mvtec_ad/manifest_baseline.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.labels.quality_audit import audit_label_distribution
from src.utils.logger import get_logger
from src.visualization.eda_plots import plot_class_distribution, plot_sample_grid

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on the MVTec AD combined manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest_baseline.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)

    logger.info("=== Overall Distribution ===")
    overall_report = audit_label_distribution(manifest)
    logger.info("\n%s", overall_report.summary())

    logger.info("=== Per-Category Distribution ===")
    for category in sorted(manifest["category"].unique()):
        cat_manifest = manifest[manifest["category"] == category]
        cat_report = audit_label_distribution(cat_manifest)
        logger.info("--- %s ---\n%s", category, cat_report.summary())

    logger.info("=== Defect Type Breakdown (defective samples only) ===")
    defect_breakdown = manifest[manifest["label"] == 1].groupby(["category", "defect_type"]).size()
    logger.info("\n%s", defect_breakdown.to_string())

    figures_dir = Path("reports/figures")
    plot_class_distribution(manifest, save_path=figures_dir / "mvtec_class_distribution_overall.png")

    for category in sorted(manifest["category"].unique()):
        cat_manifest = manifest[manifest["category"] == category].reset_index(drop=True)
        plot_class_distribution(cat_manifest, save_path=figures_dir / f"mvtec_{category}_class_distribution.png")
        plot_sample_grid(cat_manifest, n_samples_per_class=3, save_path=figures_dir / f"mvtec_{category}_sample_grid.png")

    logger.info("EDA figures saved to %s", figures_dir)


if __name__ == "__main__":
    main()