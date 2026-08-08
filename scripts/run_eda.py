#!/usr/bin/env python
"""
Entry point: exploratory data analysis.

Usage:
    python scripts/run_eda.py --config configs/dataset_casting.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import build_manifest_from_directory
from src.labels.quality_audit import audit_label_distribution
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.visualization.eda_plots import plot_class_distribution, plot_sample_grid

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]

    class_map = {name: idx for idx, name in enumerate(dataset_cfg.get("classes", []))}
    manifest = build_manifest_from_directory(dataset_cfg["raw_dir"], class_map)

    report = audit_label_distribution(manifest)
    logger.info("\n%s", report.summary())

    figures_dir = Path("reports/figures")
    plot_class_distribution(manifest, save_path=figures_dir / f"{dataset_cfg['name']}_class_distribution.png")
    plot_sample_grid(manifest, save_path=figures_dir / f"{dataset_cfg['name']}_sample_grid.png")

    logger.info("EDA figures saved to %s", figures_dir)


if __name__ == "__main__":
    main()