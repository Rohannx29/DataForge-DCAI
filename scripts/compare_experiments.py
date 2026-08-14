#!/usr/bin/env python
"""
Entry point: build the final cross-condition comparison table and plots.

Compares TEST SET metrics across conditions WITHIN A SINGLE DATASET.
--dataset is REQUIRED and must match what was passed to train_baseline.py,
since experiment names are per-dataset (see scripts/train_baseline.py).

Usage:
    python scripts/compare_experiments.py --dataset mvtec_ad
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment.comparison import build_comparison_table, significance_test
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.visualization.results_plots import plot_condition_comparison

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all experimental conditions for one dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. 'casting', 'mvtec_ad') — must match what was used in train_baseline.py")
    parser.add_argument("--base-config", default="configs/base_config.yaml")
    parser.add_argument("--metric", default="metrics.test_f1", help="MLflow metric column to compare (default: true test-set F1)")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    experiment_name = f"{base_cfg['experiment_tracking']['experiment_name']}-{args.dataset}"

    table = build_comparison_table(experiment_name, metric=args.metric)
    logger.info("\n%s", table.to_string(index=False))

    output_csv = f"reports/comparison_table_{args.dataset}.csv"
    table.to_csv(output_csv, index=False)
    logger.info("Comparison table saved to %s", output_csv)

    output_fig = f"reports/figures/condition_comparison_{args.dataset}.png"
    plot_condition_comparison(table, metric_name="Test F1 Score", save_path=output_fig)
    logger.info("Comparison plot saved to %s", output_fig)

    if {"baseline", "dcai_improved"}.issubset(set(table["condition"])):
        result = significance_test(experiment_name, "baseline", "dcai_improved", metric=args.metric)
        logger.info("Significance test (baseline vs. dcai_improved): %s", result)


if __name__ == "__main__":
    main()