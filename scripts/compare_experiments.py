#!/usr/bin/env python
"""
Entry point: build the final cross-condition comparison table and plots.

Compares TEST SET metrics across conditions (never validation metrics —
those were used for model selection during training and would bias the
comparison). See scripts/train_baseline.py for how test_* metrics are logged.

Usage:
    python scripts/compare_experiments.py --experiment-name defect-dcai
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment.comparison import build_comparison_table, significance_test
from src.utils.logger import get_logger
from src.visualization.results_plots import plot_condition_comparison

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all experimental conditions")
    parser.add_argument("--experiment-name", default="defect-dcai")
    parser.add_argument("--metric", default="metrics.test_f1", help="MLflow metric column to compare (default: true test-set F1)")
    args = parser.parse_args()

    table = build_comparison_table(args.experiment_name, metric=args.metric)
    logger.info("\n%s", table.to_string(index=False))
    table.to_csv("reports/comparison_table.csv", index=False)

    plot_condition_comparison(table, metric_name="Test F1 Score", save_path="reports/figures/condition_comparison.png")

    if {"baseline", "dcai_improved"}.issubset(set(table["condition"])):
        result = significance_test(args.experiment_name, "baseline", "dcai_improved", metric=args.metric)
        logger.info("Significance test (baseline vs. dcai_improved): %s", result)


if __name__ == "__main__":
    main()