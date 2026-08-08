#!/usr/bin/env python
"""
Entry point: run the active learning loop (dcai_improved condition).

Usage:
    python scripts/train_active_learning.py \
        --config configs/dataset_casting.yaml \
        --al-config configs/active_learning_config.yaml \
        --model-config configs/model_config.yaml
"""
import argparse

import pandas as pd

from src.active_learning.loop import run_active_learning_loop
from src.experiment.tracker import log_metrics, track_run
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.visualization.results_plots import plot_active_learning_curve

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the active learning loop")
    parser.add_argument("--config", required=True, help="Dataset config YAML")
    parser.add_argument("--al-config", default="configs/active_learning_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    args = parser.parse_args()

    dataset_cfg = load_config(args.config)
    al_cfg = load_config(args.al_config)
    set_seed(42)

    # TODO: wire up train_fn / evaluate_fn closures using src.models.train.train_model
    # and src.models.evaluate.evaluate_model once the baseline pipeline (train_baseline.py)
    # is validated end-to-end. Placeholder raises to make this explicit rather than
    # silently running with incomplete logic.
    raise NotImplementedError(
        "Wire train_fn/evaluate_fn closures after validating train_baseline.py — "
        "see src/active_learning/loop.py TODO for details."
    )


if __name__ == "__main__":
    main()
