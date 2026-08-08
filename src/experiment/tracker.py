"""
Experiment tracking wrapper.

Thin wrapper around MLflow so every script logs metrics/params the same way,
tagged consistently with the experimental condition (baseline/cleaned/
noise_corrected/dcai_improved) — this tagging is what makes cross-condition
comparison in comparison.py possible.
"""
from contextlib import contextmanager
from typing import Any

import mlflow

from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_CONDITIONS = {"baseline", "cleaned", "noise_corrected", "dcai_improved"}


@contextmanager
def track_run(experiment_name: str, condition: str, run_name: str | None = None):
    """Context manager wrapping an MLflow run, tagged with the experimental condition.

    Args:
        experiment_name: MLflow experiment name (typically from base_config.yaml).
        condition: One of VALID_CONDITIONS — required for later comparison.py to work.
        run_name: Optional human-readable run name.

    Yields:
        The active mlflow run context.

    Raises:
        ValueError: If condition is not one of the four recognized experimental conditions.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {VALID_CONDITIONS}, got '{condition}'")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("condition", condition)
        logger.info("Started MLflow run '%s' [condition=%s]", run.info.run_id, condition)
        yield run


def log_params(params: dict[str, Any]) -> None:
    """Log a dict of hyperparameters to the active MLflow run."""
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of metrics to the active MLflow run, optionally at a given step."""
    mlflow.log_metrics(metrics, step=step)
