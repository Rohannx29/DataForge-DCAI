"""
Cross-condition comparison.

Pulls logged runs for all four experimental conditions from MLflow and builds
the comparison table / statistical tests that form the core "Experimental
Results" section of the report. Requires n_runs >= 3 per condition (see
configs/model_config.yaml -> evaluation.n_runs) to compute confidence intervals.
"""
import mlflow
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_condition_runs(experiment_name: str, condition: str) -> pd.DataFrame:
    """Fetch all logged runs for a given experimental condition.

    Args:
        experiment_name: MLflow experiment name.
        condition: One of "baseline", "cleaned", "noise_corrected", "dcai_improved".

    Returns:
        DataFrame of runs (one row per run) with metrics as columns.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.condition = '{condition}'",
    )
    logger.info("Fetched %d runs for condition '%s'", len(runs), condition)
    return runs


def build_comparison_table(experiment_name: str, metric: str = "metrics.f1") -> pd.DataFrame:
    """Build a summary table (mean, std, 95% CI) of a metric across all four conditions.

    Args:
        experiment_name: MLflow experiment name.
        metric: MLflow metric column name to summarize (must have been logged
            via src.experiment.tracker.log_metrics).

    Returns:
        DataFrame with columns [condition, n_runs, mean, std, ci_low, ci_high].
    """
    conditions = ["baseline", "cleaned", "noise_corrected", "dcai_improved"]
    rows = []

    for condition in conditions:
        runs = fetch_condition_runs(experiment_name, condition)
        if metric not in runs.columns or runs.empty:
            logger.warning("No data for condition '%s', metric '%s'", condition, metric)
            continue

        values = runs[metric].dropna().values
        mean, std = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
        ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=std / np.sqrt(len(values))) if len(values) > 1 else (mean, mean)

        rows.append({
            "condition": condition,
            "n_runs": len(values),
            "mean": mean,
            "std": std,
            "ci_low": ci[0],
            "ci_high": ci[1],
        })

    return pd.DataFrame(rows)


def significance_test(experiment_name: str, condition_a: str, condition_b: str, metric: str = "metrics.f1") -> dict:
    """Run an independent-samples t-test between two experimental conditions.

    Used to justify claims like "DCAI-improved significantly outperforms baseline
    (p < 0.05)" in the report, rather than relying on a single-run point estimate.

    Args:
        experiment_name: MLflow experiment name.
        condition_a: First condition to compare.
        condition_b: Second condition to compare.
        metric: Metric column to test.

    Returns:
        Dict with 't_statistic', 'p_value', and both conditions' means.
    """
    runs_a = fetch_condition_runs(experiment_name, condition_a)[metric].dropna().values
    runs_b = fetch_condition_runs(experiment_name, condition_b)[metric].dropna().values

    if len(runs_a) < 2 or len(runs_b) < 2:
        raise ValueError("Need at least 2 runs per condition for a t-test — increase evaluation.n_runs in config")

    t_stat, p_value = stats.ttest_ind(runs_a, runs_b)
    return {
        "condition_a": condition_a, "mean_a": runs_a.mean(),
        "condition_b": condition_b, "mean_b": runs_b.mean(),
        "t_statistic": t_stat, "p_value": p_value,
    }
