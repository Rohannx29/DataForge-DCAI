"""
Results / comparison plots.

Generates the primary figures for the "Experimental Results" section:
the active-learning labels-vs-performance curve, and the four-condition
bar-chart comparison with confidence intervals.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.active_learning.loop import ActiveLearningHistory


def plot_active_learning_curve(
    history: ActiveLearningHistory,
    random_baseline_history: ActiveLearningHistory | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot validation F1 vs. number of labeled samples across AL rounds.

    This is the primary evidence figure for the active learning contribution —
    ideally overlaid against a random-sampling baseline to show the efficiency gain.

    Args:
        history: ActiveLearningHistory from the active-learning-guided run.
        random_baseline_history: Optional history from a random-sampling run for comparison.
        save_path: If provided, saves the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.n_labeled, history.val_f1, marker="o", label="Active Learning (Uncertainty Sampling)")

    if random_baseline_history:
        ax.plot(random_baseline_history.n_labeled, random_baseline_history.val_f1, marker="s", linestyle="--", label="Random Sampling (Baseline)")

    ax.set_xlabel("Number of Labeled Samples")
    ax.set_ylabel("Validation F1 Score")
    ax.set_title("Active Learning: Labeling Efficiency Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_condition_comparison(comparison_table: pd.DataFrame, metric_name: str = "F1 Score", save_path: str | Path | None = None) -> None:
    """Bar chart comparing all four experimental conditions with 95% CI error bars.

    Args:
        comparison_table: Output of src.experiment.comparison.build_comparison_table().
        metric_name: Human-readable metric name for the y-axis label.
        save_path: If provided, saves the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    errors = [
        comparison_table["mean"] - comparison_table["ci_low"],
        comparison_table["ci_high"] - comparison_table["mean"],
    ]
    ax.bar(comparison_table["condition"], comparison_table["mean"], yerr=errors, capsize=5)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Across Experimental Conditions (95% CI)")
    plt.xticks(rotation=15)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
