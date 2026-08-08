"""
Active learning query loop.

Orchestrates the iterative label-acquisition process: train -> predict on pool
-> select query batch -> "query" the oracle -> add to labeled set -> repeat.
This produces the labels-used-vs-performance curve that is the primary result
of the DCAI-improved experimental condition.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.active_learning.sampling import diversity_sampling, uncertainty_sampling
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ActiveLearningHistory:
    """Tracks metrics at each active learning round for plotting the learning curve."""

    round_numbers: list[int] = field(default_factory=list)
    n_labeled: list[int] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    val_auroc: list[float] = field(default_factory=list)


def simulated_oracle_query(pool_manifest: pd.DataFrame, query_indices: np.ndarray, noise_rate: float = 0.0) -> pd.DataFrame:
    """Simulate a human annotator by revealing ground-truth labels for queried samples.

    Args:
        pool_manifest: Unlabeled pool DataFrame (labels present but hidden from the model).
        query_indices: Indices selected by a sampling strategy for labeling.
        noise_rate: Probability of the oracle returning an incorrect label
            (simulates annotator error; 0.0 = perfect oracle).

    Returns:
        DataFrame subset of pool_manifest for the queried indices, ready to
        append to the labeled set.
    """
    queried = pool_manifest.iloc[query_indices].copy()

    if noise_rate > 0:
        rng = np.random.RandomState(42)
        flip_mask = rng.random(len(queried)) < noise_rate
        classes = pool_manifest["label"].unique()
        for idx in queried.index[flip_mask]:
            true_label = queried.at[idx, "label"]
            wrong_choices = [c for c in classes if c != true_label]
            queried.at[idx, "label"] = rng.choice(wrong_choices)
        logger.info("Oracle noise injected into %d / %d queried labels", flip_mask.sum(), len(queried))

    return queried


def run_active_learning_loop(
    labeled_manifest: pd.DataFrame,
    pool_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    n_rounds: int,
    query_batch_size: int,
    strategy: str,
    train_fn,
    evaluate_fn,
) -> ActiveLearningHistory:
    """Run the full active learning loop for n_rounds.

    Args:
        labeled_manifest: Initial seed labeled set.
        pool_manifest: Initial unlabeled pool (ground truth present but hidden).
        val_manifest: Fixed validation set used to score each round.
        n_rounds: Number of query rounds to run.
        query_batch_size: Number of samples queried per round.
        strategy: "uncertainty_sampling" or "diversity_sampling" (see sampling.py).
        train_fn: Callable(manifest) -> trained model, wraps src.models.train.train_model.
        evaluate_fn: Callable(model, manifest) -> dict of metrics, wraps src.models.evaluate.evaluate_model.

    Returns:
        ActiveLearningHistory tracking performance vs. labels-used across all rounds.
    """
    history = ActiveLearningHistory()

    for round_num in range(n_rounds):
        model = train_fn(labeled_manifest)
        val_metrics = evaluate_fn(model, val_manifest)

        history.round_numbers.append(round_num)
        history.n_labeled.append(len(labeled_manifest))
        history.val_f1.append(val_metrics["f1"])
        history.val_auroc.append(val_metrics.get("auroc", float("nan")))

        logger.info(
            "AL Round %d | n_labeled=%d | val_f1=%.4f", round_num, len(labeled_manifest), val_metrics["f1"]
        )

        if len(pool_manifest) == 0:
            logger.info("Unlabeled pool exhausted — stopping active learning loop")
            break

        # TODO: obtain pred_probs on pool_manifest via evaluate_fn / model inference,
        # then select query indices via uncertainty_sampling(pred_probs, query_batch_size).
        # Placeholder: random selection until model inference wiring is implemented.
        query_indices = np.random.choice(len(pool_manifest), size=min(query_batch_size, len(pool_manifest)), replace=False)

        queried = simulated_oracle_query(pool_manifest, query_indices)
        labeled_manifest = pd.concat([labeled_manifest, queried]).reset_index(drop=True)
        pool_manifest = pool_manifest.drop(pool_manifest.index[query_indices]).reset_index(drop=True)

    return history
