"""
Active learning sampling strategies.

Each strategy takes model predictions on an unlabeled pool and returns the
indices most valuable to query next. Kept separate from loop.py so strategies
can be swapped/compared without touching orchestration logic.
"""
import numpy as np
from sklearn.cluster import KMeans


def uncertainty_sampling(pred_probs: np.ndarray, batch_size: int, method: str = "entropy") -> np.ndarray:
    """Select the most uncertain samples from the unlabeled pool.

    Args:
        pred_probs: Array of shape (n_pool, n_classes) — model's predicted
            probabilities on the current unlabeled pool.
        batch_size: Number of samples to select for labeling.
        method: "entropy", "least_confidence", or "margin".

    Returns:
        Array of pool indices (length batch_size) selected for querying,
        ordered by descending uncertainty.
    """
    if method == "entropy":
        eps = 1e-12
        scores = -np.sum(pred_probs * np.log(pred_probs + eps), axis=1)
    elif method == "least_confidence":
        scores = 1 - np.max(pred_probs, axis=1)
    elif method == "margin":
        sorted_probs = np.sort(pred_probs, axis=1)
        scores = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])  # smaller margin = more uncertain
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

    top_indices = np.argsort(-scores)[:batch_size]
    return top_indices


def diversity_sampling(embeddings: np.ndarray, batch_size: int, n_clusters: int = 20, seed: int = 42) -> np.ndarray:
    """Select a diverse batch via k-means clustering on feature embeddings.

    Ensures queried samples aren't all near-duplicates of each other (a known
    failure mode of pure uncertainty sampling, which can over-sample from one
    region of feature space).

    Args:
        embeddings: Array of shape (n_pool, embedding_dim) — penultimate-layer
            features from the current model for each pool sample.
        batch_size: Number of samples to select.
        n_clusters: Number of k-means clusters to partition the pool into.
        seed: Random seed for KMeans reproducibility.

    Returns:
        Array of pool indices (length batch_size), one representative closest
        to each cluster centroid, cycling through clusters if batch_size > n_clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(embeddings)
    selected = []

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
        closest = cluster_indices[np.argmin(distances)]
        selected.append(closest)
        if len(selected) >= batch_size:
            break

    return np.array(selected[:batch_size])


def hybrid_sampling(
    pred_probs: np.ndarray,
    embeddings: np.ndarray,
    batch_size: int,
    uncertainty_weight: float = 0.7,
    diversity_weight: float = 0.3,
) -> np.ndarray:
    """Combine uncertainty and diversity sampling via weighted scoring.

    Args:
        pred_probs: Array of shape (n_pool, n_classes).
        embeddings: Array of shape (n_pool, embedding_dim).
        batch_size: Number of samples to select.
        uncertainty_weight: Weight given to the uncertainty score.
        diversity_weight: Weight given to the diversity (cluster-distance) score.

    Returns:
        Array of pool indices (length batch_size) selected by combined score.
    """
    # TODO: implement combined scoring once uncertainty_sampling and diversity_sampling
    # are validated independently in Phase 2 (see reports/final_report.md progress log).
    raise NotImplementedError("Hybrid sampling planned for Phase 2 — see project timeline.")
