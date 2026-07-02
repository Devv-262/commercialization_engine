"""
clustering.py — KMeans clustering on concept engagement/demand features.

Finds the optimal k (2–6) using silhouette score, then:
- Assigns each concept a cluster_id
- Computes a cluster_repeatability_score: the mean repeatability of concepts
  in the same cluster, normalised across clusters. Feeds the Reusable Asset outcome.

This module is the ML layer — no narrative or decision logic here.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLUSTER_FEATURES,
    CLUSTER_K_MAX,
    CLUSTER_K_MIN,
    LOG_FORMAT,
    LOG_LEVEL,
    RANDOM_SEED,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def select_optimal_k(
    feature_matrix: np.ndarray,
    k_min: int = CLUSTER_K_MIN,
    k_max: int = CLUSTER_K_MAX,
) -> int:
    """
    Select the optimal number of clusters via silhouette score.

    Fits KMeans for each k in [k_min, k_max] and returns the k with the
    highest silhouette score. Silhouette measures cluster cohesion vs. separation.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Standardised feature matrix (n_concepts × n_features).
    k_min : int
        Minimum number of clusters to evaluate.
    k_max : int
        Maximum number of clusters to evaluate.

    Returns
    -------
    int
        Optimal k.
    """
    best_k = k_min
    best_score = -1.0

    # Cap k_max at n_samples - 1 (silhouette requirement)
    effective_k_max = min(k_max, feature_matrix.shape[0] - 1)

    for k in range(k_min, effective_k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        score = silhouette_score(feature_matrix, labels)
        logger.debug("k=%d → silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k

    logger.info("Optimal k selected: %d (silhouette=%.4f)", best_k, best_score)
    return best_k


def fit_clusters(
    feature_df: pd.DataFrame,
    cluster_features: list[str] = None,
) -> pd.DataFrame:
    """
    Fit KMeans clustering and return enriched feature DataFrame.

    Adds two columns to the feature DataFrame:
    - cluster_id: integer cluster assignment for each concept.
    - cluster_repeatability_score: normalised mean repeatability of the concept's
      cluster. Used as the segment_similarity signal for the Reusable Asset outcome.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output from feature_engineering.build_feature_table(). Must contain
        concept_id and the clustering feature columns.
    cluster_features : list[str], optional
        Feature names to use for clustering. Defaults to CLUSTER_FEATURES from config.

    Returns
    -------
    pd.DataFrame
        feature_df enriched with 'cluster_id' and 'cluster_repeatability_score'.
    """
    if cluster_features is None:
        cluster_features = CLUSTER_FEATURES

    # Filter to available columns
    available_features = [f for f in cluster_features if f in feature_df.columns]
    if len(available_features) < 2:
        logger.warning(
            "Fewer than 2 clustering features available (%s). Assigning all to cluster 0.",
            available_features,
        )
        enriched = feature_df.copy()
        enriched["cluster_id"] = 0
        enriched["cluster_repeatability_score"] = 0.5
        return enriched

    logger.info("Fitting KMeans with features: %s", available_features)

    matrix = feature_df[available_features].fillna(0.0).values
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)

    optimal_k = select_optimal_k(scaled_matrix)

    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_matrix)

    enriched = feature_df.copy()
    enriched["cluster_id"] = cluster_labels

    # Compute cluster_repeatability_score:
    # For each cluster, compute mean repeatability of its members.
    # Normalise across clusters so the score is in [0, 1].
    cluster_mean_repeat = (
        enriched.groupby("cluster_id")["repeatability"].mean().rename("cluster_mean_repeat")
    )
    enriched = enriched.join(cluster_mean_repeat, on="cluster_id")

    min_repeat = enriched["cluster_mean_repeat"].min()
    max_repeat = enriched["cluster_mean_repeat"].max()
    if max_repeat > min_repeat:
        enriched["cluster_repeatability_score"] = (
            enriched["cluster_mean_repeat"] - min_repeat
        ) / (max_repeat - min_repeat)
    else:
        enriched["cluster_repeatability_score"] = 0.5

    enriched = enriched.drop(columns=["cluster_mean_repeat"])

    logger.info(
        "Clustering complete. %d concepts assigned to %d clusters.",
        len(enriched),
        optimal_k,
    )
    logger.info(
        "Cluster distribution:\n%s",
        enriched["cluster_id"].value_counts().to_string(),
    )
    return enriched


if __name__ == "__main__":
    from src.data_pipeline.clean_validate import load_and_clean_all
    from src.data_pipeline.load_to_db import run_pipeline
    from src.features.feature_engineering import build_feature_table

    run_pipeline()
    data = load_and_clean_all()
    features = build_feature_table(
        data["concepts"], data["demo_signals"], data["usage_signals"], data["commercial_signals"]
    )
    clustered = fit_clusters(features)
    print(clustered[["concept_id", "cluster_id", "cluster_repeatability_score"]].to_string())
