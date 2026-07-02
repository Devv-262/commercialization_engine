"""
confidence.py — Statistical confidence scoring (not ML).

Confidence reflects how much we should trust the readiness_score for each
concept, derived from:
1. Sample size — concepts with fewer customer observations are less reliable.
2. Score variance — high cross-customer variance in signals weakens certainty.
3. Cluster tightness — concepts far from their cluster centroid are less typical.

This is NOT a machine-learning model — it is a statistical quality metric.
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    CONFIDENCE_MIN_SAMPLES,
    CONFIDENCE_VARIANCE_SCALE,
    LOG_FORMAT,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _sample_size_factor(
    demo_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
    concept_ids: list[str],
) -> pd.Series:
    """
    Compute a sample-size confidence factor per concept.

    Concepts with fewer than CONFIDENCE_MIN_SAMPLES unique customers across
    all signal types receive a harsh penalty. The factor saturates at ~30
    unique customers (log-scaled).

    Parameters
    ----------
    demo_df, usage_df, commercial_df : pd.DataFrame
        Cleaned signal DataFrames.
    concept_ids : list[str]
        Full list of concept IDs.

    Returns
    -------
    pd.Series
        Index = concept_id, values in [0, 1]. Higher = more samples.
    """
    demo_counts = demo_df.groupby("concept_id")["customer_id"].nunique()
    usage_counts = usage_df.groupby("concept_id")["customer_id"].nunique()
    commercial_counts = commercial_df.groupby("concept_id")["customer_id"].nunique()

    counts = pd.DataFrame(index=concept_ids)
    counts["n_demo"] = counts.index.map(demo_counts).fillna(0)
    counts["n_usage"] = counts.index.map(usage_counts).fillna(0)
    counts["n_commercial"] = counts.index.map(commercial_counts).fillna(0)
    counts["total"] = counts[["n_demo", "n_usage", "n_commercial"]].max(axis=1)

    saturation = 30.0
    factor = np.log1p(counts["total"]) / np.log1p(saturation)
    factor = factor.clip(0.0, 1.0)

    # Harsh penalty for concepts below minimum threshold
    below_min = counts["total"] < CONFIDENCE_MIN_SAMPLES
    factor[below_min] = factor[below_min] * 0.3

    factor.name = "sample_size_factor"
    return factor


def _variance_penalty(
    demo_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
    concept_ids: list[str],
) -> pd.Series:
    """
    Compute a variance-based confidence penalty per concept.

    High variance in feedback_score and pilot_interest across customers
    suggests inconsistent demand — we should be less confident in the
    readiness score even if the mean is high.

    Parameters
    ----------
    demo_df : pd.DataFrame
        Cleaned demo_signals table.
    commercial_df : pd.DataFrame
        Cleaned commercial_signals table.
    concept_ids : list[str]
        Full list of concept IDs.

    Returns
    -------
    pd.Series
        Index = concept_id, values in [0, 1]. Higher = lower variance = more confident.
    """
    feedback_var = demo_df.groupby("concept_id")["feedback_score"].var().fillna(0)
    pilot_var = commercial_df.groupby("concept_id")["pilot_interest"].var().fillna(0)

    combined_var = pd.DataFrame(index=concept_ids)
    combined_var["fb_var"] = combined_var.index.map(feedback_var).fillna(0)
    combined_var["pi_var"] = combined_var.index.map(pilot_var).fillna(0)

    # Normalise variances: feedback on 0-10 scale → var max ≈ 25; pilot on 0-1 → var max ≈ 0.25
    normalised_var = (combined_var["fb_var"] / 25.0 + combined_var["pi_var"] / 0.25) / 2.0
    penalty = 1.0 - (normalised_var * CONFIDENCE_VARIANCE_SCALE).clip(0.0, 1.0)
    penalty = penalty.clip(0.0, 1.0)
    penalty.name = "variance_penalty"
    return penalty


def _cluster_tightness(
    feature_df: pd.DataFrame,
    concept_ids: list[str],
) -> pd.Series:
    """
    Compute cluster tightness as a confidence signal per concept.

    Concepts closer to their cluster centroid (lower intra-cluster distance)
    are more representative of a demand pattern → higher confidence.

    Uses the cluster_repeatability_score as a proxy: concepts in clusters
    with consistent repeatability are tighter.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature DataFrame enriched with cluster_id and cluster_repeatability_score.
    concept_ids : list[str]
        Full list of concept IDs.

    Returns
    -------
    pd.Series
        Index = concept_id, values in [0, 1]. Higher = tighter cluster.
    """
    if "cluster_repeatability_score" not in feature_df.columns:
        return pd.Series(0.5, index=concept_ids, name="cluster_tightness")

    indexed = feature_df.set_index("concept_id") if "concept_id" in feature_df.columns else feature_df

    # Within each cluster, compute std of key features as a spread metric
    cluster_features = ["demand_intensity", "repeatability", "engagement_depth"]
    available = [f for f in cluster_features if f in indexed.columns]

    if not available or "cluster_id" not in indexed.columns:
        return pd.Series(0.5, index=concept_ids, name="cluster_tightness")

    cluster_spread = indexed.groupby("cluster_id")[available].std().mean(axis=1)
    # Map cluster spread back to concepts
    concept_cluster = indexed["cluster_id"]
    concept_spread = concept_cluster.map(cluster_spread).fillna(0.5)

    # Invert: low spread = high tightness
    tightness = 1.0 - concept_spread.clip(0.0, 1.0)
    tightness = tightness.reindex(concept_ids).fillna(0.5)
    tightness.name = "cluster_tightness"
    return tightness


def compute_confidence_scores(
    demo_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    concept_ids: list[str],
) -> pd.Series:
    """
    Compute final confidence scores (0–100) for each concept.

    Combines three signals:
    - Sample size factor (40% weight)
    - Variance penalty (35% weight)
    - Cluster tightness (25% weight)

    Parameters
    ----------
    demo_df, usage_df, commercial_df : pd.DataFrame
        Cleaned signal DataFrames.
    feature_df : pd.DataFrame
        Feature DataFrame enriched with cluster info.
    concept_ids : list[str]
        Full list of concept IDs.

    Returns
    -------
    pd.Series
        Index = concept_id, values = confidence_score in [0, 100].
    """
    logger.info("Computing confidence scores for %d concepts...", len(concept_ids))

    sample_factor = _sample_size_factor(demo_df, usage_df, commercial_df, concept_ids)
    var_penalty = _variance_penalty(demo_df, commercial_df, concept_ids)
    tightness = _cluster_tightness(feature_df, concept_ids)

    # Weighted combination
    raw_confidence = (
        0.40 * sample_factor
        + 0.35 * var_penalty
        + 0.25 * tightness
    )

    confidence_scores = (raw_confidence * 100.0).clip(0.0, 100.0).round(2)
    confidence_scores.name = "confidence_score"

    logger.info(
        "Confidence scores — mean: %.2f, min: %.2f, max: %.2f",
        confidence_scores.mean(),
        confidence_scores.min(),
        confidence_scores.max(),
    )
    return confidence_scores
