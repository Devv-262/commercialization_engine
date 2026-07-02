"""
feature_engineering.py — Compute the 8 engineered features for each concept.

All feature functions are pure: they take cleaned DataFrames in, return
a concept-level feature DataFrame out. No hidden state or side effects.

Features:
    1. demand_intensity   — weighted blend of feedback, follow-up rate, urgency
    2. repeatability      — normalised repeat usage days + session count
    3. engagement_depth   — weighted blend of clicks, time spent, active users
    4. segment_similarity — Shannon entropy of segment distribution per concept
                            (pre-clustering proxy; replaced by cluster signal post-clustering)
    5. revenue_potential  — normalised blend of WTP and expected value
    6. feasibility        — 1 - delivery_complexity, adjusted for implementation risk
    7. strategic_fit      — passed through from concepts table (already 0-1)
    8. confidence         — statistical confidence (delegated to confidence.py post-model;
                            placeholder here = sample-size proxy)
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from src.config import (
    DEMAND_WEIGHTS,
    ENGAGEMENT_WEIGHTS,
    LOG_FORMAT,
    LOG_LEVEL,
    REPEATABILITY_WEIGHTS,
    REVENUE_WEIGHTS,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Apply min-max normalisation to a Series, returning values in [0, 1].

    If all values are identical, returns a Series of 0.5 (neutral signal).

    Parameters
    ----------
    series : pd.Series
        Numeric series to normalise.

    Returns
    -------
    pd.Series
        Normalised series in [0, 1].
    """
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(0.5, index=series.index, name=series.name)
    return (series - s_min) / (s_max - s_min)


# ---------------------------------------------------------------------------
# Feature 1: demand_intensity
# ---------------------------------------------------------------------------


def compute_demand_intensity(
    demo_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute demand_intensity per concept.

    demand_intensity: float  # 0–1, weighted blend of normalised feedback score,
                             # follow-up request rate, and mean urgency signal.
    Reflects how strongly customers signal they want this product.

    Parameters
    ----------
    demo_df : pd.DataFrame
        Cleaned demo_signals table.
    commercial_df : pd.DataFrame
        Cleaned commercial_signals table.

    Returns
    -------
    pd.Series
        Index = concept_id, values = demand_intensity in [0, 1].
    """
    # Aggregate demo signals per concept
    demo_agg = demo_df.groupby("concept_id").agg(
        mean_feedback=("feedback_score", "mean"),
        follow_up_rate=("follow_up_requested", "mean"),
    )
    demo_agg["feedback_norm"] = _min_max_normalize(demo_agg["mean_feedback"] / 10.0)

    # Aggregate commercial urgency
    urgency_agg = commercial_df.groupby("concept_id")["urgency_score"].mean()

    combined = demo_agg.join(urgency_agg, how="left").fillna(0.0)

    demand_intensity = (
        DEMAND_WEIGHTS["feedback_score_norm"] * combined["feedback_norm"]
        + DEMAND_WEIGHTS["follow_up_rate"] * combined["follow_up_rate"]
        + DEMAND_WEIGHTS["urgency_score"] * combined["urgency_score"]
    )
    demand_intensity.name = "demand_intensity"
    logger.debug("Computed demand_intensity for %d concepts.", len(demand_intensity))
    return demand_intensity.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Feature 2: repeatability
# ---------------------------------------------------------------------------


def compute_repeatability(usage_df: pd.DataFrame) -> pd.Series:
    """
    Compute repeatability per concept.

    repeatability: float  # 0–1, weighted blend of normalised repeat_usage_days
                          # and normalised trial_sessions. Measures whether
                          # customers return to use the product after first contact.

    Parameters
    ----------
    usage_df : pd.DataFrame
        Cleaned usage_signals table.

    Returns
    -------
    pd.Series
        Index = concept_id, values = repeatability in [0, 1].
    """
    agg = usage_df.groupby("concept_id").agg(
        mean_repeat_days=("repeat_usage_days", "mean"),
        mean_sessions=("trial_sessions", "mean"),
    )
    agg["repeat_norm"] = _min_max_normalize(agg["mean_repeat_days"])
    agg["sessions_norm"] = _min_max_normalize(agg["mean_sessions"])

    repeatability = (
        REPEATABILITY_WEIGHTS["repeat_usage_days_norm"] * agg["repeat_norm"]
        + REPEATABILITY_WEIGHTS["trial_sessions_norm"] * agg["sessions_norm"]
    )
    repeatability.name = "repeatability"
    logger.debug("Computed repeatability for %d concepts.", len(repeatability))
    return repeatability.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Feature 3: engagement_depth
# ---------------------------------------------------------------------------


def compute_engagement_depth(usage_df: pd.DataFrame) -> pd.Series:
    """
    Compute engagement_depth per concept.

    engagement_depth: float  # 0–1, weighted blend of normalised feature_clicks,
                              # time_spent (minutes), and active_users. Captures
                              # how deeply customers explored the product during trial.

    Parameters
    ----------
    usage_df : pd.DataFrame
        Cleaned usage_signals table.

    Returns
    -------
    pd.Series
        Index = concept_id, values = engagement_depth in [0, 1].
    """
    agg = usage_df.groupby("concept_id").agg(
        mean_clicks=("feature_clicks", "mean"),
        mean_time=("time_spent", "mean"),
        mean_active=("active_users", "mean"),
    )
    agg["clicks_norm"] = _min_max_normalize(agg["mean_clicks"])
    agg["time_norm"] = _min_max_normalize(agg["mean_time"])
    agg["active_norm"] = _min_max_normalize(agg["mean_active"])

    engagement_depth = (
        ENGAGEMENT_WEIGHTS["feature_clicks_norm"] * agg["clicks_norm"]
        + ENGAGEMENT_WEIGHTS["time_spent_norm"] * agg["time_norm"]
        + ENGAGEMENT_WEIGHTS["active_users_norm"] * agg["active_norm"]
    )
    engagement_depth.name = "engagement_depth"
    logger.debug("Computed engagement_depth for %d concepts.", len(engagement_depth))
    return engagement_depth.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Feature 4: segment_similarity (pre-clustering proxy)
# ---------------------------------------------------------------------------


def compute_segment_similarity_raw(demo_df: pd.DataFrame) -> pd.Series:
    """
    Compute segment_similarity_raw per concept as Shannon entropy of segment distribution.

    segment_similarity_raw: float  # 0–1 (normalised entropy). Pre-clustering proxy
                                    # for whether demand comes from multiple distinct
                                    # market segments. Higher entropy = more diverse
                                    # segment coverage = stronger reusability signal.
                                    # Replaced post-clustering by cluster_repeatability_score.

    Parameters
    ----------
    demo_df : pd.DataFrame
        Cleaned demo_signals table (must contain 'segment' column).

    Returns
    -------
    pd.Series
        Index = concept_id, values = normalised segment entropy in [0, 1].
    """
    def normalised_entropy(segment_series: pd.Series) -> float:
        counts = segment_series.value_counts(normalize=True)
        raw_entropy = float(scipy_entropy(counts))
        max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
        return raw_entropy / max_entropy if max_entropy > 0 else 0.0

    segment_entropy = demo_df.groupby("concept_id")["segment"].apply(normalised_entropy)
    segment_entropy.name = "segment_similarity_raw"
    logger.debug("Computed segment_similarity_raw for %d concepts.", len(segment_entropy))
    return segment_entropy.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Feature 5: revenue_potential
# ---------------------------------------------------------------------------


def compute_revenue_potential(commercial_df: pd.DataFrame) -> pd.Series:
    """
    Compute revenue_potential per concept.

    revenue_potential: float  # 0–1, normalised weighted blend of mean
                               # willingness_to_pay and mean expected_value.
                               # Reflects the economic attractiveness of commercialising
                               # this concept given customer-stated value signals.

    Parameters
    ----------
    commercial_df : pd.DataFrame
        Cleaned commercial_signals table.

    Returns
    -------
    pd.Series
        Index = concept_id, values = revenue_potential in [0, 1].
    """
    agg = commercial_df.groupby("concept_id").agg(
        mean_wtp=("willingness_to_pay", "mean"),
        mean_ev=("expected_value", "mean"),
    )
    agg["wtp_norm"] = _min_max_normalize(agg["mean_wtp"])
    agg["ev_norm"] = _min_max_normalize(agg["mean_ev"])

    revenue_potential = (
        REVENUE_WEIGHTS["willingness_to_pay_norm"] * agg["wtp_norm"]
        + REVENUE_WEIGHTS["expected_value_norm"] * agg["ev_norm"]
    )
    revenue_potential.name = "revenue_potential"
    logger.debug("Computed revenue_potential for %d concepts.", len(revenue_potential))
    return revenue_potential.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Feature 6: feasibility
# ---------------------------------------------------------------------------


def compute_feasibility(
    concepts_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute feasibility per concept.

    feasibility: float  # 0–1, derived as (1 - delivery_complexity) penalised by
                         # mean implementation_risk from commercial signals.
                         # A concept that is easy to deliver but has high customer-perceived
                         # integration risk is not scored as fully feasible.

    Parameters
    ----------
    concepts_df : pd.DataFrame
        Cleaned concepts table (must contain 'delivery_complexity').
    commercial_df : pd.DataFrame
        Cleaned commercial_signals table (must contain 'implementation_risk').

    Returns
    -------
    pd.Series
        Index = concept_id, values = feasibility in [0, 1].
    """
    risk_agg = commercial_df.groupby("concept_id")["implementation_risk"].mean()
    concept_indexed = concepts_df.set_index("concept_id")

    combined = concept_indexed[["delivery_complexity"]].join(risk_agg, how="left").fillna(0.5)

    # feasibility = (1 - delivery_complexity) * (1 - implementation_risk)
    feasibility = (1.0 - combined["delivery_complexity"]) * (1.0 - combined["implementation_risk"])
    feasibility.name = "feasibility"
    logger.debug("Computed feasibility for %d concepts.", len(feasibility))
    return feasibility.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Feature 7: strategic_fit (pass-through)
# ---------------------------------------------------------------------------


def compute_strategic_fit(concepts_df: pd.DataFrame) -> pd.Series:
    """
    Pass through the strategic_fit field from the concepts table.

    strategic_fit: float  # 0–1, already normalised at generation time.
                           # Represents alignment of the concept with the
                           # organisation's strategic direction and portfolio priorities.

    Parameters
    ----------
    concepts_df : pd.DataFrame
        Cleaned concepts table.

    Returns
    -------
    pd.Series
        Index = concept_id, values = strategic_fit in [0, 1].
    """
    result = concepts_df.set_index("concept_id")["strategic_fit"]
    result.name = "strategic_fit"
    return result.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Feature 8: confidence_proxy (sample-size proxy, pre-model)
# ---------------------------------------------------------------------------


def compute_confidence_proxy(
    demo_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute a pre-model confidence proxy based on observation counts per concept.

    confidence_proxy: float  # 0–1. Concepts with fewer than 3 unique customer
                              # touchpoints across all signal types are assigned
                              # low confidence by design. Scaled by log of total
                              # observations up to a saturation point.
                              # Full confidence scoring (including variance and
                              # cluster tightness) is done in models/confidence.py
                              # after the model has been trained.

    Parameters
    ----------
    demo_df : pd.DataFrame
        Cleaned demo_signals table.
    usage_df : pd.DataFrame
        Cleaned usage_signals table.
    commercial_df : pd.DataFrame
        Cleaned commercial_signals table.

    Returns
    -------
    pd.Series
        Index = concept_id, values = confidence proxy in [0, 1].
    """
    demo_counts = demo_df.groupby("concept_id")["customer_id"].nunique().rename("demo_customers")
    usage_counts = usage_df.groupby("concept_id")["customer_id"].nunique().rename("usage_customers")
    commercial_counts = (
        commercial_df.groupby("concept_id")["customer_id"].nunique().rename("commercial_customers")
    )

    all_concept_ids = set(demo_df["concept_id"].unique())
    counts = (
        pd.DataFrame(index=list(all_concept_ids))
        .join(demo_counts)
        .join(usage_counts)
        .join(commercial_counts)
        .fillna(0)
    )
    counts["total_customers"] = counts[["demo_customers", "usage_customers", "commercial_customers"]].max(axis=1)

    # Log-scaled saturation at ~30 unique customers
    saturation_point = 30.0
    confidence_proxy = np.log1p(counts["total_customers"]) / np.log1p(saturation_point)
    confidence_proxy = confidence_proxy.clip(0.0, 1.0)
    confidence_proxy.name = "confidence_proxy"
    logger.debug("Computed confidence_proxy for %d concepts.", len(confidence_proxy))
    return confidence_proxy


# ---------------------------------------------------------------------------
# Master feature table builder
# ---------------------------------------------------------------------------


def build_feature_table(
    concepts_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete concept-level feature table from cleaned signal DataFrames.

    Joins all 8 engineered features into a single DataFrame indexed by concept_id.
    This is the input to both the clustering and readiness model steps.

    Parameters
    ----------
    concepts_df : pd.DataFrame
        Cleaned concepts table.
    demo_df : pd.DataFrame
        Cleaned demo_signals table.
    usage_df : pd.DataFrame
        Cleaned usage_signals table.
    commercial_df : pd.DataFrame
        Cleaned commercial_signals table.

    Returns
    -------
    pd.DataFrame
        Columns: concept_id, demand_intensity, repeatability, engagement_depth,
                 segment_similarity_raw, revenue_potential, feasibility,
                 strategic_fit, confidence_proxy
        All features are in [0, 1].
    """
    logger.info("Building concept-level feature table...")

    demand_intensity = compute_demand_intensity(demo_df, commercial_df)
    repeatability = compute_repeatability(usage_df)
    engagement_depth = compute_engagement_depth(usage_df)
    segment_similarity_raw = compute_segment_similarity_raw(demo_df)
    revenue_potential = compute_revenue_potential(commercial_df)
    feasibility = compute_feasibility(concepts_df, commercial_df)
    strategic_fit = compute_strategic_fit(concepts_df)
    confidence_proxy = compute_confidence_proxy(demo_df, usage_df, commercial_df)

    feature_df = pd.DataFrame(
        {
            "demand_intensity": demand_intensity,
            "repeatability": repeatability,
            "engagement_depth": engagement_depth,
            "segment_similarity_raw": segment_similarity_raw,
            "revenue_potential": revenue_potential,
            "feasibility": feasibility,
            "strategic_fit": strategic_fit,
            "confidence_proxy": confidence_proxy,
        }
    )

    # Merge in delivery_complexity from concepts (needed by decision engine)
    concept_meta = concepts_df.set_index("concept_id")[["delivery_complexity", "concept_name"]]
    feature_df = feature_df.join(concept_meta, how="left")
    feature_df.index.name = "concept_id"
    feature_df = feature_df.reset_index()

    logger.info(
        "Feature table complete. Shape: %s. Columns: %s",
        feature_df.shape,
        feature_df.columns.tolist(),
    )
    return feature_df


if __name__ == "__main__":
    from src.data_pipeline.clean_validate import load_and_clean_all
    from src.data_pipeline.load_to_db import run_pipeline

    run_pipeline()
    data = load_and_clean_all()
    features = build_feature_table(
        data["concepts"],
        data["demo_signals"],
        data["usage_signals"],
        data["commercial_signals"],
    )
    print(features.to_string())
