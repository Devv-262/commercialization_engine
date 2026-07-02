"""
decision_engine.py — Maps scores to 5 commercial outcomes using a decision matrix.

Outcomes (from config.py):
    1. MVP Build       — High readiness + high confidence + low-to-moderate complexity
    2. Customer Pilot   — (High readiness + high confidence + high complexity) OR
                         (moderate readiness + named urgency/budget signal)
    3. Reusable Asset   — High readiness + high cluster_repeatability (multi-segment demand)
    4. Incubate         — Moderate readiness + low confidence
    5. Archive          — Low readiness, OR high complexity/value ratio, OR negative engagement

This module is the AI insight layer — it applies business-rule logic to ML outputs.
"""

import logging

import pandas as pd

from src.config import (
    ARCHIVE_ABANDONMENT_RATE,
    ARCHIVE_COMPLEXITY_VALUE_RATIO,
    ARCHIVE_OBJECTION_COUNT,
    CLUSTER_REPEATABILITY_HIGH,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    DELIVERY_COMPLEXITY_HIGH,
    DELIVERY_COMPLEXITY_LOW_TO_MODERATE,
    LOG_FORMAT,
    LOG_LEVEL,
    OUTCOME_ARCHIVE,
    OUTCOME_CUSTOMER_PILOT,
    OUTCOME_INCUBATE,
    OUTCOME_MVP_BUILD,
    OUTCOME_REUSABLE_ASSET,
    PILOT_BUDGET_THRESHOLD,
    PILOT_URGENCY_THRESHOLD,
    READINESS_HIGH,
    READINESS_MODERATE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _compute_archive_signals(
    concept_id: str,
    usage_df: pd.DataFrame,
    demo_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute archive-specific negative signals for a single concept.

    Returns abandonment_rate and mean_objections, which are used to detect
    negative-trending engagement patterns.

    Parameters
    ----------
    concept_id : str
        Concept to evaluate.
    usage_df : pd.DataFrame
        Cleaned usage_signals table.
    demo_df : pd.DataFrame
        Cleaned demo_signals table.

    Returns
    -------
    dict[str, float]
        Keys: 'abandonment_rate', 'mean_objections'.
    """
    concept_usage = usage_df[usage_df["concept_id"] == concept_id]
    concept_demo = demo_df[demo_df["concept_id"] == concept_id]

    # Abandonment rate: abandoned_features / feature_clicks
    total_clicks = concept_usage["feature_clicks"].sum()
    total_abandoned = concept_usage["abandoned_features"].sum()
    abandonment_rate = total_abandoned / max(total_clicks, 1)

    # Mean objections
    mean_objections = concept_demo["objections_count"].mean() if len(concept_demo) > 0 else 0.0

    return {
        "abandonment_rate": float(abandonment_rate),
        "mean_objections": float(mean_objections),
    }


def _compute_pilot_signals(
    concept_id: str,
    commercial_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute customer pilot signals: max urgency and budget from named customers.

    Parameters
    ----------
    concept_id : str
        Concept to evaluate.
    commercial_df : pd.DataFrame
        Cleaned commercial_signals table.

    Returns
    -------
    dict[str, float]
        Keys: 'max_urgency', 'max_budget'.
    """
    concept_commercial = commercial_df[commercial_df["concept_id"] == concept_id]

    if len(concept_commercial) == 0:
        return {"max_urgency": 0.0, "max_budget": 0.0}

    return {
        "max_urgency": float(concept_commercial["urgency_score"].max()),
        "max_budget": float(concept_commercial["budget_signal"].max()),
    }


def classify_outcome(
    readiness_score: float,
    confidence_score: float,
    cluster_repeatability: float,
    delivery_complexity: float,
    revenue_potential: float,
    archive_signals: dict[str, float],
    pilot_signals: dict[str, float],
) -> str:
    """
    Apply the 5-outcome decision matrix to a single concept's scores.

    The matrix is evaluated in priority order:
    1. Archive (checked first — negative signals override positive scores)
    2. MVP Build
    3. Customer Pilot
    4. Reusable Asset
    5. Incubate (fallback for moderate-readiness low-confidence)
    6. Default: Incubate (safety net for unmatched cases)

    Parameters
    ----------
    readiness_score : float
        0–100 readiness score from GBR model.
    confidence_score : float
        0–100 statistical confidence score.
    cluster_repeatability : float
        0–1 cluster repeatability score.
    delivery_complexity : float
        0–1 delivery complexity from concepts table.
    revenue_potential : float
        0–1 revenue potential feature.
    archive_signals : dict[str, float]
        Keys: 'abandonment_rate', 'mean_objections'.
    pilot_signals : dict[str, float]
        Keys: 'max_urgency', 'max_budget'.

    Returns
    -------
    str
        One of the 5 outcome labels from config.py.
    """
    # --- 1. Archive check (negative signals override) ---
    if readiness_score < READINESS_MODERATE:
        return OUTCOME_ARCHIVE

    # High complexity relative to value
    if revenue_potential > 0 and delivery_complexity / max(revenue_potential, 0.01) > ARCHIVE_COMPLEXITY_VALUE_RATIO:
        return OUTCOME_ARCHIVE

    # Negative engagement trending
    if (
        archive_signals["abandonment_rate"] > ARCHIVE_ABANDONMENT_RATE
        and archive_signals["mean_objections"] > ARCHIVE_OBJECTION_COUNT
    ):
        return OUTCOME_ARCHIVE

    # --- 2. MVP Build ---
    if (
        readiness_score >= READINESS_HIGH
        and confidence_score >= CONFIDENCE_HIGH
        and delivery_complexity <= DELIVERY_COMPLEXITY_LOW_TO_MODERATE
    ):
        return OUTCOME_MVP_BUILD

    # --- 3. Reusable Asset ---
    if (
        readiness_score >= READINESS_HIGH
        and cluster_repeatability >= CLUSTER_REPEATABILITY_HIGH
    ):
        return OUTCOME_REUSABLE_ASSET

    # --- 4. Customer Pilot ---
    if readiness_score >= READINESS_HIGH and confidence_score >= CONFIDENCE_HIGH:
        # High readiness + high confidence + high complexity → pilot instead of MVP
        return OUTCOME_CUSTOMER_PILOT

    if (
        readiness_score >= READINESS_MODERATE
        and pilot_signals["max_urgency"] >= PILOT_URGENCY_THRESHOLD
        and pilot_signals["max_budget"] >= PILOT_BUDGET_THRESHOLD
    ):
        # Moderate readiness but strong named-customer urgency + budget
        return OUTCOME_CUSTOMER_PILOT

    # --- 5. Incubate ---
    if readiness_score >= READINESS_MODERATE and confidence_score < CONFIDENCE_HIGH:
        return OUTCOME_INCUBATE

    # --- Default fallback ---
    return OUTCOME_INCUBATE


def run_decision_engine(
    feature_df: pd.DataFrame,
    readiness_scores: pd.Series,
    confidence_scores: pd.Series,
    usage_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the decision engine across all concepts, producing the outcome table.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature table with cluster_repeatability_score, delivery_complexity, revenue_potential.
    readiness_scores : pd.Series
        Index = concept_id, values = readiness_score 0-100.
    confidence_scores : pd.Series
        Index = concept_id, values = confidence_score 0-100.
    usage_df, demo_df, commercial_df : pd.DataFrame
        Cleaned signal DataFrames for archive/pilot signal computation.

    Returns
    -------
    pd.DataFrame
        Columns: concept_id, readiness_score, confidence_score, cluster_repeatability,
                 delivery_complexity, recommended_outcome.
    """
    logger.info("Running decision engine on %d concepts...", len(feature_df))

    indexed = feature_df.set_index("concept_id") if "concept_id" in feature_df.columns else feature_df

    results = []
    for concept_id in indexed.index:
        row = indexed.loc[concept_id]
        readiness = float(readiness_scores.get(concept_id, 50.0))
        confidence = float(confidence_scores.get(concept_id, 50.0))
        cluster_repeat = float(row.get("cluster_repeatability_score", 0.0))
        complexity = float(row.get("delivery_complexity", 0.5))
        rev_potential = float(row.get("revenue_potential", 0.5))

        archive_signals = _compute_archive_signals(concept_id, usage_df, demo_df)
        pilot_signals = _compute_pilot_signals(concept_id, commercial_df)

        outcome = classify_outcome(
            readiness_score=readiness,
            confidence_score=confidence,
            cluster_repeatability=cluster_repeat,
            delivery_complexity=complexity,
            revenue_potential=rev_potential,
            archive_signals=archive_signals,
            pilot_signals=pilot_signals,
        )

        results.append(
            {
                "concept_id": concept_id,
                "readiness_score": readiness,
                "confidence_score": confidence,
                "cluster_repeatability": cluster_repeat,
                "delivery_complexity": complexity,
                "recommended_outcome": outcome,
            }
        )

    outcome_df = pd.DataFrame(results)

    logger.info(
        "Decision engine complete. Outcome distribution:\n%s",
        outcome_df["recommended_outcome"].value_counts().to_string(),
    )
    return outcome_df
