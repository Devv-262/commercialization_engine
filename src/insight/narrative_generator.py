"""
narrative_generator.py — Deterministic template-based natural language generation.

Converts SHAP output + decision outcome into plain-English paragraphs citing
specific evidence. NO external LLM API calls — fully reproducible and free to run.

This module is part of the AI insight layer — it takes ML output and produces
business-readable explanations.
"""

import logging

import pandas as pd

from src.config import (
    LOG_FORMAT,
    LOG_LEVEL,
    NARRATIVE_HIGH_THRESHOLD,
    NARRATIVE_LOW_THRESHOLD,
    OUTCOME_ARCHIVE,
    OUTCOME_CUSTOMER_PILOT,
    OUTCOME_INCUBATE,
    OUTCOME_MVP_BUILD,
    OUTCOME_REUSABLE_ASSET,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature display names (human-readable labels)
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "demand_intensity": "customer demand intensity",
    "repeatability": "repeat usage patterns",
    "engagement_depth": "feature engagement depth",
    "segment_similarity_raw": "cross-segment appeal",
    "revenue_potential": "revenue potential",
    "feasibility": "delivery feasibility",
    "strategic_fit": "strategic alignment",
    "confidence_proxy": "data confidence level",
}

# ---------------------------------------------------------------------------
# Outcome-specific framing templates
# ---------------------------------------------------------------------------
OUTCOME_FRAMES: dict[str, str] = {
    OUTCOME_MVP_BUILD: (
        "This concept is recommended for **MVP Build**. It shows strong commercial "
        "readiness with high confidence and manageable delivery complexity. "
    ),
    OUTCOME_CUSTOMER_PILOT: (
        "This concept is recommended for **Customer Pilot**. There is sufficient "
        "readiness signal, and at least one named customer shows strong urgency "
        "and budget alignment to support a targeted pilot engagement. "
    ),
    OUTCOME_REUSABLE_ASSET: (
        "This concept is recommended as a **Reusable Asset**. It demonstrates "
        "strong readiness and consistent demand patterns across multiple market "
        "segments, making it suitable for a platform-level investment. "
    ),
    OUTCOME_INCUBATE: (
        "This concept is recommended for **Incubation**. While showing moderate "
        "promise, the current evidence base is insufficient for a full build "
        "decision. More customer validation and signal collection is advised. "
    ),
    OUTCOME_ARCHIVE: (
        "This concept is recommended for **Archive**. Current signals indicate "
        "low commercial readiness, negative engagement trends, or unfavourable "
        "complexity-to-value economics. Resources should be redirected elsewhere. "
    ),
}


def _describe_feature_contribution(
    feature_name: str,
    shap_value: float,
    feature_value: float,
    direction: str,
) -> str:
    """
    Generate a single-sentence description of one feature's contribution.

    Parameters
    ----------
    feature_name : str
        Raw feature name (e.g. 'demand_intensity').
    shap_value : float
        SHAP contribution value.
    feature_value : float
        Actual feature value for this concept.
    direction : str
        'positive' or 'negative'.

    Returns
    -------
    str
        A sentence fragment describing this contribution.
    """
    display_name = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace("_", " "))
    strength = abs(shap_value)

    if strength >= NARRATIVE_HIGH_THRESHOLD:
        magnitude = "strongly"
    elif strength >= NARRATIVE_LOW_THRESHOLD:
        magnitude = "moderately"
    else:
        magnitude = "marginally"

    if direction == "positive":
        return (
            f"{display_name} ({feature_value:.2f}) {magnitude} supports "
            f"commercialisation readiness"
        )
    else:
        return (
            f"{display_name} ({feature_value:.2f}) {magnitude} weakens "
            f"the readiness case"
        )


def generate_concept_narrative(
    concept_id: str,
    concept_name: str,
    outcome: str,
    readiness_score: float,
    confidence_score: float,
    top_features: list[dict],
) -> str:
    """
    Generate a full business-readable narrative for a single concept.

    Combines:
    - Outcome framing (from templates)
    - Readiness/confidence scores
    - Top SHAP feature contributions with direction and magnitude
    - Actionable next-step recommendation

    Parameters
    ----------
    concept_id : str
        Concept identifier.
    concept_name : str
        Human-readable concept name.
    outcome : str
        One of the 5 outcome labels.
    readiness_score : float
        0–100 readiness score.
    confidence_score : float
        0–100 confidence score.
    top_features : list[dict]
        Top SHAP feature contributions from shap_explainer.get_top_features_per_concept().

    Returns
    -------
    str
        Multi-sentence plain-English narrative.
    """
    # Outcome frame
    frame = OUTCOME_FRAMES.get(outcome, f"This concept is classified as **{outcome}**. ")

    # Score summary
    score_summary = (
        f"**{concept_name}** ({concept_id}) scores **{readiness_score:.1f}/100** on "
        f"commercialisation readiness with a confidence level of **{confidence_score:.1f}/100**. "
    )

    # Feature evidence
    if top_features:
        positive_features = [f for f in top_features if f["direction"] == "positive"]
        negative_features = [f for f in top_features if f["direction"] == "negative"]

        evidence_parts = []
        if positive_features:
            pos_descriptions = [
                _describe_feature_contribution(
                    f["feature"], f["shap_value"], f["feature_value"], f["direction"]
                )
                for f in positive_features
            ]
            evidence_parts.append("Key strengths: " + "; ".join(pos_descriptions) + ".")

        if negative_features:
            neg_descriptions = [
                _describe_feature_contribution(
                    f["feature"], f["shap_value"], f["feature_value"], f["direction"]
                )
                for f in negative_features
            ]
            evidence_parts.append("Key risks: " + "; ".join(neg_descriptions) + ".")

        evidence_text = " ".join(evidence_parts)
    else:
        evidence_text = "Insufficient SHAP data to detail feature contributions."

    # Combine
    narrative = f"{frame}{score_summary}{evidence_text}"
    return narrative


def generate_all_narratives(
    outcome_df: pd.DataFrame,
    shap_features: dict[str, list[dict]],
    concepts_df: pd.DataFrame,
) -> dict[str, str]:
    """
    Generate narratives for all concepts.

    Parameters
    ----------
    outcome_df : pd.DataFrame
        Output from decision_engine.run_decision_engine().
    shap_features : dict[str, list[dict]]
        Output from shap_explainer.get_top_features_per_concept().
    concepts_df : pd.DataFrame
        Cleaned concepts table (for concept_name lookup).

    Returns
    -------
    dict[str, str]
        Keys = concept_id, values = narrative string.
    """
    name_lookup = concepts_df.set_index("concept_id")["concept_name"].to_dict()
    narratives: dict[str, str] = {}

    for _, row in outcome_df.iterrows():
        concept_id = row["concept_id"]
        narrative = generate_concept_narrative(
            concept_id=concept_id,
            concept_name=name_lookup.get(concept_id, concept_id),
            outcome=row["recommended_outcome"],
            readiness_score=row["readiness_score"],
            confidence_score=row["confidence_score"],
            top_features=shap_features.get(concept_id, []),
        )
        narratives[concept_id] = narrative

    logger.info("Generated narratives for %d concepts.", len(narratives))
    return narratives


def generate_executive_summary(
    outcome_df: pd.DataFrame,
    concepts_df: pd.DataFrame,
) -> str:
    """
    Generate an executive summary paragraph for stakeholder reporting.

    Summarises: how many concepts per outcome, top MVP candidates, top archive
    candidates, and overall portfolio readiness.

    Parameters
    ----------
    outcome_df : pd.DataFrame
        Decision engine output.
    concepts_df : pd.DataFrame
        Concepts table for name lookup.

    Returns
    -------
    str
        Multi-paragraph executive summary string.
    """
    name_lookup = concepts_df.set_index("concept_id")["concept_name"].to_dict()
    outcome_counts = outcome_df["recommended_outcome"].value_counts().to_dict()
    total = len(outcome_df)
    mean_readiness = outcome_df["readiness_score"].mean()

    summary_lines = [
        f"# Executive Summary — AI/ML Commercialisation Portfolio\n",
        f"**Portfolio Size:** {total} concepts evaluated\n",
        f"**Mean Readiness Score:** {mean_readiness:.1f}/100\n\n",
        "## Outcome Distribution\n",
    ]

    for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        summary_lines.append(f"- **{outcome}**: {count} concepts ({pct:.0f}%)\n")

    # Top MVP candidates
    mvp_candidates = outcome_df[outcome_df["recommended_outcome"] == OUTCOME_MVP_BUILD].sort_values(
        "readiness_score", ascending=False
    )
    if len(mvp_candidates) > 0:
        summary_lines.append("\n## Recommended for Immediate Investment (MVP Build)\n")
        for _, row in mvp_candidates.head(5).iterrows():
            name = name_lookup.get(row["concept_id"], row["concept_id"])
            summary_lines.append(
                f"- **{name}** — Readiness: {row['readiness_score']:.1f}, "
                f"Confidence: {row['confidence_score']:.1f}\n"
            )

    # Archive candidates
    archive_candidates = outcome_df[outcome_df["recommended_outcome"] == OUTCOME_ARCHIVE].sort_values(
        "readiness_score", ascending=True
    )
    if len(archive_candidates) > 0:
        summary_lines.append("\n## Recommended for Archive (Redirect Resources)\n")
        for _, row in archive_candidates.head(5).iterrows():
            name = name_lookup.get(row["concept_id"], row["concept_id"])
            summary_lines.append(
                f"- **{name}** — Readiness: {row['readiness_score']:.1f}, "
                f"Confidence: {row['confidence_score']:.1f}\n"
            )

    # Closing recommendation
    summary_lines.append(
        "\n## Strategic Recommendation\n"
        f"The portfolio shows a mean readiness of {mean_readiness:.1f}/100. "
    )
    n_actionable = outcome_counts.get(OUTCOME_MVP_BUILD, 0) + outcome_counts.get(OUTCOME_CUSTOMER_PILOT, 0)
    summary_lines.append(
        f"{n_actionable} concepts ({n_actionable / total * 100:.0f}%) are actionable "
        f"for immediate build or pilot engagement. "
    )
    n_incubate = outcome_counts.get(OUTCOME_INCUBATE, 0)
    if n_incubate > 0:
        summary_lines.append(
            f"{n_incubate} concepts require further incubation with additional "
            f"customer validation before committing resources."
        )

    return "".join(summary_lines)
