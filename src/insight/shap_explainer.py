"""
shap_explainer.py — SHAP-based explainability for the readiness model.

Runs shap.TreeExplainer on the trained GradientBoostingRegressor to produce
per-concept SHAP values (which features pushed the score up/down and by how much).

This module is part of the AI insight layer but contains no narrative logic.
"""

import logging

import numpy as np
import pandas as pd
import shap

from src.config import LOG_FORMAT, LOG_LEVEL, SHAP_TOP_N_FEATURES
from src.models.readiness_model import MODEL_FEATURES

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def compute_shap_values(
    model,
    x_matrix: np.ndarray,
    feature_names: list[str] = None,
) -> shap.Explanation:
    """
    Compute SHAP values for all concepts using TreeExplainer.

    Parameters
    ----------
    model : GradientBoostingRegressor
        Trained readiness model.
    x_matrix : np.ndarray
        Feature matrix (n_concepts x n_features).
    feature_names : list[str], optional
        Feature names. Defaults to MODEL_FEATURES.

    Returns
    -------
    shap.Explanation
        SHAP explanation object with .values, .base_values, .data attributes.
    """
    if feature_names is None:
        feature_names = MODEL_FEATURES

    logger.info("Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_matrix)
    shap_values.feature_names = feature_names

    logger.info(
        "SHAP values computed. Shape: %s, base_value: %.4f",
        shap_values.values.shape,
        float(np.mean(shap_values.base_values)),
    )
    return shap_values


def get_top_features_per_concept(
    shap_values: shap.Explanation,
    concept_ids: list[str],
    top_n: int = SHAP_TOP_N_FEATURES,
) -> dict[str, list[dict]]:
    """
    Extract the top N contributing features per concept from SHAP values.

    For each concept returns a list of dicts with:
    - feature: str — feature name
    - shap_value: float — SHAP contribution (positive = pushes score up)
    - feature_value: float — actual feature value for this concept
    - direction: str — "positive" or "negative"

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation from compute_shap_values().
    concept_ids : list[str]
        Concept IDs in the same order as x_matrix rows.
    top_n : int
        Number of top features to return per concept.

    Returns
    -------
    dict[str, list[dict]]
        Keys = concept_id, values = list of top feature contribution dicts.
    """
    feature_names = shap_values.feature_names
    result: dict[str, list[dict]] = {}

    for idx, concept_id in enumerate(concept_ids):
        concept_shap = shap_values.values[idx]
        concept_data = shap_values.data[idx]

        # Sort by absolute SHAP value descending
        sorted_indices = np.argsort(np.abs(concept_shap))[::-1][:top_n]

        top_features = []
        for feat_idx in sorted_indices:
            sv = float(concept_shap[feat_idx])
            top_features.append(
                {
                    "feature": feature_names[feat_idx],
                    "shap_value": round(sv, 4),
                    "feature_value": round(float(concept_data[feat_idx]), 4),
                    "direction": "positive" if sv > 0 else "negative",
                }
            )
        result[concept_id] = top_features

    logger.info("Extracted top-%d SHAP features for %d concepts.", top_n, len(result))
    return result


def get_global_feature_importance(
    shap_values: shap.Explanation,
) -> pd.DataFrame:
    """
    Compute global feature importance from mean absolute SHAP values.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation from compute_shap_values().

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_abs_shap. Sorted descending.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": shap_values.feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return importance_df
