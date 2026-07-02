"""
readiness_model.py — GradientBoostingRegressor for commercialization readiness scoring.

Approach:
1. Bootstrap a synthetic training label from a documented rule baseline
   (weighted sum of demand + value + feasibility + strategic fit).
   The rule is a transparent supplement — not the sole method.
2. Train GradientBoostingRegressor on the 8 engineered features → synthetic label.
3. Output: readiness_score (0-100) per concept + feature importances.

Chosen over LinearRegression because it captures non-linear feature interactions
(e.g., high engagement + low decision-maker presence isn't simply additive)
and is compatible with SHAP TreeExplainer.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    LOG_FORMAT,
    LOG_LEVEL,
    RANDOM_SEED,
    READINESS_LABEL_NOISE_STD,
    READINESS_LABEL_WEIGHTS,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Features fed to the GBR model
MODEL_FEATURES: list[str] = [
    "demand_intensity",
    "repeatability",
    "engagement_depth",
    "segment_similarity_raw",
    "revenue_potential",
    "feasibility",
    "strategic_fit",
    "confidence_proxy",
]


def _build_synthetic_label(feature_df: pd.DataFrame) -> pd.Series:
    """
    Build a synthetic training label from the documented rule baseline.

    The label is a weighted sum of key features (all 0-1) plus small Gaussian
    noise to prevent the GBR from perfectly replicating the linear baseline.
    This gives the non-linear model room to learn interaction effects that
    the rule baseline cannot capture.

    Label formula (from READINESS_LABEL_WEIGHTS in config.py):
        label = Σ (weight_i × feature_i) + N(0, READINESS_LABEL_NOISE_STD)

    The final label is clipped to [0, 1] before scaling to [0, 100].

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature DataFrame with all required feature columns.

    Returns
    -------
    pd.Series
        Synthetic readiness label in [0, 1], indexed by concept_id.
    """
    np.random.seed(RANDOM_SEED)
    label = pd.Series(0.0, index=feature_df.index)

    for feature_name, weight in READINESS_LABEL_WEIGHTS.items():
        if feature_name in feature_df.columns:
            label += weight * feature_df[feature_name]
        else:
            logger.warning("Label baseline feature '%s' not found in feature_df.", feature_name)

    noise = np.random.normal(loc=0.0, scale=READINESS_LABEL_NOISE_STD, size=len(label))
    label = (label + noise).clip(0.0, 1.0)

    logger.debug(
        "Synthetic label stats — mean: %.4f, std: %.4f, min: %.4f, max: %.4f",
        label.mean(),
        label.std(),
        label.min(),
        label.max(),
    )
    return label


def train_readiness_model(
    feature_df: pd.DataFrame,
) -> tuple[GradientBoostingRegressor, np.ndarray, pd.DataFrame]:
    """
    Train a GradientBoostingRegressor to predict commercialisation readiness.

    Returns the trained model, the training feature matrix (X), and the
    full feature DataFrame with concept_id intact for downstream SHAP use.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Concept-level feature table from build_feature_table(), enriched by
        fit_clusters(). Must contain concept_id and all MODEL_FEATURES columns.

    Returns
    -------
    tuple[GradientBoostingRegressor, np.ndarray, pd.DataFrame]
        - Trained GBR model.
        - X: feature matrix (n_concepts × n_features) as numpy array.
        - X_df: feature DataFrame with concept_id index preserved.
    """
    available_features = [f for f in MODEL_FEATURES if f in feature_df.columns]
    missing = set(MODEL_FEATURES) - set(available_features)
    if missing:
        logger.warning("Missing model features — padding with 0.5: %s", missing)
        for f in missing:
            feature_df = feature_df.copy()
            feature_df[f] = 0.5

    indexed = feature_df.set_index("concept_id") if "concept_id" in feature_df.columns else feature_df

    x_df = indexed[MODEL_FEATURES].fillna(0.0)
    x_matrix = x_df.values

    # Build synthetic label
    synthetic_label = _build_synthetic_label(x_df)

    # Train GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )
    model.fit(x_matrix, synthetic_label.values)

    # Cross-validation for logging (internal quality check)
    cv_scores = cross_val_score(model, x_matrix, synthetic_label.values, cv=3, scoring="r2")
    logger.info(
        "GBR cross-validation R² (3-fold): %.4f ± %.4f",
        cv_scores.mean(),
        cv_scores.std(),
    )

    logger.info(
        "GBR feature importances: %s",
        dict(zip(MODEL_FEATURES, model.feature_importances_.round(4))),
    )

    return model, x_matrix, x_df


def predict_readiness_scores(
    model: GradientBoostingRegressor,
    x_matrix: np.ndarray,
    concept_ids: list[str],
) -> pd.Series:
    """
    Predict readiness scores (0–100) for all concepts using the trained model.

    The raw [0, 1] predictions are scaled to [0, 100] for readability in the
    dashboard and stakeholder reports.

    Parameters
    ----------
    model : GradientBoostingRegressor
        Trained readiness model.
    x_matrix : np.ndarray
        Feature matrix used during training (or new concept feature matrix).
    concept_ids : list[str]
        Concept IDs in the same order as x_matrix rows.

    Returns
    -------
    pd.Series
        Index = concept_id, values = readiness_score in [0, 100].
    """
    raw_predictions = model.predict(x_matrix)

    # Scale predictions from model's [0,1] output range to [0, 100]
    scaler = MinMaxScaler(feature_range=(0.0, 100.0))
    scaled_scores = scaler.fit_transform(raw_predictions.reshape(-1, 1)).flatten()

    readiness_series = pd.Series(
        scaled_scores.round(2),
        index=concept_ids,
        name="readiness_score",
    )

    logger.info(
        "Readiness scores — mean: %.2f, min: %.2f, max: %.2f",
        readiness_series.mean(),
        readiness_series.min(),
        readiness_series.max(),
    )
    return readiness_series


def get_feature_importance_df(
    model: GradientBoostingRegressor,
    feature_names: list[str] = MODEL_FEATURES,
) -> pd.DataFrame:
    """
    Return a sorted DataFrame of global feature importances from the trained GBR.

    Parameters
    ----------
    model : GradientBoostingRegressor
        Trained readiness model.
    feature_names : list[str]
        Names of the features in order.

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance. Sorted descending by importance.
    """
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importance_df


if __name__ == "__main__":
    from src.data_pipeline.clean_validate import load_and_clean_all
    from src.data_pipeline.load_to_db import run_pipeline
    from src.features.feature_engineering import build_feature_table
    from src.models.clustering import fit_clusters

    run_pipeline()
    data = load_and_clean_all()
    features = build_feature_table(
        data["concepts"], data["demo_signals"], data["usage_signals"], data["commercial_signals"]
    )
    features = fit_clusters(features)
    model, X, X_df = train_readiness_model(features)
    scores = predict_readiness_scores(model, X, X_df.index.tolist())
    print(scores.sort_values(ascending=False))
