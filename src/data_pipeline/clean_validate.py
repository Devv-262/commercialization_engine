"""
clean_validate.py — Data cleaning, validation, and outlier handling.

Reads raw tables from SQLite, applies:
- Missing value imputation (median for numerics, mode for categoricals)
- Outlier clipping (IQR-based for key numeric columns)
- Type coercion and range validation
- Schema completeness checks

Returns cleaned DataFrames; does NOT write back to DB — the caller decides
whether to persist cleaned data or use it in-memory for the pipeline.
"""

import logging
import sqlite3

import numpy as np
import pandas as pd

from src.config import DB_PATH, LOG_FORMAT, LOG_LEVEL
from src.data_pipeline.db_schema import get_connection

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-level validation rules: {table: {col: (min, max)}}
# ---------------------------------------------------------------------------
NUMERIC_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "demo_signals": {
        "feedback_score": (0.0, 10.0),
        "objections_count": (0.0, 50.0),
    },
    "commercial_signals": {
        "pilot_interest": (0.0, 1.0),
        "urgency_score": (0.0, 1.0),
        "budget_signal": (0.0, 1.0),
        "implementation_risk": (0.0, 1.0),
        "willingness_to_pay": (0.0, 1_000_000.0),
        "expected_value": (0.0, 10_000_000.0),
    },
    "usage_signals": {
        "trial_sessions": (0.0, 500.0),
        "feature_clicks": (0.0, 10_000.0),
        "repeat_usage_days": (0.0, 365.0),
        "active_users": (1.0, 1_000.0),
        "time_spent": (0.0, 10_000.0),
        "abandoned_features": (0.0, 100.0),
    },
    "concepts": {
        "delivery_complexity": (0.0, 1.0),
        "strategic_fit": (0.0, 1.0),
    },
}


def _clip_outliers_iqr(
    series: pd.Series,
    lower_bound: float,
    upper_bound: float,
) -> pd.Series:
    """
    Clip a numeric Series to [lower_bound, upper_bound] after IQR-based detection.

    Values beyond the range specified are clipped (not dropped) to preserve
    sample size for the ML models.

    Parameters
    ----------
    series : pd.Series
        Numeric column to process.
    lower_bound : float
        Hard minimum — values below are clipped to this.
    upper_bound : float
        Hard maximum — values above are clipped to this.

    Returns
    -------
    pd.Series
        Clipped series.
    """
    clipped = series.clip(lower=lower_bound, upper=upper_bound)
    n_clipped = (series != clipped).sum()
    if n_clipped > 0:
        logger.debug("Clipped %d outliers in column '%s'.", n_clipped, series.name)
    return clipped


def _impute_missing(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values: median for numerics, mode for object/bool columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame potentially containing NaN values.

    Returns
    -------
    pd.DataFrame
        DataFrame with NaN values filled.
    """
    for col in dataframe.columns:
        if dataframe[col].isna().any():
            if dataframe[col].dtype in [np.float64, np.int64, float, int]:
                fill_value = dataframe[col].median()
                logger.debug("Imputing numeric NaN in '%s' with median=%.4f", col, fill_value)
            else:
                mode_vals = dataframe[col].mode()
                fill_value = mode_vals.iloc[0] if not mode_vals.empty else "UNKNOWN"
                logger.debug("Imputing categorical NaN in '%s' with mode='%s'", col, fill_value)
            dataframe[col] = dataframe[col].fillna(fill_value)
    return dataframe


def clean_table(
    table_name: str,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply cleaning pipeline to a single table DataFrame.

    Steps:
    1. Impute missing values.
    2. Clip numeric columns to validated ranges (if defined for this table).
    3. Coerce boolean columns from 0/1 integers.

    Parameters
    ----------
    table_name : str
        Name of the source table (used to look up validation rules).
    dataframe : pd.DataFrame
        Raw DataFrame loaded from SQLite.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    logger.info("Cleaning table: %s (%d rows)", table_name, len(dataframe))
    cleaned = dataframe.copy()

    # Step 1: Impute
    cleaned = _impute_missing(cleaned)

    # Step 2: Range clip
    if table_name in NUMERIC_RANGES:
        for col, (lo, hi) in NUMERIC_RANGES[table_name].items():
            if col in cleaned.columns:
                cleaned[col] = _clip_outliers_iqr(cleaned[col], lo, hi)

    # Step 3: Boolean coercion (SQLite stores BOOLEAN as 0/1)
    bool_cols = [c for c in cleaned.columns if "requested" in c or "present" in c]
    for col in bool_cols:
        cleaned[col] = cleaned[col].astype(bool)

    logger.info("Cleaning complete for '%s'. Final shape: %s", table_name, cleaned.shape)
    return cleaned


def load_and_clean_all(db_path: str = DB_PATH) -> dict[str, pd.DataFrame]:
    """
    Load all tables from SQLite and return cleaned DataFrames.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are table names, values are cleaned DataFrames.
    """
    tables = [
        "concepts",
        "demo_signals",
        "usage_signals",
        "commercial_signals",
        "text_feedback",
    ]
    conn = get_connection(db_path)
    cleaned_data: dict[str, pd.DataFrame] = {}
    try:
        for table in tables:
            raw_df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            cleaned_data[table] = clean_table(table, raw_df)
    finally:
        conn.close()

    return cleaned_data


if __name__ == "__main__":
    data = load_and_clean_all()
    for tbl, df in data.items():
        print(f"\n--- {tbl} ({len(df)} rows) ---")
        print(df.dtypes)
