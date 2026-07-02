"""
config.py — Centralized configuration for thresholds, weights, and constants.

All magic numbers used across the commercialization engine live here.
No logic file should contain hard-coded numeric thresholds.
"""

import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# ---------------------------------------------------------------------------
# Random seed (applied to NumPy, random, scikit-learn for reproducibility)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Data generation parameters
# ---------------------------------------------------------------------------
NUM_CONCEPTS: int = 20          # number of AI product concepts to simulate
NUM_CUSTOMERS: int = 50         # unique customer IDs in the synthetic dataset
NUM_DEMO_RECORDS: int = 120     # total demo signal rows
NUM_USAGE_RECORDS: int = 120    # total usage signal rows
NUM_COMMERCIAL_RECORDS: int = 120
NUM_FEEDBACK_RECORDS: int = 80

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: str = "data/commercialization.db"

# ---------------------------------------------------------------------------
# Feature engineering weights
# ---------------------------------------------------------------------------

# demand_intensity: weighted average of feedback, follow-up rate, urgency
DEMAND_WEIGHTS: dict = {
    "feedback_score_norm": 0.40,   # normalised 0-10 feedback → 0-1
    "follow_up_rate": 0.35,        # fraction of demos where follow-up requested
    "urgency_score": 0.25,         # mean urgency from commercial signals
}

# repeatability: blend of normalised repeat usage days and session count
REPEATABILITY_WEIGHTS: dict = {
    "repeat_usage_days_norm": 0.60,
    "trial_sessions_norm": 0.40,
}

# engagement_depth: blend of clicks, time spent, active users
ENGAGEMENT_WEIGHTS: dict = {
    "feature_clicks_norm": 0.35,
    "time_spent_norm": 0.40,
    "active_users_norm": 0.25,
}

# revenue_potential: blend of WTP and expected_value
REVENUE_WEIGHTS: dict = {
    "willingness_to_pay_norm": 0.50,
    "expected_value_norm": 0.50,
}

# ---------------------------------------------------------------------------
# Synthetic readiness label baseline (rule-based supplement for training)
# ---------------------------------------------------------------------------
READINESS_LABEL_WEIGHTS: dict = {
    "demand_intensity": 0.25,
    "repeatability": 0.15,
    "engagement_depth": 0.15,
    "revenue_potential": 0.20,
    "feasibility": 0.15,
    "strategic_fit": 0.10,
}

READINESS_LABEL_NOISE_STD: float = 0.05  # Gaussian noise added to the label

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
CLUSTER_K_MIN: int = 2
CLUSTER_K_MAX: int = 6
CLUSTER_FEATURES: list = [
    "demand_intensity",
    "repeatability",
    "engagement_depth",
    "segment_similarity_raw",  # pre-cluster placeholder (demo-segment entropy)
]

# ---------------------------------------------------------------------------
# Decision engine thresholds (all 0-100 scale for readiness / confidence)
# ---------------------------------------------------------------------------

# Readiness score bands
READINESS_HIGH: float = 65.0       # ≥ this → "high"
READINESS_MODERATE: float = 40.0   # ≥ this and < HIGH → "moderate"
# < READINESS_MODERATE → "low"

# Confidence score bands
CONFIDENCE_HIGH: float = 60.0
CONFIDENCE_LOW: float = 35.0

# Delivery complexity bands (0-1 scale, from concepts table)
DELIVERY_COMPLEXITY_LOW_TO_MODERATE: float = 0.55  # ≤ this → low-to-moderate
DELIVERY_COMPLEXITY_HIGH: float = 0.55             # > this → high

# Cluster repeatability threshold (0-1)
CLUSTER_REPEATABILITY_HIGH: float = 0.55

# Customer Pilot: named urgency + budget thresholds (0-1)
PILOT_URGENCY_THRESHOLD: float = 0.65
PILOT_BUDGET_THRESHOLD: float = 0.55

# Archive: abandonment rate threshold (abandoned_features / feature_clicks)
ARCHIVE_ABANDONMENT_RATE: float = 0.40
# Archive: mean objections count threshold
ARCHIVE_OBJECTION_COUNT: float = 3.5
# Archive: value/complexity ratio — archive if complexity too high vs expected value
ARCHIVE_COMPLEXITY_VALUE_RATIO: float = 1.5  # delivery_complexity / revenue_potential

# ---------------------------------------------------------------------------
# Confidence scoring parameters
# ---------------------------------------------------------------------------
CONFIDENCE_MIN_SAMPLES: int = 3    # concepts with fewer samples → low confidence
CONFIDENCE_VARIANCE_SCALE: float = 4.0  # scaling factor for variance penalty

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
SHAP_TOP_N_FEATURES: int = 4  # number of top features cited in narrative

# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------
NARRATIVE_HIGH_THRESHOLD: float = 0.60   # SHAP contribution fraction for "strong"
NARRATIVE_LOW_THRESHOLD: float = 0.20    # SHAP contribution fraction for "weak"

# ---------------------------------------------------------------------------
# Outcome labels (single source of truth)
# ---------------------------------------------------------------------------
OUTCOME_MVP_BUILD: str = "MVP Build"
OUTCOME_CUSTOMER_PILOT: str = "Customer Pilot"
OUTCOME_REUSABLE_ASSET: str = "Reusable Asset"
OUTCOME_INCUBATE: str = "Incubate"
OUTCOME_ARCHIVE: str = "Archive"

ALL_OUTCOMES: list = [
    OUTCOME_MVP_BUILD,
    OUTCOME_CUSTOMER_PILOT,
    OUTCOME_REUSABLE_ASSET,
    OUTCOME_INCUBATE,
    OUTCOME_ARCHIVE,
]

# Outcome badge colors for Streamlit (CSS color strings)
OUTCOME_COLORS: dict = {
    OUTCOME_MVP_BUILD: "#27ae60",        # green
    OUTCOME_CUSTOMER_PILOT: "#2980b9",   # blue
    OUTCOME_REUSABLE_ASSET: "#16a085",   # teal
    OUTCOME_INCUBATE: "#f39c12",         # amber
    OUTCOME_ARCHIVE: "#c0392b",          # red
}
