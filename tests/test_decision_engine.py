"""
test_decision_engine.py — Unit tests covering all 5 outcomes in the decision matrix.
"""

import pytest

from src.config import (
    OUTCOME_ARCHIVE,
    OUTCOME_CUSTOMER_PILOT,
    OUTCOME_INCUBATE,
    OUTCOME_MVP_BUILD,
    OUTCOME_REUSABLE_ASSET,
)
from src.insight.decision_engine import classify_outcome


def test_classify_outcome_archive_due_to_readiness():
    # Low readiness unconditionally falls to Archive
    outcome = classify_outcome(
        readiness_score=30.0,
        confidence_score=90.0,
        cluster_repeatability=0.8,
        delivery_complexity=0.2,
        revenue_potential=0.5,
        archive_signals={"abandonment_rate": 0.1, "mean_objections": 1.0},
        pilot_signals={"max_urgency": 0.0, "max_budget": 0.0},
    )
    assert outcome == OUTCOME_ARCHIVE


def test_classify_outcome_archive_due_to_negative_engagement():
    # High readiness but terrible engagement trends
    outcome = classify_outcome(
        readiness_score=80.0,
        confidence_score=90.0,
        cluster_repeatability=0.8,
        delivery_complexity=0.2,
        revenue_potential=0.5,
        archive_signals={"abandonment_rate": 0.6, "mean_objections": 5.0},
        pilot_signals={"max_urgency": 0.0, "max_budget": 0.0},
    )
    assert outcome == OUTCOME_ARCHIVE


def test_classify_outcome_mvp_build():
    # High readiness + high confidence + low complexity
    outcome = classify_outcome(
        readiness_score=80.0,
        confidence_score=80.0,
        cluster_repeatability=0.4,
        delivery_complexity=0.3,
        revenue_potential=0.5,
        archive_signals={"abandonment_rate": 0.1, "mean_objections": 1.0},
        pilot_signals={"max_urgency": 0.0, "max_budget": 0.0},
    )
    assert outcome == OUTCOME_MVP_BUILD


def test_classify_outcome_reusable_asset():
    # High readiness + high cluster repeatability
    # Note: delivery complexity doesn't matter as much here
    outcome = classify_outcome(
        readiness_score=80.0,
        confidence_score=80.0,
        cluster_repeatability=0.8,
        delivery_complexity=0.7,
        revenue_potential=0.5,
        archive_signals={"abandonment_rate": 0.1, "mean_objections": 1.0},
        pilot_signals={"max_urgency": 0.0, "max_budget": 0.0},
    )
    assert outcome == OUTCOME_REUSABLE_ASSET


def test_classify_outcome_customer_pilot_high_complexity():
    # High readiness + high confidence + HIGH complexity = Pilot (too big for MVP)
    outcome = classify_outcome(
        readiness_score=80.0,
        confidence_score=80.0,
        cluster_repeatability=0.4,
        delivery_complexity=0.9,
        revenue_potential=0.9,
        archive_signals={"abandonment_rate": 0.1, "mean_objections": 1.0},
        pilot_signals={"max_urgency": 0.0, "max_budget": 0.0},
    )
    assert outcome == OUTCOME_CUSTOMER_PILOT


def test_classify_outcome_customer_pilot_named_customer():
    # Moderate readiness but strong named customer urgency/budget
    outcome = classify_outcome(
        readiness_score=50.0,
        confidence_score=80.0,
        cluster_repeatability=0.4,
        delivery_complexity=0.4,
        revenue_potential=0.5,
        archive_signals={"abandonment_rate": 0.1, "mean_objections": 1.0},
        pilot_signals={"max_urgency": 0.9, "max_budget": 0.8},
    )
    assert outcome == OUTCOME_CUSTOMER_PILOT


def test_classify_outcome_incubate_low_confidence():
    # Moderate readiness but low confidence (e.g. small sample size)
    outcome = classify_outcome(
        readiness_score=50.0,
        confidence_score=30.0,
        cluster_repeatability=0.4,
        delivery_complexity=0.4,
        revenue_potential=0.5,
        archive_signals={"abandonment_rate": 0.1, "mean_objections": 1.0},
        pilot_signals={"max_urgency": 0.0, "max_budget": 0.0},
    )
    assert outcome == OUTCOME_INCUBATE
