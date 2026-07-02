"""
generate_demo_signals.py — Generate synthetic customer demo signal records.

Produces realistic, non-uniform distributions:
- feedback_score: Beta-distributed (skewed toward 6-8 range, not perfectly uniform)
- objections_count: Poisson-distributed (λ=2, as specified)
- follow_up_requested / decision_maker_present: Bernoulli with concept-varying probability
"""

import logging
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.config import (
    LOG_FORMAT,
    LOG_LEVEL,
    NUM_CONCEPTS,
    NUM_CUSTOMERS,
    NUM_DEMO_RECORDS,
    RANDOM_SEED,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

fake = Faker()
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

SEGMENTS: list[str] = [
    "Enterprise",
    "Mid-Market",
    "SMB",
    "Government",
    "Startup",
]

DEMO_START_DATE: date = date(2024, 1, 1)
DEMO_END_DATE: date = date(2025, 3, 31)


def _random_date(start: date, end: date) -> str:
    """Return a random ISO date string between start and end inclusive."""
    delta_days = (end - start).days
    return (start + timedelta(days=random.randint(0, delta_days))).isoformat()


def generate_demo_signals(
    concept_ids: list[str],
    num_records: int = NUM_DEMO_RECORDS,
    num_customers: int = NUM_CUSTOMERS,
) -> pd.DataFrame:
    """
    Generate synthetic demo signal records for the provided concept IDs.

    Distributions used:
    - feedback_score: Beta(α=6, β=3) scaled to [0, 10] — skewed toward 6-9,
      reflecting that demos shown to potential customers tend to get favourable
      but not perfect scores.
    - objections_count: Poisson(λ=2) — right-skewed, most demos see 0-3 objections.
    - follow_up_requested: Bernoulli(p=0.55) — slight majority request follow-up.
    - decision_maker_present: Bernoulli(p=0.40) — minority of demos have DM present.

    Parameters
    ----------
    concept_ids : list[str]
        List of concept IDs to assign records to (sampled with replacement).
    num_records : int
        Total number of demo signal rows to generate.
    num_customers : int
        Pool of unique customer IDs to draw from.

    Returns
    -------
    pd.DataFrame
        Columns matching the demo_signals table schema.
    """
    customer_pool = [f"CUST{str(i + 1).zfill(4)}" for i in range(num_customers)]

    # Skewed feedback — Beta(6,3) → mean ≈ 0.67 → × 10 ≈ 6.7 average
    raw_feedback = np.random.beta(a=6, b=3, size=num_records) * 10.0

    # Poisson objections
    objections = np.random.poisson(lam=2, size=num_records)

    rows = []
    for i in range(num_records):
        concept_id = random.choice(concept_ids)
        customer_id = random.choice(customer_pool)
        segment = random.choice(SEGMENTS)
        demo_date = _random_date(DEMO_START_DATE, DEMO_END_DATE)
        feedback_score = round(float(np.clip(raw_feedback[i], 0.0, 10.0)), 2)
        follow_up_requested = bool(np.random.binomial(n=1, p=0.55))
        decision_maker_present = bool(np.random.binomial(n=1, p=0.40))
        objections_count = int(objections[i])

        rows.append(
            {
                "concept_id": concept_id,
                "customer_id": customer_id,
                "segment": segment,
                "demo_date": demo_date,
                "feedback_score": feedback_score,
                "follow_up_requested": follow_up_requested,
                "decision_maker_present": decision_maker_present,
                "objections_count": objections_count,
            }
        )

    demo_df = pd.DataFrame(rows)
    logger.info(
        "Generated %d demo signal records across %d concepts.",
        len(demo_df),
        demo_df["concept_id"].nunique(),
    )
    return demo_df


if __name__ == "__main__":
    concept_ids = [f"C{str(i + 1).zfill(3)}" for i in range(NUM_CONCEPTS)]
    df = generate_demo_signals(concept_ids)
    print(df.head(10).to_string())
