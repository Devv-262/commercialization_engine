"""
generate_commercial_signals.py — Generate synthetic commercial signal records.

Reflects real-world variance in commercial readiness signals:
- pilot_interest, urgency_score, budget_signal, implementation_risk: Beta-distributed
- willingness_to_pay: LogNormal — heavy-tailed, realistic enterprise WTP variance
- expected_value: LogNormal — similarly distributed, correlated loosely with WTP
"""

import logging
import random

import numpy as np
import pandas as pd
from faker import Faker

from src.config import (
    LOG_FORMAT,
    LOG_LEVEL,
    NUM_CONCEPTS,
    NUM_CUSTOMERS,
    NUM_COMMERCIAL_RECORDS,
    RANDOM_SEED,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

fake = Faker()
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def generate_commercial_signals(
    concept_ids: list[str],
    num_records: int = NUM_COMMERCIAL_RECORDS,
    num_customers: int = NUM_CUSTOMERS,
) -> pd.DataFrame:
    """
    Generate synthetic commercial signal records for the provided concept IDs.

    Distributions used:
    - pilot_interest:       Beta(3, 2) — skewed toward moderate-high interest.
    - urgency_score:        Beta(2, 3) — skewed toward lower urgency (realistic).
    - budget_signal:        Beta(2, 4) — budget is typically the limiting factor.
    - implementation_risk:  Beta(2, 3) — most concepts have manageable risk.
    - willingness_to_pay:   LogNormal(μ=11.0, σ=1.0) → median ≈ $60k, range $20k-$500k.
    - expected_value:       LogNormal(μ=12.0, σ=1.2) → median ≈ $162k, range $40k-$2M.

    Parameters
    ----------
    concept_ids : list[str]
        List of concept IDs to assign records to (sampled with replacement).
    num_records : int
        Total number of commercial signal rows to generate.
    num_customers : int
        Pool of unique customer IDs to draw from.

    Returns
    -------
    pd.DataFrame
        Columns matching the commercial_signals table schema.
    """
    customer_pool = [f"CUST{str(i + 1).zfill(4)}" for i in range(num_customers)]

    pilot_interest = np.random.beta(a=3, b=2, size=num_records)
    urgency_score = np.random.beta(a=2, b=3, size=num_records)
    budget_signal = np.random.beta(a=2, b=4, size=num_records)
    implementation_risk = np.random.beta(a=2, b=3, size=num_records)
    willingness_to_pay = np.random.lognormal(mean=11.0, sigma=1.0, size=num_records)
    expected_value = np.random.lognormal(mean=12.0, sigma=1.2, size=num_records)

    rows = []
    for i in range(num_records):
        concept_id = random.choice(concept_ids)
        customer_id = random.choice(customer_pool)

        rows.append(
            {
                "concept_id": concept_id,
                "customer_id": customer_id,
                "pilot_interest": round(float(pilot_interest[i]), 4),
                "urgency_score": round(float(urgency_score[i]), 4),
                "budget_signal": round(float(budget_signal[i]), 4),
                "willingness_to_pay": round(float(willingness_to_pay[i]), 2),
                "expected_value": round(float(expected_value[i]), 2),
                "implementation_risk": round(float(implementation_risk[i]), 4),
            }
        )

    commercial_df = pd.DataFrame(rows)
    logger.info(
        "Generated %d commercial signal records across %d concepts.",
        len(commercial_df),
        commercial_df["concept_id"].nunique(),
    )
    return commercial_df


if __name__ == "__main__":
    concept_ids = [f"C{str(i + 1).zfill(3)}" for i in range(NUM_CONCEPTS)]
    df = generate_commercial_signals(concept_ids)
    print(df.describe())
