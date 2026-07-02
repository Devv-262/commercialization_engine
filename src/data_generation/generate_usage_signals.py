"""
generate_usage_signals.py — Generate synthetic product usage signal records.

Non-uniform distributions are used to reflect realistic trial behaviour:
- trial_sessions: Poisson(λ=5) — most customers try a handful of times
- feature_clicks: NegBinomial — overdispersed click counts
- time_spent: LogNormal — heavy-tailed time-on-product distribution
- abandoned_features: Poisson(λ=1.5) — low abandonment is common
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
    NUM_USAGE_RECORDS,
    RANDOM_SEED,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

fake = Faker()
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def generate_usage_signals(
    concept_ids: list[str],
    num_records: int = NUM_USAGE_RECORDS,
    num_customers: int = NUM_CUSTOMERS,
) -> pd.DataFrame:
    """
    Generate synthetic usage signal records for the provided concept IDs.

    Distributions used:
    - trial_sessions: Poisson(λ=5) — number of distinct trial sessions.
    - feature_clicks: NegativeBinomial(n=5, p=0.4) — overdispersed integer,
      captures variance in how deeply customers explore features.
    - repeat_usage_days: Poisson(λ=8) — days over which the customer returned.
    - active_users: Poisson(λ=3) — team members actively using the trial.
    - time_spent: LogNormal(μ=3.5, σ=0.8) minutes — heavy tail for power users.
    - abandoned_features: Poisson(λ=1.5) — number of features started but dropped.

    Parameters
    ----------
    concept_ids : list[str]
        List of concept IDs to assign records to (sampled with replacement).
    num_records : int
        Total number of usage signal rows to generate.
    num_customers : int
        Pool of unique customer IDs to draw from.

    Returns
    -------
    pd.DataFrame
        Columns matching the usage_signals table schema.
    """
    customer_pool = [f"CUST{str(i + 1).zfill(4)}" for i in range(num_customers)]

    trial_sessions = np.random.poisson(lam=5, size=num_records)
    feature_clicks = np.random.negative_binomial(n=5, p=0.4, size=num_records)
    repeat_usage_days = np.random.poisson(lam=8, size=num_records)
    active_users = np.clip(np.random.poisson(lam=3, size=num_records), 1, None)
    time_spent = np.random.lognormal(mean=3.5, sigma=0.8, size=num_records)
    abandoned_features = np.random.poisson(lam=1.5, size=num_records)

    rows = []
    for i in range(num_records):
        concept_id = random.choice(concept_ids)
        customer_id = random.choice(customer_pool)

        rows.append(
            {
                "concept_id": concept_id,
                "customer_id": customer_id,
                "trial_sessions": int(trial_sessions[i]),
                "feature_clicks": int(feature_clicks[i]),
                "repeat_usage_days": int(repeat_usage_days[i]),
                "active_users": int(active_users[i]),
                "time_spent": round(float(time_spent[i]), 2),
                "abandoned_features": int(abandoned_features[i]),
            }
        )

    usage_df = pd.DataFrame(rows)
    logger.info(
        "Generated %d usage signal records across %d concepts.",
        len(usage_df),
        usage_df["concept_id"].nunique(),
    )
    return usage_df


if __name__ == "__main__":
    concept_ids = [f"C{str(i + 1).zfill(3)}" for i in range(NUM_CONCEPTS)]
    df = generate_usage_signals(concept_ids)
    print(df.describe())
