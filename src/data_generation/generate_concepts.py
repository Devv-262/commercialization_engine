"""
generate_concepts.py — Generate synthetic AI product concept records.

Produces a DataFrame of NUM_CONCEPTS rows with realistic industry-spread
and non-uniform complexity/fit distributions. Seeded for reproducibility.
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
    RANDOM_SEED,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

fake = Faker()
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Concept catalogue — pre-defined realistic AI/ML product ideas
# ---------------------------------------------------------------------------

CONCEPT_CATALOGUE: list[dict] = [
    {
        "concept_name": "Predictive Churn Scorer",
        "industry": "Telecom",
        "problem_area": "Customer Retention",
        "target_user": "CX Ops Manager",
    },
    {
        "concept_name": "Contract Risk Analyzer",
        "industry": "Legal",
        "problem_area": "Risk Mitigation",
        "target_user": "General Counsel",
    },
    {
        "concept_name": "Dynamic Pricing Engine",
        "industry": "Retail",
        "problem_area": "Revenue Optimization",
        "target_user": "Pricing Director",
    },
    {
        "concept_name": "Supply Chain Anomaly Detector",
        "industry": "Manufacturing",
        "problem_area": "Operational Resilience",
        "target_user": "Supply Chain Lead",
    },
    {
        "concept_name": "Clinical Trial Matching AI",
        "industry": "Healthcare",
        "problem_area": "Patient Recruitment",
        "target_user": "Clinical Operations Head",
    },
    {
        "concept_name": "Fraud Pattern Classifier",
        "industry": "Financial Services",
        "problem_area": "Fraud Prevention",
        "target_user": "Risk Analytics Manager",
    },
    {
        "concept_name": "ESG Reporting Copilot",
        "industry": "Energy",
        "problem_area": "Sustainability Compliance",
        "target_user": "Sustainability Officer",
    },
    {
        "concept_name": "Talent Intelligence Platform",
        "industry": "HR Tech",
        "problem_area": "Workforce Planning",
        "target_user": "CHRO",
    },
    {
        "concept_name": "Intelligent Document Router",
        "industry": "Insurance",
        "problem_area": "Claims Processing",
        "target_user": "Operations Director",
    },
    {
        "concept_name": "Demand Forecasting Suite",
        "industry": "FMCG",
        "problem_area": "Inventory Optimization",
        "target_user": "Demand Planner",
    },
    {
        "concept_name": "Code Review Accelerator",
        "industry": "Software",
        "problem_area": "Engineering Productivity",
        "target_user": "Engineering Manager",
    },
    {
        "concept_name": "Customer Sentiment Radar",
        "industry": "E-Commerce",
        "problem_area": "Brand Experience",
        "target_user": "CMO",
    },
    {
        "concept_name": "Regulatory Change Monitor",
        "industry": "Financial Services",
        "problem_area": "Compliance Automation",
        "target_user": "Chief Compliance Officer",
    },
    {
        "concept_name": "Field Service Optimizer",
        "industry": "Utilities",
        "problem_area": "Asset Management",
        "target_user": "Field Operations VP",
    },
    {
        "concept_name": "Medical Imaging Triage AI",
        "industry": "Healthcare",
        "problem_area": "Diagnostic Efficiency",
        "target_user": "Radiology Department Head",
    },
    {
        "concept_name": "Loan Default Predictor",
        "industry": "Banking",
        "problem_area": "Credit Risk",
        "target_user": "Credit Risk Officer",
    },
    {
        "concept_name": "Ad Spend Optimizer",
        "industry": "Marketing",
        "problem_area": "Campaign ROI",
        "target_user": "Performance Marketing Lead",
    },
    {
        "concept_name": "Knowledge Graph Builder",
        "industry": "Consulting",
        "problem_area": "Knowledge Management",
        "target_user": "Chief Knowledge Officer",
    },
    {
        "concept_name": "Predictive Maintenance Advisor",
        "industry": "Aerospace",
        "problem_area": "Equipment Reliability",
        "target_user": "MRO Director",
    },
    {
        "concept_name": "Real-Time Translation Layer",
        "industry": "Global Enterprises",
        "problem_area": "Cross-Language Collaboration",
        "target_user": "CTO",
    },
]


def generate_concepts(num_concepts: int = NUM_CONCEPTS) -> pd.DataFrame:
    """
    Generate a DataFrame of synthetic AI product concept records.

    Each concept gets:
    - A unique concept_id (e.g. "C001")
    - Fields drawn from the pre-defined catalogue
    - delivery_complexity: Beta(2, 3) — skewed toward moderate-low complexity
    - strategic_fit: Beta(3, 2) — skewed toward higher fit

    Parameters
    ----------
    num_concepts : int
        Number of concepts to generate (capped by catalogue size).

    Returns
    -------
    pd.DataFrame
        Columns: concept_id, concept_name, industry, problem_area,
                 target_user, delivery_complexity, strategic_fit
    """
    n = min(num_concepts, len(CONCEPT_CATALOGUE))
    records = CONCEPT_CATALOGUE[:n]

    concept_ids = [f"C{str(i + 1).zfill(3)}" for i in range(n)]

    # Non-uniform distributions — not uniform to reflect real world variance
    delivery_complexity = np.random.beta(a=2, b=3, size=n).round(4)
    strategic_fit = np.random.beta(a=3, b=2, size=n).round(4)

    rows = []
    for idx, record in enumerate(records):
        rows.append(
            {
                "concept_id": concept_ids[idx],
                "concept_name": record["concept_name"],
                "industry": record["industry"],
                "problem_area": record["problem_area"],
                "target_user": record["target_user"],
                "delivery_complexity": float(delivery_complexity[idx]),
                "strategic_fit": float(strategic_fit[idx]),
            }
        )

    concepts_df = pd.DataFrame(rows)
    logger.info("Generated %d concept records.", len(concepts_df))
    return concepts_df


if __name__ == "__main__":
    df = generate_concepts()
    print(df.to_string())
