"""
generate_text_feedback.py — Generate synthetic customer text feedback records.

Uses template-based text generation (no LLM required).
Templates are parameterised by concept name, problem area, and target user
to produce varied, realistic-sounding qualitative feedback.
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
    NUM_FEEDBACK_RECORDS,
    RANDOM_SEED,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

fake = Faker()
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Template banks — varied enough to avoid identical text rows
# ---------------------------------------------------------------------------

COMMENT_TEMPLATES: list[str] = [
    "Really impressed by how {concept_name} handles {problem_area}. Our {target_user} loved the demo.",
    "The {concept_name} prototype shows promise, especially for {problem_area}, though it needs more polish.",
    "We've been looking for a solution like {concept_name} for {problem_area} — this hits the mark.",
    "Interesting approach to {problem_area}. The {target_user} had questions about scalability.",
    "Strong demo from {concept_name}. We could see this being useful for our {target_user}.",
    "The {problem_area} use case is compelling. {concept_name} exceeded our initial expectations.",
    "We'd like to see more customisation options before committing to {concept_name}.",
    "Initial reaction from our {target_user} team was positive — the {problem_area} workflow is intuitive.",
    "Not sure it fully solves {problem_area} yet, but {concept_name} is on the right track.",
    "The ROI potential for {concept_name} in {problem_area} looks meaningful. Let's discuss next steps.",
]

PAIN_POINT_TEMPLATES: list[str] = [
    "Current {problem_area} processes are manual, slow, and error-prone.",
    "We lose significant time each week to manual {problem_area} workflows — automation is overdue.",
    "Our {target_user} spends too much time on {problem_area} instead of strategic work.",
    "Data quality issues in {problem_area} lead to costly mistakes for our team.",
    "Lack of real-time visibility into {problem_area} is a persistent operational challenge.",
    "Existing tools for {problem_area} don't integrate with our stack — fragmentation is a pain point.",
    "Scaling {problem_area} operations with headcount is not sustainable.",
    "Regulatory pressure around {problem_area} is increasing — we need a faster compliance path.",
]

OBJECTION_TEMPLATES: list[str] = [
    "Concerned about integration complexity with existing systems.",
    "Data privacy and security need to be validated before we can proceed.",
    "Budget approval cycle is long — need stronger business case.",
    "Worried about model accuracy in edge cases relevant to our use of {problem_area}.",
    "Unclear how quickly the {target_user} team could adopt this tool.",
    "Vendor lock-in is a concern — would prefer open APIs.",
    "Need references from similar organisations before committing.",
    "Our IT team flagged concerns about on-premise vs. cloud deployment.",
    "The pricing model needs to be restructured for our volume.",
    "Would need a longer pilot to validate the {problem_area} outcomes.",
]

CAPABILITY_TEMPLATES: list[str] = [
    "Real-time alerting when {problem_area} thresholds are breached.",
    "Integration with our existing CRM / ERP stack.",
    "Explainable outputs so our {target_user} can understand and trust the recommendations.",
    "Role-based access control for different team members.",
    "Automated reporting to support {problem_area} governance requirements.",
    "Ability to customise the model for our specific {problem_area} context.",
    "Historical trend analysis, not just current-state predictions.",
    "Mobile-friendly interface for field {target_user} access.",
    "Bulk data import and export capabilities.",
    "SLA-backed uptime guarantees for production deployment.",
]


def _fill_template(template: str, concept_name: str, problem_area: str, target_user: str) -> str:
    """Fill a template string with concept-specific values."""
    return template.format(
        concept_name=concept_name,
        problem_area=problem_area,
        target_user=target_user,
    )


def generate_text_feedback(
    concepts_df: pd.DataFrame,
    num_records: int = NUM_FEEDBACK_RECORDS,
    num_customers: int = NUM_CUSTOMERS,
) -> pd.DataFrame:
    """
    Generate synthetic text feedback records tied to concepts.

    Each row contains four free-text fields drawn from template banks,
    parameterised by concept name, problem area, and target user so the
    text is contextually coherent without requiring an LLM.

    Parameters
    ----------
    concepts_df : pd.DataFrame
        DataFrame from generate_concepts() — must contain concept_id,
        concept_name, problem_area, target_user columns.
    num_records : int
        Total number of text feedback rows to generate.
    num_customers : int
        Pool of unique customer IDs to draw from.

    Returns
    -------
    pd.DataFrame
        Columns matching the text_feedback table schema.
    """
    customer_pool = [f"CUST{str(i + 1).zfill(4)}" for i in range(num_customers)]
    concept_lookup = concepts_df.set_index("concept_id")[
        ["concept_name", "problem_area", "target_user"]
    ].to_dict("index")

    concept_ids = concepts_df["concept_id"].tolist()

    rows = []
    for _ in range(num_records):
        concept_id = random.choice(concept_ids)
        customer_id = random.choice(customer_pool)
        meta = concept_lookup[concept_id]

        comment = _fill_template(
            random.choice(COMMENT_TEMPLATES),
            meta["concept_name"],
            meta["problem_area"],
            meta["target_user"],
        )
        pain_point = _fill_template(
            random.choice(PAIN_POINT_TEMPLATES),
            meta["concept_name"],
            meta["problem_area"],
            meta["target_user"],
        )
        objection = _fill_template(
            random.choice(OBJECTION_TEMPLATES),
            meta["concept_name"],
            meta["problem_area"],
            meta["target_user"],
        )
        capability = _fill_template(
            random.choice(CAPABILITY_TEMPLATES),
            meta["concept_name"],
            meta["problem_area"],
            meta["target_user"],
        )

        rows.append(
            {
                "concept_id": concept_id,
                "customer_id": customer_id,
                "customer_comments": comment,
                "pain_point_statements": pain_point,
                "objection_themes": objection,
                "requested_capabilities": capability,
            }
        )

    feedback_df = pd.DataFrame(rows)
    logger.info(
        "Generated %d text feedback records across %d concepts.",
        len(feedback_df),
        feedback_df["concept_id"].nunique(),
    )
    return feedback_df


if __name__ == "__main__":
    from src.data_generation.generate_concepts import generate_concepts

    concepts_df = generate_concepts()
    df = generate_text_feedback(concepts_df)
    print(df.head(5).to_string())
