"""
db_schema.py — SQLite table definitions (DDL).

Creates all six tables for the commercialization engine. Run this module
directly or call create_all_tables() from the pipeline entry point.
"""

import logging
import sqlite3
from pathlib import Path

from src.config import DB_PATH, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL statements (one per table, ordered by dependency)
# ---------------------------------------------------------------------------

DDL_CONCEPTS = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_id          TEXT PRIMARY KEY,
    concept_name        TEXT NOT NULL,
    industry            TEXT,
    problem_area        TEXT,
    target_user         TEXT,
    delivery_complexity REAL,   -- 0-1 normalised
    strategic_fit       REAL    -- 0-1 normalised
);
"""

DDL_DEMO_SIGNALS = """
CREATE TABLE IF NOT EXISTS demo_signals (
    demo_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id             TEXT REFERENCES concepts(concept_id),
    customer_id            TEXT,
    segment               TEXT,
    demo_date             DATE,
    feedback_score        REAL,    -- 0-10
    follow_up_requested   BOOLEAN,
    decision_maker_present BOOLEAN,
    objections_count      INTEGER
);
"""

DDL_USAGE_SIGNALS = """
CREATE TABLE IF NOT EXISTS usage_signals (
    usage_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id           TEXT REFERENCES concepts(concept_id),
    customer_id          TEXT,
    trial_sessions       INTEGER,
    feature_clicks       INTEGER,
    repeat_usage_days    INTEGER,
    active_users         INTEGER,
    time_spent           REAL,    -- minutes
    abandoned_features   INTEGER
);
"""

DDL_COMMERCIAL_SIGNALS = """
CREATE TABLE IF NOT EXISTS commercial_signals (
    commercial_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id           TEXT REFERENCES concepts(concept_id),
    customer_id          TEXT,
    pilot_interest       REAL,    -- 0-1
    urgency_score        REAL,    -- 0-1
    budget_signal        REAL,    -- 0-1
    willingness_to_pay   REAL,    -- currency
    expected_value       REAL,    -- currency
    implementation_risk  REAL     -- 0-1
);
"""

DDL_TEXT_FEEDBACK = """
CREATE TABLE IF NOT EXISTS text_feedback (
    feedback_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id             TEXT REFERENCES concepts(concept_id),
    customer_id            TEXT,
    customer_comments      TEXT,
    pain_point_statements  TEXT,
    objection_themes       TEXT,
    requested_capabilities TEXT
);
"""

DDL_CONCEPT_SCORES = """
CREATE TABLE IF NOT EXISTS concept_scores (
    concept_id           TEXT PRIMARY KEY REFERENCES concepts(concept_id),
    readiness_score      REAL,      -- 0-100
    confidence_score     REAL,      -- 0-100
    cluster_id           INTEGER,
    recommended_outcome  TEXT,      -- one of the 5 outcomes
    evidence_json        TEXT,      -- structured evidence (SHAP + rules)
    generated_at         TIMESTAMP
);
"""

ALL_DDL: list[str] = [
    DDL_CONCEPTS,
    DDL_DEMO_SIGNALS,
    DDL_USAGE_SIGNALS,
    DDL_COMMERCIAL_SIGNALS,
    DDL_TEXT_FEEDBACK,
    DDL_CONCEPT_SCORES,
]


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Return a SQLite connection with foreign key enforcement enabled.

    Parameters
    ----------
    db_path : str
        Relative or absolute path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def create_all_tables(db_path: str = DB_PATH) -> None:
    """
    Execute all DDL statements to create tables (idempotent — IF NOT EXISTS).

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """
    logger.info("Creating database schema at: %s", db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            for ddl in ALL_DDL:
                conn.execute(ddl)
        logger.info("Schema creation complete — %d tables created/verified.", len(ALL_DDL))
    finally:
        conn.close()


def drop_all_tables(db_path: str = DB_PATH) -> None:
    """
    Drop all tables (for resetting the database during development).

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """
    table_names = [
        "concept_scores",
        "text_feedback",
        "commercial_signals",
        "usage_signals",
        "demo_signals",
        "concepts",
    ]
    logger.warning("Dropping all tables in: %s", db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            for table in table_names:
                conn.execute(f"DROP TABLE IF EXISTS {table};")
        logger.info("All tables dropped.")
    finally:
        conn.close()


if __name__ == "__main__":
    create_all_tables()
