"""
load_to_db.py — Pipeline entry point: generates all mock data and writes it to SQLite.

Run this module directly to (re)build the full database from scratch:
    python -m src.data_pipeline.load_to_db

Steps:
    1. Drop and recreate all tables (fresh run).
    2. Generate all five datasets.
    3. Write each dataset to its corresponding table via pandas + sqlite3.
"""

import logging
import sqlite3

import pandas as pd

from src.config import DB_PATH, LOG_FORMAT, LOG_LEVEL
from src.data_generation.generate_commercial_signals import generate_commercial_signals
from src.data_generation.generate_concepts import generate_concepts
from src.data_generation.generate_demo_signals import generate_demo_signals
from src.data_generation.generate_text_feedback import generate_text_feedback
from src.data_generation.generate_usage_signals import generate_usage_signals
from src.data_pipeline.db_schema import create_all_tables, drop_all_tables, get_connection

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _write_dataframe(
    dataframe: pd.DataFrame,
    table_name: str,
    conn: sqlite3.Connection,
) -> None:
    """
    Write a pandas DataFrame to a SQLite table using executemany for reliability.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data to insert.
    table_name : str
        Target SQLite table name.
    conn : sqlite3.Connection
        Open database connection.
    """
    if dataframe.empty:
        logger.warning("DataFrame for table '%s' is empty — skipping.", table_name)
        return

    cols = ", ".join(dataframe.columns.tolist())
    placeholders = ", ".join(["?"] * len(dataframe.columns))
    sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
    records = [tuple(row) for row in dataframe.itertuples(index=False, name=None)]
    with conn:
        conn.executemany(sql, records)
    logger.info("Inserted %d rows into '%s'.", len(records), table_name)


def run_pipeline(db_path: str = DB_PATH, reset: bool = True) -> None:
    """
    Full data generation and load pipeline.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    reset : bool
        If True, drops all existing tables before recreating them.
        Set to False for incremental loads (not recommended for fresh runs).
    """
    logger.info("=== Commercialization Engine — Data Pipeline Start ===")

    # --- Schema ---
    if reset:
        drop_all_tables(db_path)
    create_all_tables(db_path)

    # --- Generate data ---
    logger.info("Generating synthetic data...")
    concepts_df = generate_concepts()
    concept_ids = concepts_df["concept_id"].tolist()

    demo_df = generate_demo_signals(concept_ids)
    usage_df = generate_usage_signals(concept_ids)
    commercial_df = generate_commercial_signals(concept_ids)
    feedback_df = generate_text_feedback(concepts_df)

    # --- Load to DB ---
    conn = get_connection(db_path)
    try:
        _write_dataframe(concepts_df, "concepts", conn)
        _write_dataframe(demo_df, "demo_signals", conn)
        _write_dataframe(usage_df, "usage_signals", conn)
        _write_dataframe(commercial_df, "commercial_signals", conn)
        _write_dataframe(feedback_df, "text_feedback", conn)
    finally:
        conn.close()

    logger.info("=== Data Pipeline Complete — DB written to: %s ===", db_path)


if __name__ == "__main__":
    run_pipeline()
