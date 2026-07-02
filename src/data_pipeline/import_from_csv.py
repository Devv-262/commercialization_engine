"""
import_from_csv.py — Loads CSV files back into the SQLite database.

If you edit the CSV files in data/csv/ manually, run this script to 
re-populate the database with your custom data.
"""

import os
import sqlite3
import pandas as pd
from src.config import DB_PATH
from src.data_pipeline.db_schema import create_all_tables

CSV_DIR = "data/csv"

def import_csvs_to_db():
    if not os.path.exists(CSV_DIR):
        print(f"CSV directory {CSV_DIR} not found. Export first.")
        return

    # Wipe the database and recreate the tables
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    create_all_tables(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    
    tables = [
        "concepts",
        "demo_signals",
        "usage_signals",
        "commercial_signals",
        "text_feedback"
    ]
    
    for table in tables:
        csv_path = os.path.join(CSV_DIR, f"{table}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Write to sqlite, replacing if it exists
                df.to_sql(table, conn, if_exists="append", index=False)
                print(f"Imported {len(df)} rows into {table}")
            except Exception as e:
                print(f"Error importing {table}: {e}")
        else:
            print(f"Warning: {csv_path} not found. Table {table} will be empty.")
            
    conn.close()
    print("\nAll CSVs imported successfully! You can now run the Streamlit app.")

if __name__ == "__main__":
    import_csvs_to_db()
