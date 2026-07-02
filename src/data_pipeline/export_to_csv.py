"""
export_to_csv.py — Exports the current SQLite database tables to CSV files.

This allows you to view and edit the data in Excel. You can then re-import it
using import_from_csv.py.
"""

import os
import sqlite3
import pandas as pd
from src.config import DB_PATH

CSV_DIR = "data/csv"

def export_all_tables_to_csv():
    os.makedirs(CSV_DIR, exist_ok=True)
    
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Run load_to_db.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    tables = [
        "concepts",
        "demo_signals",
        "usage_signals",
        "commercial_signals",
        "text_feedback"
    ]
    
    for table in tables:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            out_path = os.path.join(CSV_DIR, f"{table}.csv")
            df.to_csv(out_path, index=False)
            print(f"Exported {len(df)} rows to {out_path}")
        except Exception as e:
            print(f"Error exporting {table}: {e}")
            
    conn.close()
    print("\nAll tables exported successfully to data/csv/")

if __name__ == "__main__":
    export_all_tables_to_csv()
