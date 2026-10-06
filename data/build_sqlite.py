"""Load the enriched contracts CSV into a SQLite database."""

import sqlite3
from pathlib import Path

import pandas as pd

import config


def build_db(csv_path: Path, db_path: Path) -> None:
    """(Re)build the `contracts` table from the enriched CSV. Idempotent."""
    df = pd.read_csv(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    df.to_sql("contracts", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX idx_contract_id ON contracts(contract_id)")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    build_db(config.CONTRACTS_CSV, config.CONTRACTS_DB)
    print(f"Built {config.CONTRACTS_DB}")
