"""Shared contracts-table loader, used by training and by golden-case authoring."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config


def load_contracts() -> pd.DataFrame:
    """Load the full `contracts` table, categorical columns cast to match training."""
    conn = sqlite3.connect(config.CONTRACTS_DB)
    df = pd.read_sql("SELECT * FROM contracts", conn)
    conn.close()
    for col in config.CATEGORICAL:
        df[col] = df[col].astype("category")
    return df
