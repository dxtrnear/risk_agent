"""Download the freMTPL2freq dataset from OpenML (cached by sklearn after first run)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.datasets import fetch_openml

import config

OPENML_DATA_ID = 41214  # freMTPL2freq


def download() -> Path:
    """Fetch freMTPL2freq and save it as raw CSV. Returns the saved path."""
    data = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True)
    df: pd.DataFrame = data.frame

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = config.RAW_DATA_DIR / "contracts_raw.csv"
    df.to_csv(raw_path, index=False)
    return raw_path


if __name__ == "__main__":
    path = download()
    print(f"Saved {path}")
