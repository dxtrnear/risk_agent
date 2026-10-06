"""Paths, model names and thresholds shared across the project."""

from pathlib import Path

ROOT = Path(__file__).parent

RAW_DATA_DIR = ROOT / "data" / "raw"
CONTRACTS_CSV = ROOT / "data" / "raw" / "contracts_enriched.csv"
CONTRACTS_DB = ROOT / "data" / "contracts.db"

RISK_MODEL_PATH = ROOT / "models" / "risk_model.joblib"

RANDOM_SEED = 42
HOLDOUT_FRACTION = 0.2
