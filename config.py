"""Paths, model names and thresholds shared across the project."""

from pathlib import Path

ROOT = Path(__file__).parent

RAW_DATA_DIR = ROOT / "data" / "raw"
CONTRACTS_CSV = ROOT / "data" / "raw" / "contracts_enriched.csv"
CONTRACTS_DB = ROOT / "data" / "contracts.db"

RISK_MODEL_PATH = ROOT / "models" / "risk_model.joblib"

POLICY_RAW_DIR = ROOT / "policy" / "raw"
POLICY_PDF = ROOT / "policy" / "raw" / "pdf-mon-auto-cg.pdf"
CLAUSES_JSON = ROOT / "policy" / "clauses.json"
POLICY_INDEX = ROOT / "policy" / "index.npz"

GEMINI_MODEL_LIGHT = "gemini-3.5-flash-lite"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

RANDOM_SEED = 42
HOLDOUT_FRACTION = 0.2
