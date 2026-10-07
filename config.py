"""Paths, model names and thresholds shared across the project."""

from pathlib import Path

ROOT = Path(__file__).parent

RAW_DATA_DIR = ROOT / "data" / "raw"
CONTRACTS_CSV = ROOT / "data" / "raw" / "contracts_enriched.csv"
CONTRACTS_DB = ROOT / "data" / "contracts.db"

RISK_MODEL_PATH = ROOT / "models" / "risk_model.joblib"
CATEGORICAL = ["area", "veh_brand", "veh_gas", "region"]
FEATURES = CATEGORICAL + ["veh_power", "veh_age", "driv_age", "bonus_malus", "density"]

POLICY_RAW_DIR = ROOT / "policy" / "raw"
POLICY_PDF = ROOT / "policy" / "raw" / "pdf-mon-auto-cg.pdf"
CLAUSES_JSON = ROOT / "policy" / "clauses.json"
POLICY_INDEX = ROOT / "policy" / "index.npz"

GEMINI_MODEL_LIGHT = "gemini-3.5-flash-lite"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

RANDOM_SEED = 42
HOLDOUT_FRACTION = 0.2

# "Large amount" escalation cutoff. freMTPL2sev (OpenML data_id=41215, real claim
# severities): 4000 EUR sits at the 93.7th percentile, i.e. this escalates the top
# ~6.3% of claims by amount alone.
CLAIM_AMOUNT_THRESHOLD_EUR = 4000

# Risk bucket from a Poisson tail probability, not a raw ratio: p = P(seeing at
# least this many HISTORICAL claims | this contract's expected rate). "Actual
# claims" is the contract's recorded claim_nb -- history, never counting the claim
# currently being filed. A ratio-based bucket was tried first and rejected: it
# collapsed to "has ever claimed" (one claim vs. ~0.05 expected always reads as a
# huge ratio), leaving the model's actual per-contract rate irrelevant.
RISK_P_LOW_MIN = 0.05  # p >= this -> low (this many claims is unsurprising)
RISK_P_MEDIUM_MIN = 0.005  # p >= this (but < LOW_MIN) -> medium; below -> high
