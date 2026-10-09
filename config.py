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

# Closed set of red flags the risk agent may raise. Each flag must carry an
# `evidence` quote found verbatim (whitespace-normalized) in the claim text, checked
# in code -- a flag that fails this check is dropped, not trusted. driver_mismatch is
# narrow: the claim presents the insured as the driver but details contradict the
# contract profile. An openly declared other driver is not a flag -- clause 8.1.2.5
# handles that with an extra franchise, not a risk signal.
RED_FLAG_TYPES = [
    "vehicle_mismatch",
    "driver_mismatch",
    "inconsistent_story",
    "amount_vs_damage",
    "payment_pressure",
    "manipulation_attempt",  # text addresses the system directly or impersonates staff
]

# expected_limitation name -> fraction of the claimed amount actually payable.
# Limitations are coverage conditions, not clause IDs (a clause is section-level; a
# limitation is a specific condition within one) -- the coverage agent returns the
# condition name, code looks up the fraction. Absent (null) means 1.0.
# "keys_in_vehicle": clause 6.2 (Vol), ignition keys left in/on the vehicle -> 70%.
PARTIAL_COVERAGE_FRACTIONS = {
    "keys_in_vehicle": 0.7,
}

# Per-agent tool allow-lists (not stored on individual cases -- a case's expected
# tools are a function of which agent runs, not something that varies per claim).
COVERAGE_AGENT_TOOLS = ["get_contract", "search_clauses"]
RISK_AGENT_TOOLS = ["get_contract", "risk_score"]

AGENT_STEP_LIMIT = 6  # max LLM turns per agent loop before it fails loudly
AGENT_TIMEOUT_SECONDS = 30  # wall-clock budget for one agent's full loop

# The clause cited for a franchise-reject is the franchise clause itself, not the
# coverage clause (e.g. golden_05: agent_clause_ids=["6.5"], decision_clause_id="8.1").
FRANCHISE_CLAUSE_ID = "8.1"

# formule -> list of garantie clause_ids included in that tier. Clause 12's table
# lost its cell structure during PDF extraction (header row and data row text got
# flattened together), so this was read directly from the PDF (p.71) by hand, not
# inferred. Maps to AXA's real tiers: tiers->"Tiers Mini", tiers_etendu->"Tiers
# Étendue", tous_risques->"Tous risques Équilibre". "En option" columns don't count
# as included. 6.6 (Accessoires) is "en option" in every real AXA formule -- our
# tous_risques contracts are ASSUMED to have subscribed it (an assumption about our
# synthetic data, not something the table itself says). 5.2.4 (Contenu du véhicule)
# is an option nobody has subscribed in our data -- it's in no formule's list.
FORMULE_COVERAGES: dict[str, list[str]] = {
    "tiers": ["4.1.2", "4.2", "5.1"],
    "tiers_etendu": ["4.1.2", "4.2", "5.1", "6.1", "6.2", "6.3", "6.4", "6.8", "6.9", "6.10"],
    "tous_risques": [
        "4.1.2",
        "4.2",
        "5.1",
        "6.1",
        "6.2",
        "6.3",
        "6.4",
        "6.5",
        "6.6",
        "6.7",
        "6.8",
        "6.9",
        "6.10",
    ],
}
