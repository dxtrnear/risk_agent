"""Turns the risk model's per-year claim-rate prediction into a low/medium/high
bucket via a Poisson tail probability (DECISIONS #30). The one place this business
logic lives -- both the agent pipeline (agents/orchestrator.py) and the golden-set
safety net (tests/test_golden_consistency.py) call this, never an LLM.
"""

import joblib
import pandas as pd
from scipy.stats import poisson

import config

_MODEL = joblib.load(config.RISK_MODEL_PATH)


def compute_risk_bucket(contract: dict) -> str:
    """`contract` is a get_contract-shaped dict: needs config.FEATURES, exposure,
    and claim_nb. "Actual claims" is the contract's recorded history only (#31) --
    never the claim currently being filed."""
    df = pd.DataFrame([contract])
    for col in config.CATEGORICAL:
        df[col] = df[col].astype("category")
    rate = float(_MODEL.predict(df[config.FEATURES])[0])
    expected_claims = rate * contract["exposure"]
    p = poisson.sf(contract["claim_nb"] - 1, expected_claims)
    if p >= config.RISK_P_LOW_MIN:
        return "low"
    if p >= config.RISK_P_MEDIUM_MIN:
        return "medium"
    return "high"
