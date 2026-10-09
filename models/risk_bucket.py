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


def bucket_from_rate(rate: float, exposure: float, claim_nb: int) -> str:
    """Same Poisson-tail bucket as compute_risk_bucket, but from an
    already-computed per-year rate (e.g. the risk_score MCP tool's own
    result) -- doesn't load or call the model itself. Takes plain numbers,
    never an agent run or tool-call object: by construction, nothing here can
    depend on whether or how an LLM chose to call a tool."""
    expected_claims = rate * exposure
    p = poisson.sf(claim_nb - 1, expected_claims)
    if p >= config.RISK_P_LOW_MIN:
        return "low"
    if p >= config.RISK_P_MEDIUM_MIN:
        return "medium"
    return "high"


def compute_risk_bucket(contract: dict) -> str:
    """`contract` is a get_contract-shaped dict: needs config.FEATURES, exposure,
    and claim_nb. "Actual claims" is the contract's recorded history only (#31) --
    never the claim currently being filed. Used by the golden-set safety net,
    which has no MCP session to call the risk_score tool through; the live
    pipeline instead calls that tool directly and uses bucket_from_rate."""
    df = pd.DataFrame([contract])
    for col in config.CATEGORICAL:
        df[col] = df[col].astype("category")
    rate = float(_MODEL.predict(df[config.FEATURES])[0])
    return bucket_from_rate(rate, contract["exposure"], contract["claim_nb"])
