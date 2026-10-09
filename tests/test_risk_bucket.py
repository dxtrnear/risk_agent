"""compute_risk_bucket against real contracts with a known, already-verified bucket
(cross-checked in tests/test_golden_consistency.py's own per-case assertions)."""

import sqlite3

import config
from models.risk_bucket import compute_risk_bucket


def _fetch_contract_row(contract_id: int) -> dict:
    conn = sqlite3.connect(f"file:{config.CONTRACTS_DB}?mode=ro", uri=True)
    cursor = conn.execute("SELECT * FROM contracts WHERE contract_id = ?", (contract_id,))
    row = cursor.fetchone()
    cols = [d[0] for d in cursor.description]
    conn.close()
    return dict(zip(cols, row))


def test_known_low_risk_contract():
    contract = _fetch_contract_row(2051860)  # golden_01, expected_risk_bucket=low
    assert compute_risk_bucket(contract) == "low"


def test_known_high_risk_contract():
    contract = _fetch_contract_row(1187103)  # golden_10, expected_risk_bucket=high
    assert compute_risk_bucket(contract) == "high"
