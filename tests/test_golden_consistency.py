"""Golden-set safety net: schema, clause-ID existence, and decision-table recompute.

Scoped to golden.json only -- production.json isn't opened before Phase 8, and a
failing assertion here could print claim content into the pytest output, which must
never happen for production cases.
"""

import json
import sqlite3
from typing import Literal

import joblib
import pandas as pd
import pytest
from pydantic import BaseModel
from scipy.stats import poisson

import config

GOLDEN_PATH = config.ROOT / "evals" / "golden" / "golden.json"
_ALL_COVERS = config.FORMULE_COVERAGES.values()
KNOWN_COVER_CLAUSES = {cid for covers in _ALL_COVERS for cid in covers}
RejectReason = Literal["formule_gap", "exclusion", "condition_not_met", "franchise"]


class ExpectedFlag(BaseModel):
    flag: Literal[tuple(config.RED_FLAG_TYPES)]
    evidence: str


class GoldenCase(BaseModel):
    case_id: str
    contract_id: int
    claim_text: str
    claim_amount_eur: float | None
    effective_amount_eur: float | None
    scenario: str
    expected_coverage: Literal["covered", "not_covered", "missing_info"]
    expected_risk_bucket: Literal["low", "medium", "high"]
    expected_decision: Literal["approve", "reject", "ask_info", "escalate"]
    rationale: str
    expected_flags: list[ExpectedFlag]
    agent_clause_ids: list[str]
    decision_clause_id: str | None
    expected_limitation: Literal["keys_in_vehicle"] | None
    reject_reason: RejectReason | None


def _load_cases() -> list[GoldenCase]:
    raw = json.loads(GOLDEN_PATH.read_text())
    return [GoldenCase(**c) for c in raw]


def _load_valid_clause_ids() -> set[str]:
    clauses = json.loads(config.CLAUSES_JSON.read_text())
    return {c["clause_id"] for c in clauses}


def _fetch_contract_row(contract_id: int) -> dict:
    conn = sqlite3.connect(f"file:{config.CONTRACTS_DB}?mode=ro", uri=True)
    query = "SELECT * FROM contracts WHERE contract_id = ?"
    cursor = conn.execute(query, (contract_id,))
    row = cursor.fetchone()
    cols = [d[0] for d in cursor.description]
    conn.close()
    assert row is not None, f"contract_id {contract_id} not found in contracts.db"
    return dict(zip(cols, row))


def _risk_bucket(contract: dict) -> str:
    df = pd.DataFrame([contract])
    for col in config.CATEGORICAL:
        df[col] = df[col].astype("category")
    rate = float(RISK_MODEL.predict(df[config.FEATURES])[0])
    expected_claims = rate * contract["exposure"]
    p = poisson.sf(contract["claim_nb"] - 1, expected_claims)
    if p >= config.RISK_P_LOW_MIN:
        return "low"
    if p >= config.RISK_P_MEDIUM_MIN:
        return "medium"
    return "high"


def _recompute_decision(case: GoldenCase, franchise: float) -> str:
    if case.expected_coverage == "missing_info":
        return "ask_info"
    if case.expected_coverage == "not_covered":
        return "reject"
    if case.claim_amount_eur is None:
        return "ask_info"

    is_rc = (case.decision_clause_id or "").startswith("4.") or any(
        c.startswith("4.") for c in case.agent_clause_ids
    )
    eff = case.effective_amount_eur
    if not is_rc and eff is not None and eff <= franchise:
        return "reject"

    has_flag = len(case.expected_flags) > 0
    amount_over = eff is not None and eff > config.CLAIM_AMOUNT_THRESHOLD_EUR
    if case.expected_risk_bucket == "high" or amount_over or has_flag:
        return "escalate"
    return "approve"


def _expected_effective_amount(case: GoldenCase) -> float | None:
    if case.claim_amount_eur is None:
        return None
    fraction = config.PARTIAL_COVERAGE_FRACTIONS.get(case.expected_limitation, 1.0)
    return case.claim_amount_eur * fraction


CASES = _load_cases()
VALID_CLAUSE_IDS = _load_valid_clause_ids()
RISK_MODEL = joblib.load(config.RISK_MODEL_PATH)


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_clause_ids_exist(case: GoldenCase):
    for clause_id in case.agent_clause_ids:
        msg = f"{case.case_id}: agent clause {clause_id!r} unknown"
        assert clause_id in VALID_CLAUSE_IDS, msg
    if case.decision_clause_id is not None:
        assert case.decision_clause_id in VALID_CLAUSE_IDS, (
            f"{case.case_id}: decision clause {case.decision_clause_id!r} unknown"
        )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_risk_bucket_matches_model(case: GoldenCase):
    contract = _fetch_contract_row(case.contract_id)
    actual = _risk_bucket(contract)
    expected = case.expected_risk_bucket
    msg = f"{case.case_id}: model gives {actual}, case expects {expected}"
    assert actual == expected, msg


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_effective_amount_is_consistent(case: GoldenCase):
    expected = _expected_effective_amount(case)
    assert case.effective_amount_eur == expected, (
        f"{case.case_id}: stored effective_amount_eur={case.effective_amount_eur}, "
        f"recomputed={expected} from claim_amount_eur={case.claim_amount_eur} "
        f"and limitation={case.expected_limitation!r}"
    )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_decision_matches_table(case: GoldenCase):
    contract = _fetch_contract_row(case.contract_id)
    recomputed = _recompute_decision(case, contract["franchise"])
    franchise = contract["franchise"]
    detail = (
        f"coverage={case.expected_coverage}, bucket={case.expected_risk_bucket}, "
        f"eff_amount={case.effective_amount_eur}, franchise={franchise}, "
        f"flags={len(case.expected_flags)}"
    )
    expected = case.expected_decision
    msg = f"{case.case_id}: table={recomputed!r}, expects={expected!r} ({detail})"
    assert recomputed == expected, msg


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_agent_clause_ids_match_formule(case: GoldenCase):
    contract = _fetch_contract_row(case.contract_id)
    subscribed = config.FORMULE_COVERAGES[contract["formule"]]
    cover_clauses = [c for c in case.agent_clause_ids if c in KNOWN_COVER_CLAUSES]

    not_covered = case.expected_coverage == "not_covered"
    is_formule_gap = not_covered and case.reject_reason == "formule_gap"

    if case.expected_coverage == "covered" and cover_clauses:
        msg = f"{case.case_id}: covered, {cover_clauses} not in formule's {subscribed}"
        assert any(c in subscribed for c in cover_clauses), msg
    elif is_formule_gap:
        wrongly_included = [c for c in cover_clauses if c in subscribed]
        msg = f"{case.case_id}: not_covered but {wrongly_included} is subscribed"
        assert not wrongly_included, msg


def test_no_duplicate_contract_ids():
    ids = [c.contract_id for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate contract_id across golden cases"


def test_no_duplicate_case_ids():
    ids = [c.case_id for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case_id in golden.json"


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_reject_reason_consistency(case: GoldenCase):
    if case.expected_decision == "reject":
        msg = f"{case.case_id}: reject with no reject_reason"
        assert case.reject_reason is not None, msg
    else:
        msg = f"{case.case_id}: not reject but has reject_reason"
        assert case.reject_reason is None, msg
