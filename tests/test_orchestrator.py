"""Pipeline's pure decision-table and validation logic -- no LLM, no MCP, no
subprocess. agents/orchestrator.py's triage() itself needs a real MCP session and
is exercised end-to-end by scripts/run_one_case.py instead (see Phase 5 guide)."""

import pytest

from agents.orchestrator import (
    _check_clause_id,
    _decide,
    _effective_amount,
    _validate_flags,
)
from agents.schemas import CoverageResult, RedFlag

FRANCHISE = 500.0


def _coverage(**overrides) -> CoverageResult:
    defaults = dict(coverage="covered", clause_id="6.5", limitation=None, reasoning="r")
    return CoverageResult(**{**defaults, **overrides})


def test_validate_flags_exact_match_kept():
    flags = [RedFlag(flag="payment_pressure", evidence="payez vite")]
    validated, dropped = _validate_flags(flags, "Bonjour, payez vite merci.")
    assert validated == flags
    assert dropped == []


def test_validate_flags_whitespace_normalized_match_kept():
    flags = [RedFlag(flag="payment_pressure", evidence="payez   vite")]
    validated, dropped = _validate_flags(flags, "Bonjour, payez vite merci.")
    assert validated == flags
    assert dropped == []


def test_validate_flags_non_matching_evidence_dropped():
    flags = [RedFlag(flag="payment_pressure", evidence="texte invente")]
    validated, dropped = _validate_flags(flags, "Bonjour, payez vite merci.")
    assert validated == []
    assert dropped == flags


def test_check_clause_id_accepts_none():
    _check_clause_id(None)  # must not raise


def test_check_clause_id_accepts_real_clause():
    _check_clause_id("6.5")  # must not raise


def test_check_clause_id_rejects_hallucinated_clause():
    with pytest.raises(ValueError, match="hallucinated"):
        _check_clause_id("99.99")


def test_effective_amount_none_when_amount_missing():
    assert _effective_amount(None, None) is None


def test_effective_amount_full_when_no_limitation():
    assert _effective_amount(1000, None) == 1000


def test_effective_amount_scaled_by_limitation():
    assert _effective_amount(1000, "keys_in_vehicle") == 700


def test_decide_missing_info_wins_first():
    decision, clause = _decide(_coverage(coverage="missing_info"), 1000, 1000, FRANCHISE, "high", [])
    assert (decision, clause) == ("ask_info", None)


def test_decide_not_covered_rejects_citing_coverage_clause():
    decision, clause = _decide(
        _coverage(coverage="not_covered", clause_id="7"), 1000, 1000, FRANCHISE, "high", []
    )
    assert (decision, clause) == ("reject", "7")


def test_decide_amount_missing_asks_info_even_if_covered():
    decision, clause = _decide(_coverage(), None, None, FRANCHISE, "low", [])
    assert (decision, clause) == ("ask_info", None)


def test_decide_below_franchise_rejects_citing_franchise_clause():
    decision, clause = _decide(_coverage(), 400, 400, FRANCHISE, "low", [])
    assert decision == "reject"
    assert clause == "8.1"


def test_decide_rc_claim_ignores_franchise():
    decision, clause = _decide(_coverage(clause_id="4.1.2"), 100, 100, FRANCHISE, "low", [])
    assert decision == "approve"


def test_decide_high_risk_escalates():
    decision, _ = _decide(_coverage(), 1000, 1000, FRANCHISE, "high", [])
    assert decision == "escalate"


def test_decide_amount_over_threshold_escalates():
    decision, _ = _decide(_coverage(), 5000, 5000, FRANCHISE, "low", [])
    assert decision == "escalate"


def test_decide_validated_flag_escalates_even_at_low_risk_and_amount():
    flag = [RedFlag(flag="payment_pressure", evidence="e")]
    decision, _ = _decide(_coverage(), 1000, 1000, FRANCHISE, "low", flag)
    assert decision == "escalate"


def test_decide_medium_risk_does_not_escalate_alone():
    decision, _ = _decide(_coverage(), 1000, 1000, FRANCHISE, "medium", [])
    assert decision == "approve"


def test_decide_otherwise_approves():
    decision, clause = _decide(_coverage(), 1000, 1000, FRANCHISE, "low", [])
    assert (decision, clause) == ("approve", "6.5")
