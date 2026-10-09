"""Boundary validation for agent output schemas -- no LLM calls."""

import pytest
from pydantic import ValidationError

from agents.schemas import CoverageResult, RedFlag, RiskResult


def test_coverage_result_accepts_missing_info_with_no_clause():
    result = CoverageResult(coverage="missing_info", clause_id=None, limitation=None, reasoning="r")
    assert result.clause_id is None


def test_coverage_result_accepts_valid_limitation():
    result = CoverageResult(
        coverage="covered", clause_id="6.2", limitation="keys_in_vehicle", reasoning="r"
    )
    assert result.limitation == "keys_in_vehicle"


def test_coverage_result_rejects_unknown_limitation():
    with pytest.raises(ValidationError):
        CoverageResult(coverage="covered", clause_id="6.2", limitation="not_a_real_limitation", reasoning="r")


def test_coverage_result_rejects_unknown_coverage_value():
    with pytest.raises(ValidationError):
        CoverageResult(coverage="probably_covered", clause_id="6.2", limitation=None, reasoning="r")


def test_red_flag_rejects_unknown_flag_type():
    with pytest.raises(ValidationError):
        RedFlag(flag="not_a_real_flag", evidence="some text")


def test_risk_result_accepts_empty_flags():
    result = RiskResult(flags=[], reasoning="nothing suspicious")
    assert result.flags == []


def test_risk_result_accepts_valid_flag():
    result = RiskResult(
        flags=[{"flag": "payment_pressure", "evidence": "pay now"}], reasoning="r"
    )
    assert result.flags[0].flag == "payment_pressure"
