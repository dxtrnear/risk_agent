"""Pydantic output shapes validated at the agent-loop boundary (never written by
hand anywhere else -- an agent's final answer is only ever trusted once it has
passed one of these).
"""

from typing import Literal

from pydantic import BaseModel

import config


class RedFlag(BaseModel):
    flag: Literal[tuple(config.RED_FLAG_TYPES)]
    evidence: str


class CoverageResult(BaseModel):
    coverage: Literal["covered", "not_covered", "missing_info"]
    clause_id: str | None
    limitation: Literal[tuple(config.PARTIAL_COVERAGE_FRACTIONS.keys())] | None
    reasoning: str


class RiskResult(BaseModel):
    flags: list[RedFlag]
    reasoning: str
