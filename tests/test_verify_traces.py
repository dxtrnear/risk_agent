"""find_problems against synthetic observations -- no real Langfuse query."""

from types import SimpleNamespace

from evals.verify_traces import find_problems


def _obs(trace_id, name, parent_observation_id="not-root"):
    return SimpleNamespace(
        trace_id=trace_id, name=name, parent_observation_id=parent_observation_id, id=f"{trace_id}-{name}"
    )


def _complete_trace(trace_id: str) -> list:
    return [
        _obs(trace_id, "process_claim", parent_observation_id=None),
        _obs(trace_id, "coverage"),
        _obs(trace_id, "risk"),
        _obs(trace_id, "get_contract"),
        _obs(trace_id, "risk_score"),
    ]


def test_complete_session_has_no_problems():
    observations = _complete_trace("t1") + _complete_trace("t2")
    assert find_problems(observations, expected_count=2) == []


def test_wrong_trace_count_is_a_problem():
    observations = _complete_trace("t1")
    problems = find_problems(observations, expected_count=2)
    assert any("expected 2 traces, found 1" in p for p in problems)


def test_missing_span_is_reported_by_trace_id():
    observations = [o for o in _complete_trace("t1") if o.name != "risk"]
    problems = find_problems(observations, expected_count=1)
    assert any("t1" in p and "risk" in p for p in problems)


def test_extra_non_root_named_process_claim_root_is_a_problem():
    observations = _complete_trace("t1") + [_obs("t2", "stray_span", parent_observation_id=None)]
    problems = find_problems(observations, expected_count=1)
    assert any("not named process_claim" in p for p in problems)


def test_missing_root_is_a_problem():
    observations = [o for o in _complete_trace("t1") if o.name != "process_claim"]
    problems = find_problems(observations, expected_count=1)
    assert any("root observation" in p for p in problems)
