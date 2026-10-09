"""Verifies a Langfuse session has one complete, correctly-nested trace per
claim: the right trace count, exactly one root observation per trace (all
named "process_claim" -- extra roots would mean a span escaped its parent),
and every trace has coverage, risk, get_contract and risk_score spans.

Used by eval runs (Phase 6) and the production loop (Phase 8) alike -- kept as
its own module rather than bundled into any one eval script.
"""

import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

REQUIRED_SPANS = {"coverage", "risk", "get_contract", "risk_score"}


def fetch_session_observations(session_id: str, window_minutes: int = 30) -> list:
    """Cursor-paginated fetch, scoped strictly to one session_id. Uses the v2
    observations endpoint (the legacy v3 trace.list/trace.get return 410
    LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION on Langfuse orgs created on/after
    2026-09-16). The time window is a safety margin, not the real filter --
    session_id is -- so every returned observation's session_id is checked to
    actually match, catching filter-leakage instead of assuming it away (this
    is exactly what let two false "bugs" through in an earlier, unscoped query)."""
    langfuse_client = get_client()
    window_start = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    window_end = datetime.now(timezone.utc) + timedelta(minutes=5)
    observations, cursor = [], None
    while True:
        response = langfuse_client.api.observations.get_many(
            session_id=session_id,
            fields="core,basic",
            from_start_time=window_start,
            to_start_time=window_end,
            limit=500,
            cursor=cursor,
        )
        observations.extend(o for o in response.data if o.session_id == session_id)
        cursor = response.meta.cursor
        if not cursor:
            break
    return observations


def find_problems(observations: list, expected_count: int) -> list[str]:
    problems = []
    by_trace: dict[str, set[str]] = {}
    for obs in observations:
        by_trace.setdefault(obs.trace_id, set()).add(obs.name or "")

    if len(by_trace) != expected_count:
        problems.append(f"expected {expected_count} traces, found {len(by_trace)}")

    roots = [o for o in observations if o.parent_observation_id is None]
    if len(roots) != expected_count:
        problems.append(f"expected {expected_count} root observations, found {len(roots)}")
    non_process_claim_roots = [o for o in roots if o.name != "process_claim"]
    if non_process_claim_roots:
        names = [(o.id, o.name) for o in non_process_claim_roots]
        problems.append(f"{len(non_process_claim_roots)} roots not named process_claim: {names}")

    for trace_id, names in by_trace.items():
        missing = REQUIRED_SPANS - names
        if missing:
            problems.append(f"trace {trace_id} missing: {sorted(missing)}")

    return problems


def verify_session(
    run_id: str, expected_count: int, max_attempts: int = 8, retry_seconds: int = 3
) -> list[str]:
    """Retries while there ARE problems, not just while the trace count looks
    right -- a trace can appear (one span ingested) well before its other
    spans finish exporting, so counting distinct trace_ids alone can look
    "done" while a claim is still missing most of its spans (ingestion lag,
    not a real gap). Returns the final problem list; empty means OK."""
    problems = ["not checked yet"]
    for _ in range(max_attempts):
        observations = fetch_session_observations(run_id)
        problems = find_problems(observations, expected_count)
        if not problems:
            break
        time.sleep(retry_seconds)
    return problems


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify a Langfuse session's traces are complete.")
    parser.add_argument("run_id")
    parser.add_argument("expected_count", type=int)
    args = parser.parse_args()

    problems = verify_session(args.run_id, args.expected_count)
    if problems:
        print("INCOMPLETE:")
        for p in problems:
            print(f"  {p}")
    else:
        print(f"OK: {args.expected_count} traces, each with {sorted(REQUIRED_SPANS)}")
