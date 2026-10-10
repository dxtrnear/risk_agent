# Claims Triage Agent

A portfolio project for AI quality engineering roles in insurance: a deliberately
small multi-agent system that triages French motor insurance claims, plus the
quality and security pipeline that measures it. The pipeline is the real
deliverable — see `CLAUDE.md` for the full design and phase plan.

## Findings

Running log of real bugs found while building the agent pipeline, kept up to date
as the project progresses (not a design log — see `DECISIONS.md` for that).

- Gemini rejects a replayed multi-turn function call that drops the model's own
  `thought_signature` — it must be captured and replayed verbatim on every turn.
- `AgentFailure` raised inside nested `async with` blocks backed by anyio task
  groups arrives at the caller wrapped in a `BaseExceptionGroup`, not as itself —
  a plain `except AgentFailure` never matches unless it's raised after they exit.
- The agent loop discarded its own tool-turn conclusion and let the final
  schema call re-decide independently from scratch — two decisions, not one,
  with the second silently overriding the first whenever they disagreed. This
  was the actual cause of what first looked like model instability.
- Langfuse spans nest correctly only with an explicit parent (`trace_context`);
  ambient OpenTelemetry context does not reliably survive the async/anyio
  boundaries a few frames down, so a first "fix" that relied on it produced 68
  separate flat traces instead of one nested tree per claim.
- Keeping the model's prior conclusion in history can leave a request ending on
  a model turn, which Gemini's API rejects outright — the restate instruction
  must be a final user turn, not a system-prompt suffix.
- The risk agent's `risk_score` tool call was agent-optional, not a correctness
  bug (the risk bucket was already computed independently in code) but a real
  design smell — fixed by having the pipeline call it itself, unconditionally.
- A trace-completeness check that retries on trace *count* can stop before a
  correct claim's slower-arriving spans finish exporting, reporting a false
  failure on a run that actually succeeded — must retry on completeness.
- Golden baseline v1 (58 cases × 3 runs, 167/174 decision match): `golden_25`'s
  one wrong run exposed two compounding defects at once, not one. `search_clauses`
  didn't retrieve the exclusion clause that run, so the coverage agent judged an
  unlicensed 16-year-old's claim "covered" with no exclusion in hand to cite — and
  independently, the risk agent raised an undeclared `driver_mismatch` false
  positive on the same claim in all three runs, which only became visible as an
  escalation in the one run where the retrieval miss had already broken coverage.
  A retrieval gap and a flag false positive, not a single clean failure.
- Some baseline-v1 runs cited clause 6.5 or 4.1.2 instead of 7 and still landed on
  the right decision (reject) — but neither clause's actual text supports the
  exclusion being invoked. Right outcome, wrong grounds. For an insurer, a refusal
  that cites a clause which doesn't actually say what it's being used to say is a
  badly grounded refusal, independent of whether the final decision happens to
  be correct.

## Limits

- `amount_vs_damage` needs a repair-cost reference, not just the model's own
  sense of what a repair "should" cost. The original `golden_23`/`golden_41`
  (kept in `evals/golden/borderline.json`, not scored) described real,
  plausible-sounding damage mechanisms (a key-scratch repaint quote, "water at
  tire height" flood damage) at amounts that aren't obviously wrong without
  external pricing data — missed 6/6 across the v1 baseline. Rewriting them to
  an unarguable disproportion (trivial/invisible damage, no justification
  offered) fixed the golden labels but didn't give the flag an actual cost
  reference; it will keep missing any real claim this ambiguous.
