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
