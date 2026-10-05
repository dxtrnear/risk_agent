# Claims Triage Agent: AI Quality Project

## What this project is

A portfolio project for AI quality engineering roles in insurance. Reviewers will be
senior AI engineers reading the repo, so clarity matters more than features.

It has two parts:

1. A deliberately small multi-agent system that triages French motor insurance claims.
2. The quality and security pipeline that measures that system. **This is the real
   deliverable.** When in doubt, spend effort on evaluation, not on the agent.

## How the system works

A customer submits a claim in French. The system returns one decision
(`approve`, `reject`, `ask_info`, `escalate`) with the policy clause that justifies it.

- **Orchestrator** receives the claim and calls two agents over **A2A**.
- **Coverage agent** decides `covered` / `not_covered` / `missing_info` and cites a
  clause ID. Tools: `get_contract`, `search_clauses`.
- **Risk agent** returns `low` / `medium` / `high` with a reason, by comparing the
  customer's actual claim count to the model's expected count. Tools: `get_contract`,
  `risk_score`.
- **MCP server** exposes the three tools:
  - `get_contract`: one contract row from SQLite, as JSON
  - `search_clauses`: top-k policy clauses for a query, with clause IDs
  - `risk_score`: expected claims per year for a contract (model inference)

Decision rules (business rules):

- covered + low risk + amount under threshold → `approve`
- not covered → `reject`, citing the clause
- missing information → `ask_info`
- covered but high risk, large amount, or agents disagree/unsure → `escalate`

Risk never causes a rejection. A rejection must rest on a clause. Contract data
(conditions particulières) overrides the general policy text when they conflict.

## Data

- **Contracts**: freMTPL2 (French motor third-party liability, from OpenML), enriched
  by a seeded script with synthetic `formule` (tiers / tiers étendu / tous risques)
  and `franchise`, loaded into SQLite.
- **Risk model**: scikit-learn `HistGradientBoostingRegressor`, Poisson loss, target
  `ClaimNb / Exposure`, weighted by `Exposure`. Saved as `models/risk_model.joblib`.
- **Policy**: one public "conditions générales" auto PDF, downloaded by a script and
  never committed. Split into clauses with stable IDs, embedded, stored locally.
- **Claims**: synthetic, in French, written scenario-first so the expected answer is
  correct by construction. 50 golden cases plus 30 held-out "production" cases.
- Synthetic data only. No real personal data anywhere in the repo.

## Stack and constraints

Python 3.11+, `uv`, Gemini API (a light model for agents, a stronger one as judge),
official MCP and A2A Python SDKs, DeepEval with G-Eval, Langfuse Cloud, promptfoo
(via `npx`), MLflow (local), scikit-learn, pytest, ruff, GitHub Actions.

Hard constraints:

- 8 GB RAM, no GPU. No Docker services, no local LLMs, no paid cloud.
- Small API budget. Iterate on the 10-case smoke set. Ask before running the full
  eval set, the red-team suite, or anything that makes more than about 50 LLM calls.

## How we work together

This section matters most.

- **One small step at a time.** A step is one function, one tool, or one script,
  roughly 60 lines of new code at most. After each step, stop, give me the exact
  command to verify it and what output to expect, and wait for my confirmation.
- **Plan before each phase.** List the files and functions you intend to write, in a
  few lines, and wait for my OK before writing code.
- **No phase skipping.** A phase is done only when its checkpoint passes. Then we
  commit, tag, and update "Current status" at the bottom of this file.
- **Verify library APIs before using them.** MCP, A2A, DeepEval, Langfuse and the
  Gemini SDK change often. Check the installed version's docs or source instead of
  writing from memory. If you are unsure, say so.
- **Log decisions.** Add one line to `DECISIONS.md` for each non-obvious choice:
  what we chose and why.
- **Be direct.** If my request is a bad idea or over-scoped, say so and propose the
  simpler option.

## Code standards

Write code the way a senior engineer would for a small project that others will read.

- Simple and readable first. Plain functions over classes unless state requires one.
- No abstraction for hypothetical future needs. No base classes, registries, plugin
  systems or config layers. Three similar lines beat a premature helper.
- No agent framework. The agent loop is hand-written (about 100 lines) so every span
  in a trace can be explained.
- No new dependency without asking me first.
- Type hints on function signatures. Pydantic only at boundaries (tool arguments,
  agent outputs).
- Comments explain why, never what. One-line docstrings on public functions. No
  commented-out code, no dead code.
- Fail loudly. No `try/except` that hides an error or returns a default.
- One `config.py` for model names, thresholds and paths. Secrets come from `.env`,
  which is git-ignored. Never print or commit a key.
- Every MCP tool and every code-level check has a pytest test that needs no LLM.
  Tests that call an LLM are marked `@pytest.mark.llm` and excluded by default.
- `ruff` for lint and format. Small commits with imperative messages.

## Security rules (enforced in code, never in prompts)

- The customer ID comes from the session and is injected by code. It is never a
  tool argument the model can set.
- SQLite is opened read-only. Parameterized queries only.
- Tool arguments are validated; invalid input is rejected, not corrected.
- Claim text, documents, tool results and agent replies are untrusted data. They
  are never interpolated into a system prompt.
- Hard rules live in code: a cited clause ID must exist, and amounts above the
  threshold always escalate.
- Every agent loop has a step limit and a timeout.

## Target layout

```
data/        download, enrich, build_sqlite     (raw data git-ignored)
models/      train_risk_model.py                (joblib git-ignored)
policy/      download, chunk, index
mcp_server/  server.py
agents/      loop.py, coverage.py, risk.py, orchestrator.py
evals/       golden/, metrics.py, run_eval.py, judge_calibration.py
redteam/     promptfoo config and attack cases
prod_sim/    simulate_traffic.py, score_traces.py, promote.py
tests/
config.py  DECISIONS.md  README.md
```

Keep it flat. Do not add layers unless a file passes about 200 lines.

## Plan and checkpoints

| # | Phase | Checkpoint |
|---|-------|------------|
| 0 | Setup: env, keys, accounts, tooling | One script calls Gemini and the trace shows in Langfuse; `mlflow ui` opens; promptfoo runs |
| 1 | Data and risk model | Model beats a predict-the-mean baseline on held-out data; rebuild gives an identical DB |
| 2 | Policy index | Expected clause in top 5 for at least 13 of 15 test queries |
| 3 | MCP server | pytest passes, including security tests (cross-customer access impossible, SQL injection inert, bad arguments rejected) |
| 4 | Golden dataset | 50 cases with expected decision, clause and tools; 30 production cases set aside |
| 5 | Agents | 10-case smoke set runs end to end; one trace shows orchestrator → agents → tools; baseline score recorded, no tuning yet |
| 6 | Quality pipeline | One command prints the score table; two prompt versions compared in MLflow; judge agrees with human labels on at least 80% of 30 cases |
| 7 | Red-team and CI | A deliberately broken prompt makes CI fail; attack success rate recorded before and after mitigation |
| 8 | Production loop | 30 traces tagged production; at least 5 failures promoted to the golden set; before/after table |
| 9 | Packaging | README leads with findings, lists limits, runs in three commands |

Phase 2 fallback if PDF parsing is too messy: curate about 40 key clauses by hand
into a JSON file. Timebox A2A plumbing in phase 5 to one day.

## Current status

Phase 0 in progress — tooling scaffold done; Gemini/Langfuse smoke deferred (API credits pending).