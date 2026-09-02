# ADR-0029 — Cost engineering: measured per-question cost, not a hand estimate

**Status:** accepted 2026-09-02

## Context

`docs/ARCHITECTURE.md` §9's cost model was a hand-computed estimate ("well
under $0.02/question") built from assumed token counts for a Claude
Haiku/Sonnet call shape — a shape the shipped system doesn't even use
(ADR-0002's 2026-08-24 amendment moved the default provider to OpenAI's
`gpt-4.1-mini`/`gpt-4.1-nano`). Nothing in the running system attaches a
real cost figure to any request; ADR-0016's rate limit bounds worst-case
call *volume*, but nothing converts "a call happened" into "here's what it
cost." A cost model nobody has measured against a real run is a guess
wearing a table.

## Options considered

1. **Estimate-only** (status quo) — keep hand-computed token-count
   assumptions in `docs/ARCHITECTURE.md`. Costs nothing to maintain, but a
   stale assumption (wrong model, wrong token count, a prompt that grew)
   silently drifts from reality with nothing to catch it.
2. **Langfuse-dashboard-only** — rely on Langfuse Cloud's own cost
   computation (its model-price matching) once traces exist. Rejected as
   the *only* mechanism: the week-5 researcher pass explicitly could not
   confirm Langfuse's price table covers `gpt-4.1-mini`/`gpt-4.1-nano` (the
   only hit in installed `langfuse` source was an unrelated example string,
   not a price-list entry) — betting the entire cost story on an unverified
   third-party price match is the same "assume it's right" risk this ADR
   exists to remove. Also: Jay has no Langfuse Cloud account today
   (`tracing.py`'s disabled-by-default contract), so this produces zero
   numbers until that changes.
3. **In-repo measured report** (chosen) — a small price table
   (`compliance_copilot.costing.PRICES`) plus `estimate_cost()`, driven by
   real per-question token usage from a real run of the golden set
   (`evals/run_cost_report.py`), reported in `docs/EVALS.md`/
   `docs/ARCHITECTURE.md` with a measurement date.

## Decision

**(a) `costing.py`** — a plain module, no classes: `PRICES` (per-MTok
in/out/cached-input, both shipped OpenAI models + embeddings + the two
documented Anthropic alternatives) and `estimate_cost(usage_by_model) ->
{"usd_total", "eur_total", "by_model"}`, honoring each model's
cached-input discount off `input_token_details.cache_read`. Every price
row is dated in a comment — the same "won't silently drift and lie"
discipline ADR-0002's own pricing notes already use. `settings.eur_usd_rate`
is a new field: a point-in-time USD→EUR snapshot (0.92, 2026-08-30 ECB
reference), explicitly NOT a live FX feed — an honest scope cut for a
portfolio project, not a hidden gap.

**(b) `evals/run_cost_report.py`** (`make cost-report`) — runs the 10
golden questions through the FULL production call shape (guard_in's
classifier, the router, retrieve, the answer call, the critic — whichever
`settings.*_enabled` turns on), with `langchain_core`'s
`get_usage_metadata_callback()` attached per question. That callback pools
usage by model name automatically, which is why the guard-tier calls
(classifier/router/critic — they all default to `gpt-4.1-nano`) show up as
one pooled bucket rather than split per node: the callback has no
node-identity to split by, and this ADR reports that honestly rather than
inventing a false split. Query-embedding cost is the one call this can't
observe via callback — `embed_query()` runs inside the MCP server
subprocess (a separate process over stdio, ADR-0007's Day-17 amendment),
invisible to any callback in the report's own process — so embedding
tokens are counted locally with `tiktoken` (the same BPE the embeddings
endpoint uses) instead, a deterministic substitute, not a guess.

**(c) A per-question wall-clock backstop, script-local, not app config.**
Live during measurement (2026-09-02): the MCP stdio subprocess wedged
completely on the second golden question of the first run — zero CPU, no
further log output, no `mcp_tool timeout` from `retrieve_node`'s own
per-call `asyncio.wait_for` (so the stall sat somewhere that per-call
timeout doesn't cover). Since `evals/run_cost_report.py` is a batch job
with 10 sequential real network calls, a 90-second per-question
`asyncio.wait_for` was added around the whole `ask_graph()` call, purely in
the report script — a stuck question is marked `degraded` (whatever usage
happened before the stall is still reported, not zeroed) and `tools` is
respawned (a fresh subprocess) before the next question, so one wedged
pipe can't take out the rest of the run. This is NOT a fix to the
underlying MCP transport issue — see "Open risk" below.

**(d) Measured result** (n=10, full detail in `docs/EVALS.md`): **€0.130
per 100 questions**, cached-token fraction 63.2% overall (81.7% on the
answer call, 34.7% on the pooled guard-tier calls — OpenAI's automatic
prefix caching needs a prompt over roughly 1,024 tokens, and the shorter
guard-tier prompts miss it more often; this is what the measured gap shows,
not a theory). One of the 10 golden questions (`c01`, a cross-regulation
AI-Act+GDPR question) reproduced a citation-validation refusal on both
measurement runs — real behavior, not a report bug, flagged `degraded`
rather than silently counted as a normal answer.

## Why not the others

- **Estimate-only**: rejected — it was already wrong (wrong provider,
  wrong models) and nothing would have caught that until this feature
  actually ran a real question through the real graph and counted.
- **Langfuse-dashboard-only**: rejected as the sole mechanism per option 2
  above — unverified price-table coverage for the shipped models, and zero
  numbers without a Langfuse Cloud account that doesn't exist today. Once
  Langfuse Cloud is live (ADR-0009), its dashboard is a reasonable
  *second* view on the same numbers, not a replacement for a repo-owned
  measured report that works with zero external dependencies.

## Security & cost implications

- **Security:** `evals/run_cost_report.py`'s JSON output (`--json`) holds
  only token counts and cost figures per golden-question id — no question
  text, no answer text, no PII. The golden questions themselves are
  already public, committed fixtures (`evals/golden_answers.jsonl`), not
  user data.
- **Cost:** the report itself costs a fraction of a cent per run (10
  questions, cheap-tier guard calls + one `gpt-4.1-mini` answer call each)
  — run manually, not wired into CI (same "real LLM spend, don't gate every
  push on it" reasoning ADR-0017 already applies to `answer-quality`).

## Future work (not implemented today)

The measured data shows the guard tier (`gpt-4.1-nano`: classifier + router
+ critic) caching at only 34.7% versus the answer tier's 81.7% — three
separate LLM round-trips per question, each with a short enough prompt to
often miss OpenAI's ~1,024-token caching threshold. A low-risk optimization
worth investigating later: consolidating the classifier and router calls
(both run before retrieval, both cheap structured-output classifications
over the same question text) into a single nano-tier call, cutting one full
round-trip per request and giving the remaining combined prompt a better
shot at crossing the caching threshold. Not implemented here — this
feature's scope is measurement, and a call-shape change needs its own eval
run to prove quality held (the same "never ship a cost-cutting change
without eval numbers" rule `docs/lessons/24_cost_engineering.md` states).

## Open risk

The MCP stdio subprocess wedge observed during measurement (point (c)
above) is a real, reproducible-looking failure mode in `retrieve_node`'s
MCP client path, not an artifact of this report script — it happened
after roughly 15–20 tool calls with no per-call timeout firing. This ADR's
90-second backstop keeps the *report* from hanging forever; it does not fix
the underlying transport issue. Worth a focused investigation (a
request-wide budget around `retrieve_node`'s whole tool-fetch loop, or
checking whether the MCP server's own logging is corrupting its stdio
JSON-RPC stream) before this path carries real production traffic.

**Sharper wedge diagnosis (review round 1).** `langchain-mcp-adapters`'s
client creates *a new session — a fresh subprocess spawn + handshake — for
every tool call*; the observed wedge (zero CPU, `asyncio.wait_for` never
firing) is consistent with a spawn-cycle syscall that cooperative
cancellation cannot preempt, not with the timeout being "in the wrong
place" and not with the one-time `get_tools()` load (which had already
succeeded). Backlog fix, before production traffic: a persistent shared MCP
session, or a hard PID-level kill on timeout — asyncio cancellation alone
cannot force a stuck syscall to return.

## How to reverse

`costing.py` and `evals/run_cost_report.py` are additive — deleting both
files and the `cost-report` Makefile target fully reverts this feature;
nothing else in the app imports from `costing.py` today (it is not wired
into the live request path — see "Future work" above for that natural next
step, deliberately not built here).

## References

- ADR-0002's amendment (2026-08-24): `gpt-4.1-mini`/`gpt-4.1-nano` as the
  shipped provider/model tiers this ADR measures.
- `developers.openai.com/api/docs/pricing`, checked 2026-08-30 — the
  `gpt-4.1-mini`/`gpt-4.1-nano`/`text-embedding-3-small` price rows in
  `costing.PRICES`.
- ADR-0002's recorded Anthropic pricing (Haiku $1.00/$5.00, Sonnet
  $3.00/$15.00 per MTok, 2026-08-23) — the two ALTERNATIVE-path rows in
  `costing.PRICES`, not re-verified in this ADR.
- `langchain_core.callbacks.usage.UsageMetadataCallbackHandler`/
  `get_usage_metadata_callback` and `langchain_core.messages.ai.
  UsageMetadata`/`InputTokenDetails` — verified against installed
  `langchain-core` source, 2026-08-30/09-02.
- `docs/EVALS.md`'s "Cost per question" section — the full measured table
  and per-item discussion this ADR summarises.
