# ADR-0026 — Trajectory evals: naming and isolating the path checks

**Status:** accepted 2026-08-29

## Context

By Week 5 the graph has real branching: `guard_in -> router -> (refuse |
retrieve) -> answer(+retry, capped at MAX_ATTEMPTS) -> critic -> hitl ->
guard_out`. `tests/test_graph.py` already asserts several of these paths —
`test_flagged_question_stream_visits_guard_in_then_refuse_never_answer`
pins the exact `["guard_in", "refuse", "guard_out"]` sequence,
`test_stream_updates_visits_answer_twice_and_never_fail_on_retry_then_success`
pins the retry count — but each lives as one assertion inside a test named
for something else, and nothing yet states a ceiling on how many LLM calls
one request can make. ADR-0022's red-team gate already proved the value of
this category of check one level down (`blocked_by(state, success_if)`
classifies every attack run by WHICH LAYER caught it, not just whether it
was caught) — this ADR generalises the same instinct to ordinary traffic.

A trajectory eval answers a different question than an answer-quality eval
(ADR-0005/0017): not "is the output right" but "did the run get there the
right way." A router that mislabels a GDPR question `ai_act`, whose
retrieval then finds the right article anyway because the two corpora
overlap, produces a perfectly faithful answer an answer-only judge scores
1.0 — while the router's own accuracy metric should show a miss. Only a
trajectory check can see that gap.

## Options considered

1. **Final-answer evals only** (extend ADR-0005/0017's faithfulness/
   relevancy judge, add nothing else). Rejected: structurally blind to a
   right-answer-via-wrong-path bug (the GDPR/AI-Act example above), a
   silently-skipped critic, or a retry that fires on every question and
   just overwrites a fine first draft — none of these change the final
   text a user sees.
2. **A dedicated trajectory-evaluation library** (e.g. LangSmith's
   trajectory evaluators). Rejected for the same EU/self-hosting reason
   ADR-0005 already rejected LangSmith evals generally — a paid, US-hosted
   product for a project whose narrative is EU data residency. It would
   also duplicate what `graph.astream(..., stream_mode="updates")` already
   gives for free: the exact ordered list of node names a run visited
   (verified live, ADR-0025's own references section).
3. **In-repo structural tests with fakes, named and isolated as their own
   category** (this ADR). No new dependency, no new marker: LangGraph's own
   `stream_mode="updates"` output IS the trajectory; a plain
   `list(update)[0]` per chunk collects it. Same "the test IS the eval"
   pattern ADR-0022 already established for the red-team gate.

## Decision

Option 3. Concretely:

- **`tests/graph_helpers.py`** (new) — the fake LLM/tool doubles and
  `_run`/`_run_stream`/`_run_turn`/`_resume_turn` helpers, moved out of
  `tests/test_graph.py` (which now imports them) rather than duplicated
  into a second copy for `tests/evals/test_trajectory.py`. Two new stream
  siblings, `_run_turn_stream`/`_resume_turn_stream`, generalise the
  existing `_run_stream` pattern to a checkpointed pause+resume run.
- **`tests/evals/test_trajectory.py`** (new) — eight named trajectory
  tests (exact `nodes_visited` list or exact counts): the two refusal
  paths, the happy path with a router label, retry-then-success, the
  low-confidence pause (and its resume), a critic outage, the exact
  `critic_confidence_min` boundary (`<` not `<=`, pinned in both
  directions), and a `guard_out`-level policy-violation refusal (a canary
  leak). Verified live (not guessed) that a pausing run's LAST trajectory
  entry is the literal string `"__interrupt__"`, never `"hitl"` itself —
  `hitl_node` halts INSIDE itself before returning any node update
  (ADR-0025's idempotency design); `"hitl"` only appears, once, on the
  RESUME call that follows.
- **`MAX_LLM_CALLS_PER_REQUEST = 1 + 1 + MAX_ATTEMPTS + 1 = 5`**
  (`graph/build.py`, next to `MAX_ATTEMPTS`) — classifier + router +
  answer(≤`MAX_ATTEMPTS`) + critic. `hitl` makes no LLM call of its own (a
  design constraint, not an oversight — see `hitl_node`'s idempotency
  docstring), so it adds nothing. A `CountingInvoke` spy wraps each fake
  in a `GraphContext` and totals real invocations across one run; two
  tests prove the happy path and the retry path never exceed this number
  (the retry path reaches it exactly, proving the bound is tight, not
  generous); a third, plain unit test asserts the constant equals its own
  stated derivation, so the two can't silently drift apart.
- **Router-label accuracy**: the Day-18 ten-question fixture
  (`graph_helpers.ROUTER_FIXTURE`) is reused (not copied) three ways —
  `test_graph.py`'s existing per-question mechanism tests, ONE new
  aggregate accuracy test in `test_trajectory.py` (structural, fake
  router, default marker), and ONE new gated `@pytest.mark.integration`
  test **added to** `tests/test_router_real_integration.py` (not a new
  file) asserting ≥9/10 correct against the real `gpt-4.1-nano` — smaller
  diff than inventing a second gating file, mirrors that file's existing
  three-question smoke test exactly. Measured 2026-08-29: **10/10** on
  both the original 3-question smoke test and the new 10-question fixture.

## Why not the others

- **Final-answer evals only**: the whole point of this ADR is the failure
  mode this misses — see Context above.
- **A dedicated trajectory library**: LangGraph's own stream already IS
  the trajectory; wrapping it in another library buys nothing but a new
  dependency and, for any hosted option, a US-residency inconsistency this
  project has already rejected once (ADR-0005).

## Security & cost implications

- **Security:** none of these assertions read question or answer TEXT —
  only node names, tool-call argument shapes (`regulation` filter values),
  and counts, the same "shape, not content" logging discipline every guard
  node already follows.
- **Cost:** the structural suite (Deliverables A/B/C-structural) is free —
  fakes only, runs on every push in the default `pytest -m "not
  integration"` job (no CI change needed: `testpaths = ["tests"]` already
  covers `tests/evals/`, and these tests carry no `integration` marker).
  The one gated real-model test costs a few cents (ten short
  `gpt-4.1-nano` completions) and only runs on `pytest -m integration`
  (manual/nightly/labelled, same two-tier split as ADR-0005/0017/0022).

## How to reverse

- Delete `tests/evals/test_trajectory.py` and the added test in
  `tests/test_router_real_integration.py` — `tests/graph_helpers.py` stays
  (it's still `test_graph.py`'s own dependency) or, if reverting that far
  too, re-inline the moved fakes back into `test_graph.py` — a pure
  file-move, no behaviour to unwind either way.
- `MAX_LLM_CALLS_PER_REQUEST` is an unread-by-production-code constant
  (test-only today) — deleting it from `build.py` breaks only the two
  tests that import it.
- No Makefile/CI change was made (see `docs/EVALS.md`): nothing to revert
  there.

## References

- ADR-0005 (eval harness, trajectory assertions named as a first-class
  category from the start), ADR-0017 (two-tier CI gate pattern), ADR-0022
  (the red-team `blocked_by`-layer precedent this ADR generalises),
  ADR-0023 (router/critic), ADR-0025 (hitl/interrupt, including the live
  verification that `stream_mode="updates"` emits a pause as a distinct
  `{"__interrupt__": (...)}` chunk).
- Live-verified (this feature): a paused run's trajectory ends with
  `"__interrupt__"`, never `"hitl"`; the resume call's trajectory is
  `["hitl", "guard_out"]` on `approve`.
