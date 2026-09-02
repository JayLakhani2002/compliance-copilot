# ADR-0028 — Resilience: explicit timeouts, bounded retries, degraded fallback

**Status:** accepted 2026-08-30

## Context

`make_llm()` (`graph/nodes.py`) constructed `ChatOpenAI(model=model,
temperature=0, max_tokens=2048)` — and the `ChatAnthropic` branch the
same way — with no `timeout` and no `max_retries` kwarg at all. Verified
against the installed `langchain_openai`/`openai` packages: `ChatOpenAI`
always passes `"timeout": self.request_timeout` to the underlying `OpenAI(...)`
client construction, even when that value is `None` — and `None` is passed
through as a literal "no timeout" (`openai`'s `SyncAPIClient.__init__` only
falls back to its own default when the value is the `NOT_GIVEN` sentinel,
not a bare `None`). The answer call — the one call every `/ask` request
actually waits on end-to-end — could hang indefinitely on a stalled
connection. `OpenAIEmbeddings` (`embeddings.py`) had the identical gap.
Contrast `make_classifier_llm()`/`make_router_llm()`/`make_critic_llm()`
(ADR-0019/ADR-0023), which already pass an explicit `timeout=`/`max_retries=0`
— this feature closes the gap on the two clients that predate that pattern,
not a system-wide rewrite of clients that already had it right.

There was also no single **request-wide deadline**: `_stream_answer`
(api.py) ran `graph.astream(...)` to completion however long that took, and
a DB outage collapsed into the same generic `internal_error` bucket as any
other bug — a client (or an on-call engineer reading logs) couldn't tell
"the database is down" from "there's a code bug" without a full traceback.

Finally, an answer-model outage today would raise `openai.APITimeoutError`/
`APIConnectionError`/`RateLimitError` straight out of `answer_node`,
propagate through `graph.astream()`, and land in `_stream_answer`'s generic
`except Exception:` clause as a bare `internal_error` — never a bare 500 in
this SSE-shaped app, but the same "no partial/unfaithful answer, but also no
constructive fallback" outcome. A retrieval-only answer instead of a hard
failure is achievable and strictly better: the articles `retrieve_node`
already fetched are sitting right there in state, honestly presentable
even when the model that was supposed to synthesize them into prose never
got the chance to.

## Options considered

1. **Status quo — no explicit timeouts anywhere but the guard-tier clients.**
   Leaves the answer/embedding clients hanging on `openai`'s implicit
   defaults (`DEFAULT_MAX_RETRIES=2`, no timeout at all) — an unbounded call
   is a resource-exhaustion vector (the same "unbounded consumption" class
   the rate limiter and body-size cap already guard from a different angle),
   and there is no fallback path when the answer model is genuinely down.
2. **`langgraph.types.RetryPolicy` on `answer_node`** (installed, confirmed
   signature: `RetryPolicy(initial_interval=0.5, backoff_factor=2.0,
   max_interval=128.0, max_attempts=3, jitter=True, retry_on=default_retry_on)`,
   attached via `add_node(..., retry_policy=...)`). Rejected: a LangGraph
   node retry re-runs `_build_messages`/prompt construction from scratch on
   every attempt — heavier than the SDK's own retry one layer down, which
   already handles the identical transient-error set (`default_retry_on`,
   installed source, retries everything except `ConnectionError`/a 5xx
   `httpx.HTTPStatusError`/`requests.HTTPError`/a short programmer-error
   list — i.e. it WOULD retry `openai.APITimeoutError` etc. by default).
   Stacking a graph-level retry on top of the SDK's own bounded retry is
   exactly the **retry storm** risk this feature exists to avoid: one real
   outage turning into `graph_attempts × sdk_retries × N` calls to a
   service that's already struggling. `langgraph`'s native per-node
   `timeout:` kwarg (`langgraph.types.TimeoutPolicy`, also confirmed
   importable) has the same objection in miniature — a per-node budget,
   when the actual ask is one *request-wide* deadline, not a per-node one.
3. **SDK-level explicit timeout/retry + a node-level fallback + one
   request-wide deadline** (this ADR). Bounds every LLM/embedding client's
   own call individually (settings.py's new fields), lets `answer_node`
   catch the narrow, correctly-scoped set of "the model never answered"
   exceptions and degrade honestly instead of raising, and wraps the whole
   `_stream_answer`/`_stream_resume` generator (they share
   `_run_graph_and_stream`) in one `asyncio.timeout(settings.request_timeout_s)`
   for the request as a whole.

## Decision

Option 3.

### A — explicit timeouts + bounded retries (settings.py, one block)

| Client | Timeout | Max retries | Why |
|---|---|---|---|
| `make_llm()` (answer, graph/nodes.py) | `answer_timeout_s=30.0` | `answer_max_retries=1` | the one call every request waits on end-to-end; one bounded retry makes the SDK's previously-implicit `DEFAULT_MAX_RETRIES=2` an explicit, deliberate choice instead |
| `get_embeddings()` (embeddings.py) | `embedding_timeout_s=10.0` | `embedding_max_retries=2` | a short single-text call, idempotent by construction, no fallback path exists yet so it's worth a couple more bounded attempts |
| classifier (guards/classifier.py) | `classifier_timeout_s=3.0` (unchanged, ADR-0019) | `0` (unchanged) | already correct — fails open, a tight timeout matters more than a retry budget it would never use |
| router (router.py) | `router_timeout_s=3.0` (unchanged, ADR-0023) | `0` (unchanged) | already correct — same fail-open reasoning as the classifier |
| critic (critic.py) | `critic_timeout_s=3.0` (unchanged, ADR-0023) | `0` (unchanged) | already correct — `critique()` always returns a verdict (pessimistic on failure), nothing useful to retry to |
| offline judge (evals/judge.py's caller — `run_answer_eval.py`/`build_calibration_set.py`) | `judge_timeout_s=10.0` (new) | `1` (new) | a dev/CI-only batch call with no explicit budget before this feature — worth bounding so an outage stalls a CI job with a clear timeout, not an indefinite hang |

Classifier/router/critic needed **no code change** — they already had this
right (ADR-0019/ADR-0023 predate this feature). The task brief's framing
("timeouts on every LLM/embeddings client... today they can hang forever")
over-generalizes from the answer/embeddings gap; this ADR fixes exactly
those two clients (plus the offline judge, for the same reason) rather than
touching code that already matched the pattern.

### B — degraded fallback (`answer_node`, `critic_node`, `guard_out_node`)

`answer_node` wraps `runtime.context.llm.invoke(messages)` in
`try/except _ANSWER_OUTAGE_EXCEPTIONS` — `openai.APITimeoutError`/
`APIConnectionError`/`RateLimitError` AND their `anthropic` namesakes
(confirmed identical class names via `dir()` on both installed packages,
disjoint hierarchies) — so this stays correct if `settings.llm_provider`
ever flips to `"anthropic"` (ADR-0002's target) with no further code
change. `APIStatusError`/generic `APIError` are deliberately excluded: a
client-side 4xx (not 429) will fail identically on a retry, so masking it
as "degraded" would hide a real bug behind a user-facing "service degraded"
message — it still reaches `/ask`'s generic `internal_error` clause.

On catching one of those, `answer_node` returns:

```python
{
    "answer": AnswerSchema(
        answer=f"Service degraded — showing retrieved articles only: {listing}",
        citations=[],
    ),
    "degraded": True,
    "citation_error": None,
    "attempts": attempts,
}
```

where `listing` is `", ".join(f"{a.regulation} {a.anchor}" for a in
state["articles"])` — regulation+anchor only, never chunk text. `degraded`
is a new `GraphState`/`GraphContext`-adjacent key, added to `_PER_TURN_RESET`
(same reset-between-turns treatment `refused` already gets, ADR-0024) so a
degraded turn can't leak `degraded=True` into the next turn's
`guard_out_node`/`critic_node`.

Because `state["answer"]` is set, `route_after_answer` (build.py, unchanged)
routes to `critic` exactly as a real answer would — **no new conditional
edge was added**. Instead:

- `critic_node` returns `{}` immediately when `state.get("degraded")` —
  nothing was drafted to critique, and this skips the critic LLM call
  entirely (not just the pause).
- `hitl_node` needed **no code change**: with `critic_node` a no-op,
  `state["critic"]` stays `None` (the per-turn reset's existing default),
  and `hitl_node`'s existing `if critic is None or ...: return {}` already
  treats that exactly like "critic disabled."
- `guard_out_node`/`check_output` (guards/output.py) exempt `degraded` from
  the citation-shaped checks (`placeholder_leak` through `scope_unsupported`)
  the SAME way they already exempt `refused` — `check_output` gained a
  `degraded: bool = False` kwarg and its `if refused:` early-return became
  `if refused or degraded:`. This is the check that actually mattered: the
  degraded message is zero-citation, often over `_SCOPE_LENGTH_THRESHOLD`
  (400 chars), and doesn't use any of `_NON_ANSWER_MARKERS` — exactly the
  shape `scope_unsupported` would otherwise wrongly block. Canary/scaffold
  leak checks still run regardless — a fallback path must never become an
  unguarded shortcut past those. `guard_out_node` also escalates a
  degraded-answer check failure to `OutputGuardError` (raise, don't
  quietly re-refuse) the same way a `refused`-answer failure already is —
  the fallback's text is fixed-shape, built only from retrieved ids, so any
  check tripping on it means the CONSTRUCTION is broken, not the question.
- A degraded turn is **not appended to `history`** — `guard_out_node`'s
  `if refused or degraded:` branch skips `_capped_history` the same way a
  refusal does (nothing was actually answered, so there's nothing worth
  replaying as prior context into the next turn's prompt).
- SSE: the `answer`/`guard_out` branches in `_run_graph_and_stream`
  (api.py) track a `current_degraded` variable the same way they already
  track `current_refused`; the `final` event gains `"degraded": <bool>`.
  Tracing gets a `degraded` score (1.0) instead of `citation_valid` on that
  path — counted, not hidden (lesson 23's own framing), the same way
  `output_blocked`/`refused` are already visible signals, not just log
  lines.
- CLI: `cli.py`'s existing per-field answer printer already renders
  whatever `AnswerSchema.answer` says — the degraded message prints as
  ordinary answer text with `citations: []`; no separate CLI code path was
  added (the message itself already says "Service degraded," ponytail:
  a dedicated CLI banner is a nice-to-have, not required for the message to
  be honest — add one if a real user finds the plain-text version unclear).

### C — request-wide deadline (`api.py`)

**Round 1 shipped a broken version of this** (caught in review — see
"Round 2" below for the fix and why the naive version failed). The
CORRECT, shipped mechanism: `_run_graph_and_stream` drives
`graph.astream(...)`'s iterator manually (`astream_iter = graph.astream(
...).__aiter__()`, `update = await anext(astream_iter)` in a `while True:`
loop) instead of `async for`, so `asyncio.timeout()` can wrap ONLY that one
`anext()` call — every `yield _sse(...)` in the loop body sits OUTSIDE any
timeout scope. A `used` counter (not a fixed wall-clock deadline — see
below) accumulates the MEASURED duration of each individual `anext()` call
(a `finally` around it); before each call, `remaining =
settings.request_timeout_s - used` is passed as that call's own
`asyncio.timeout(remaining)` budget, and `remaining <= 0` raises
`TimeoutError` directly if the cumulative budget is already spent. A
`TimeoutError` (stdlib — `asyncio.TimeoutError` has been the same class as
the builtin since Python 3.11, confirmed live) — whether from
`asyncio.timeout`'s own expiry or the `remaining <= 0` check — is caught by
`except TimeoutError:` emitting `_sse("error", {"type": "timeout"})`, then
the generator returns; a `finally: await astream_iter.aclose()` on the
outer try runs on every exit path (normal completion, the `__interrupt__`
`return`, or any of the five `except` clauses), so the underlying
`Pregel.astream` generator is always closed instead of being left for GC/
`loop.shutdown_asyncgens()` to throw `GeneratorExit` into later.

**This budget bounds the app's OWN work (guard_in + router + retrieve + up
to `MAX_ATTEMPTS` answer calls + critic — build.py's
`MAX_LLM_CALLS_PER_REQUEST`) — never how long the CLIENT takes to read the
response.** A slow SSE client (or any socket backpressure) is uvicorn/
reverse-proxy territory, a separate, pre-existing concern this feature does
not touch — it must never silently borrow this budget.

**Round 2 (reviewer BLOCKER 1) — what round 1 got wrong, twice:**

1. *First shipped shape:* `async with asyncio.timeout(settings.request_
   timeout_s): async for update in graph.astream(...): ... yield ...` — a
   `yield` sits INSIDE the timeout scope. `asyncio.timeout()` cancels
   whatever the current TASK is doing at the deadline; it does not care
   which frame that is. When the generator is parked at that `yield`
   because a SLOW CONSUMER (not slow graph work) hasn't asked for the next
   chunk yet, the `CancelledError` lands in the CONSUMER's own await
   (Starlette's `await send(...)`), never inside `_run_graph_and_stream` —
   `except TimeoutError:` never runs, no `{"type": "timeout"}` event is
   ever sent, and the un-`aclose()`d generator later raised an actual
   `RuntimeError: async generator ignored GeneratorExit` at shutdown
   (reviewer-reproduced with two throwaway scripts, not theoretical).
2. *First FIX attempt (still wrong):* moving to a manual `anext()` loop
   with a fixed `deadline = time.monotonic() + settings.request_timeout_s`
   computed once, checking `remaining = deadline - time.monotonic()` at
   the top of each iteration. This correctly moves the `asyncio.timeout`
   scope off the `yield` — but a plain wall-clock deadline still counts
   elapsed real time regardless of WHERE it was spent: the gap between one
   `yield` and the consumer's next pull is real, ticking wall-clock time
   too, so a slow consumer could still exhaust `remaining` before the next
   `anext()` even started, reintroducing a milder version of the exact
   problem this fix exists to remove (caught by a new test — a fast graph
   with an artificial slow-consumer delay between pulls still tripped the
   timeout). Fixed by replacing the fixed deadline with the `used`
   accumulator described above, which only ever grows by a MEASURED
   `anext()` duration — time spent between calls (however long) is never
   added to it, so it cannot be inflated by consumer slowness.

Tests: `test_request_deadline_emits_timeout_event_and_ends_stream` (Case
A — slow graph work, via `TestClient`) unchanged and still passing;
`test_slow_consumer_between_chunks_does_not_trigger_timeout` (Case B —
new, drives `_run_graph_and_stream` directly with a real `asyncio.sleep`
between pulls longer than the deadline, asserts no timeout fires);
`test_request_deadline_is_cumulative_across_many_fast_then_slow_chunks`
(new — three separate node calls each individually under the deadline but
summing past it must still trip, proving the budget isn't reset per
chunk).

### D — DB failure specificity + `/readyz`

`_run_graph_and_stream` gained `except OperationalError:` (before the
generic `except Exception:`) emitting `{"type": "dependency_unavailable"}` —
this project's only failure surface for a mid-graph DB error is the SSE
stream (there is no non-streaming path to return a bare 503 from), so the
brief's "503 for non-stream paths" language doesn't map onto this
codebase's shape; a typed SSE `error` event is the equivalent here. A new
`GET /readyz` route runs `session.execute(text("SELECT 1"))` through the
existing `get_session` dependency, returning 200 or (on `OperationalError`)
503 — no auth, no rate limit (`@limiter.exempt`, mirroring `/healthz`).
`/healthz` is unchanged: still zero DB work, proven by a new test that
overrides `get_session` to a session whose `.execute()` always raises and
asserts `/healthz` still returns 200 while `/readyz` returns 503 with the
SAME override in place — liveness genuinely doesn't share fate with
readiness.

## Why not the others

RetryPolicy/per-node timeout (option 2) were rejected primarily for the
retry-storm risk (re-explained above) and secondarily because the ask was
for one request-wide budget, which a per-node mechanism can't express
without summing every node's own timeout by hand anyway — `asyncio.timeout`
around the whole streaming loop says exactly what's meant, once.

A tenacity `@retry` wrapper around the answer call was considered and
rejected the same way ADR-0019-adjacent code already implicitly rejected it
for the guard-tier clients: `tenacity` is already a transitive dependency
(pulled in by another package, confirmed in `uv.lock`, not a direct
`pyproject.toml` dependency), but stacking it on top of the SDK's own
retry is the same double-retry risk as option 2, just implemented as a
decorator instead of a graph feature. The one place a second retry LAYER
earns its keep is a cross-NODE fallback — which is exactly what B is, not
a same-call retry.

## Security & cost

An unbounded LLM/embedding call is a resource-exhaustion vector — the same
"unbounded consumption" category the rate limiter (ADR-0016) and the
body-size cap already guard from a different angle. A hung provider
connection tying up a worker indefinitely doesn't require an attacker to
cause it on purpose; a slow day at the provider is enough. Bounded retries
cap the worst-case spend per failing request (an SDK silently retrying on
a persistent 5xx multiplies token spend on calls that were already going
to fail); a degraded response spends only what the already-failed attempt(s)
cost and pays nothing further — cheaper than the ceiling a healthy run
already hits (`tests/evals/test_trajectory.py`'s new
`test_answer_outage_trajectory_makes_fewer_llm_calls_than_the_ceiling`
proves this, not just asserts it). The degraded fallback still passes
through `guard_out`'s canary/scaffold leak checks — a fallback path must
never become an unguarded shortcut for a prompt-injection payload that
somehow ends up echoed into the fallback text (it can't today, since the
fallback is built only from `RetrievedChunk.regulation`/`.anchor`, both
our own DB-assigned ids — but the check running anyway is defence in
depth, not a hole being left open on the assumption that can't happen).

## How to reverse

- Timeouts/retries: revert the `timeout=`/`max_retries=` kwargs on
  `make_llm()`/`get_embeddings()` and drop the five new `settings.py`
  fields — a one-file, one-block change (settings.py's own "one place to
  see every config key" design).
- Degraded fallback: remove `_ANSWER_OUTAGE_EXCEPTIONS`'s `try/except` in
  `answer_node` (the exception then bubbles up exactly as it did before
  this feature, into `_stream_answer`'s generic `internal_error` clause);
  drop `degraded` from `_PER_TURN_RESET`/`GraphState`; revert
  `check_output`'s `degraded` kwarg. Each of these is independent — this
  feature doesn't require reverting all of it at once.
- Request deadline: remove the `async with asyncio.timeout(...):` wrap and
  its `except TimeoutError:` clause; `settings.request_timeout_s` becomes
  unused and can be deleted with it.
- `/readyz`: delete the route; `/healthz` is untouched either way.

## References

- ADR-0002 (LLM provider/model choice), ADR-0007 (MCP tool timeout, a
  DIFFERENT budget — per-tool-call, not per-request or per-answer-call).
- ADR-0014 (citation validation/retry-once — this feature does not touch
  the citation-retry loop; a citation failure and an answer-model outage
  are different failure classes with different fallbacks by design).
- ADR-0019/ADR-0023 (classifier/router/critic's pre-existing timeout/
  fail-policy pattern, reused rather than reinvented here).
- ADR-0021 (guard_out's `refused` exemption, the pattern `degraded` mirrors).
- ADR-0024 (per-turn reset, extended here to cover `degraded`).
