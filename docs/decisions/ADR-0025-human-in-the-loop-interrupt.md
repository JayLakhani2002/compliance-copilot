# ADR-0025 — Human-in-the-loop: `interrupt()` on low critic confidence

**Status:** accepted 2026-08-29

## Context

ADR-0023 shipped a critic node that scores whether a drafted answer's prose
claims actually follow from its own cited excerpts (semantic faithfulness),
but it only *records* that verdict — nothing branches on it yet
(`docs/ARCHITECTURE.md` §4's own "records first, blocks later" note).
`settings.critic_confidence_min` (default 0.6) already exists as an unread
placeholder for exactly this day. ADR-0024 made the graph durable
(`AsyncPostgresSaver`, a `thread_id` per conversation) — a prerequisite for
this feature, not a coincidence: LangGraph raises `RuntimeError` if
`Command(resume=...)` is used without a checkpointer.

The question this ADR answers: what happens when the critic's confidence in
an otherwise citation-valid, schema-valid answer is low? Shipping it anyway
risks a wrong-but-confident-sounding legal-adjacent answer reaching a user.

## Options considered

1. **A `"confidence": "low"` label in the response, nothing else.** Cheapest
   to build, but a label is something a client can just as easily ignore as
   read — nothing structurally stops the answer from reaching a user
   regardless of the score.
2. **Synchronous human review before ANY answer is returned** (a human in
   the loop on every request). Defeats the point of an automated assistant
   — most answers are fine; gating every single one on a human would move
   all latency and cost onto attention, not just the borderline cases.
3. **`interrupt()` + the existing Postgres checkpointer, only on a
   low-confidence verdict** (this ADR). The run cannot continue until a
   human decision arrives; the pause is durable across a process restart
   (ADR-0024's checkpointer already gives this for free); it costs nothing
   extra when the critic is confident, which is the common case.

## Decision

Option 3.

### `hitl_node` (`graph/nodes.py`), wired between `critic` and `guard_out`

- Pure pass-through (`return {}`) unless the critic ran **and**
  `critic.confidence < settings.critic_confidence_min` — mirrors
  `router_node`/`critic_node`'s own "disabled/not-applicable means no-op"
  contract, so a disabled critic (`GraphContext.critic=None`) or a
  confident verdict never pauses anything.
- **Idempotency is the load-bearing design constraint.** The installed
  `langgraph/types.py`'s `interrupt()` docstring states explicitly: "The
  graph resumes from the start of the node, re-executing all logic." A node
  that made an LLM call (or any other side effect) before `interrupt()`
  would silently repeat that call on every resume. `hitl_node` takes no
  `runtime` and makes no LLM call at all — the confidence check re-reads
  already-checkpointed state (`state["critic"]`, `state["answer"]`), a
  cheap comparison safe to repeat.
- `interrupt(payload)` checkpoints and surfaces exactly three things to the
  operator: the draft answer (`answer.model_dump()`), the critic's
  confidence + reasoning, and the question — already `guard_in`-redacted by
  the time it reaches this node (no new PII exposure; same trust-boundary
  posture ADR-0024 already established for checkpointed state).
- On resume, `interrupt()` returns the decision payload
  (`{"decision": "approve"|"edit"|"reject", "edited_answer": str | None}`)
  and the node returns a `Command`:
  - `approve` → `Command(goto="guard_out")` — no `update`, the draft
    proceeds unchanged.
  - `edit` → `Command(update={"answer": AnswerSchema(answer=edited,
    citations=<the draft's own citations>)}, goto="guard_out")`. The
    draft's citations are kept, **never** the operator's own claimed
    citations (there are none to trust) — this is a deliberate reversal of
    "a human wrote it, so it's safe": `guard_out`'s checks apply to this
    text exactly as they would to a model's, because a human can introduce
    the same categories of problem (a canary leak, a scaffold artifact, a
    scope-drift ramble) a model can.
  - `reject` → `Command(update={"answer": AnswerSchema(answer=
    REFUSAL_TEXT, citations=[]), "refused": True}, goto="guard_out")` — the
    SAME fixed refusal shape `refuse_node` produces, so a client handles
    exactly one refusal shape regardless of which path produced it
    (ADR-0021's "guard blocks, never swaps" posture, reused here).
- **Every branch still routes through `guard_out`** (`build.py`:
  `critic -> hitl -> guard_out`, an unconditional edge on the pass-through
  path; a resumed node's `Command(goto="guard_out")` dynamically overrides
  that same edge, verified live against the installed `langgraph` package
  — a node's `Command.goto` wins regardless of a static edge also
  registered for that node). ADR-0021's invariant — `guard_out` is the
  final gate on every terminal path — is unchanged by this feature, just
  one hop later on the branch that paused.

### API (`api.py`)

- `_stream_answer`'s astream loop checks for a `{"__interrupt__": (...)}`
  chunk (verified live against the installed `langgraph` pregel loop,
  matching the documented shape) and, when present, emits SSE event
  `interrupt` — `{thread_id, draft, confidence, reasoning, interrupt_id}`
  — then ends the stream with **no** `final` event.
- New `POST /resume` — same auth (`X-API-Key`) and rate limit
  (`SlowAPIMiddleware`) as `/ask`. Body: `{thread_id: str, decision:
  "approve"|"edit"|"reject", edited_answer: str | None}`,
  `extra="forbid"`, `edited_answer` required if and only if
  `decision == "edit"` (a `model_validator`, not just docstring
  convention), capped at `settings.max_question_chars` (reused — no
  separate "max answer length" setting exists, same order of magnitude of
  text as a question).
- **Validated BEFORE the streaming response starts, not inside the
  generator**: `graph.aget_state(config)` — an unknown `thread_id` (no
  checkpointed values at all, or no checkpointer configured) is 404; a
  known thread with `snapshot.next`/`snapshot.interrupts` both empty (not
  currently paused) is 409. Raising an `HTTPException` from inside a
  `StreamingResponse` body would arrive after a 200 and headers were
  already sent — this has to happen in the route handler itself.
- On success, resumes via `graph.astream(Command(resume={"decision": ...,
  "edited_answer": ...}), ...)` and streams the remainder (`guard_out ->
  final`) through the SAME event-emission loop `/ask` uses (extracted into
  `_run_graph_and_stream`, shared by both routes) — no separate "resume"
  event vocabulary to maintain.
- `current_answer`/`current_refused` (the loop's running "what does `final`
  carry" state) are seeded from the paused snapshot's own values when
  resuming — `answer_node`/`critic_node` already ran (and were streamed) in
  the EARLIER call that paused; this resume's own `astream` never re-emits
  them, so without seeding, an `approve` resume (which writes no `update`
  at all) would try to `.model_dump()` a `None`.

### CLI (`cli.py`)

- `ask` prints `under review (thread_id ...): critic confidence N below
  threshold` to stderr (exit code 6) instead of an answer, when
  `graph.ainvoke`'s return dict carries a top-level `__interrupt__` key.
- New `resume <thread_id> --decision approve|edit|reject [--answer TEXT]`
  — validates the thread is known and paused first (same
  `snapshot.values`/`.next`/`.interrupts` check `api.py`'s
  `_require_paused_thread` uses), prints a clean error instead of a
  `Command(resume=...)` failure three calls deep.

## Why not the others

- **A confidence label only (option 1)**: rejected — structurally
  indistinguishable from every other piece of metadata a client can ignore;
  it doesn't stop a wrong-but-confident answer from reaching a user, which
  is the actual risk this feature exists to close.
- **Synchronous review on every request (option 2)**: rejected — most
  answers score confidently; gating all of them on a human converts a
  cheap automated system into a slow manual one, for no benefit on the
  large majority of requests that were never uncertain.

## Security & cost implications

- **Shared-API-key gap, still open (ADR-0016, not solved here).** This API
  has one shared `X-API-Key` across every caller — any key holder who
  knows or guesses a valid `thread_id`+`interrupt_id` pair can resume it
  (round 2 adds the `interrupt_id` check, but it only guards against a
  STALE reference, not against a caller who never should have had the pair
  in the first place). There is no binding between the key that started a
  run and the key allowed to resolve its pause. Documented in `/resume`'s
  own docstring; accepted, not fixed, the same posture ADR-0016/ADR-0024
  already take for the equivalent gap on `/ask`'s `thread_id`.
- **Stale paused runs never expire.** `# ponytail:` — a paused thread
  nobody ever resumes stays paused (and its checkpoint row) forever;
  LangGraph checkpoints have no built-in TTL. Add a background job that
  auto-expires a stale pause into a refusal past a retention window the
  day real traffic makes this matter — not before (today's traffic is the
  author's own testing).
- **Zero extra LLM cost.** `interrupt()`/`Command()` are pure control-flow
  primitives — the pause itself makes no API call. The real cost is an
  operator's reading time, which is exactly why
  `settings.critic_confidence_min` starts conservative and is meant to be
  tuned from real critic-score traffic (ADR-0023's own "records first"
  design), not intuition.
- **No new PII exposure.** The interrupt payload is built from
  already-`guard_in`-redacted state — same trust-boundary posture the
  checkpointer already carries every other piece of graph state at
  (ADR-0001/ADR-0024).
- **Idempotency, verified, not assumed.** `hitl_node` makes no LLM call —
  pinned by `tests/test_graph.py::test_resume_does_not_recall_
  answer_or_critic_llm`, which counts `answer`/`critic` LLM invocations
  across a pause+resume and asserts neither increases.

## Round 2 (reviewer request-changes, 2026-08-29)

Round 1 review found two BLOCKERs and three SHOULDs, all fixed in this
revision. Each decision below is a deliberate tradeoff, not a default —
documented here rather than left implicit in the diff.

### BLOCKER 1 — critic-outage pause storm

**Problem:** `critique()`'s exception path (critic.py) returns a
pessimistic verdict (`confidence=0.0`) on ANY failure — a rate limit, a
timeout, a provider 5xx. `hitl_node`'s original gate (`critic.confidence <
critic_confidence_min`) could not distinguish that from a genuine
low-faithfulness verdict, so a nano/Haiku-tier outage would pause EVERY
single request for human review — an availability failure on the exact
same shape ADR-0019 already reasoned about for the classifier's own
fail-open, shipped on the opposite (fail-closed-to-human) default without
the comparison ever being made.

**Decision:** `CriticVerdict` gains `error: SkipJsonSchema[bool] = False`
— `SkipJsonSchema` (pydantic 2.13) keeps it OUT of the structured-output
JSON schema `with_structured_output(CriticVerdict, ...)` sends to the
model (verified: `CriticVerdict.model_json_schema()`'s `required` list has
no `error`), so it can only ever be set by `critique()`'s own exception
handler, never by the LLM deciding on its own that it "errored".
`hitl_node`'s gate becomes `critic is None or critic.error or
critic.confidence >= critic_confidence_min` — an outage now falls through
to `guard_out` exactly like a disabled critic does, no pause. Logged as a
named `critic_unavailable` guardrail event (`critic_node`, once per turn —
NOT in `hitl_node`, which re-executes on every resume and would log it
repeatedly) and a Langfuse score (`api.py`'s critic branch), so the
tradeoff is visible, not silently absorbed.

**Tradeoff, explicit:** LLM-judge coverage (the faithfulness check) is
lost for the duration of a critic-tier outage — an answer that WOULD have
paused for review under normal conditions now ships un-reviewed, backed
only by `guard_out`'s independent deterministic checks (citation-exists,
scaffold/canary/placeholder, scope heuristic — still run regardless). This
is the same tradeoff ADR-0019 already accepted for the classifier: an
outage-triggered block turning into a full product outage costs more than
the marginal safety loss of one guard layer going dark temporarily.
Rejected the alternative (round-1's option (i): keep pausing on an outage,
but tag it `"cause": "critic_error"` so an operator can bulk-triage) —
that still turns an outage into an operational incident (a growing queue
of pauses nobody asked for), where fail-open turns it into a metrics
signal instead (`critic_unavailable`'s count/rate is exactly as
actionable, without the availability cost).

### BLOCKER 2 — `/ask` could silently supersede a paused thread; `/resume` had no anti-staleness check

**Problem, reproduced live:** `graph.astream({"question": ...},
config=already-paused-thread-id)` was never rejected — LangGraph happily
started a new run from `START` on the SAME thread and OVERWROTE the paused
checkpoint (the original draft/critic verdict/interrupt vanished with no
error, no log line). Separately, `ResumeRequest` had no `interrupt_id`
field — an operator holding a stale `interrupt_id` (from an SSE event
issued before someone else's `/ask` call re-paused the same thread on a
different question) could have their approve/edit/reject decision silently
applied to the WRONG draft.

**Decision, two parts:**
1. `/ask` (`api.py`'s `ask()` route) and CLI `ask` (`_run_ask`) both run a
   pre-flight `graph.aget_state(...)` check on a CLIENT-SUPPLIED
   `thread_id` (a freshly-minted one has no prior state, skip the
   round-trip) — if `snapshot.next`/`.interrupts` show it's currently
   paused, reject: 409 for the API (`_reject_if_paused`), exit code 9 for
   the CLI, both BEFORE any run starts. Mirrors the status `/resume`
   already uses for the inverse condition ("not currently paused").
2. `ResumeRequest` gains a REQUIRED `interrupt_id: str`. `_require_paused_
   thread` (extended, not a new function) compares it against
   `snapshot.interrupts[0].id` — the ACTUAL pending interrupt's id, read
   fresh from `aget_state` — and 409s on any mismatch. The CLI `resume`
   command doesn't take a `--interrupt-id` flag at all: it reads
   `snapshot.interrupts[0].id` directly off the SAME freshly-fetched
   snapshot it validates against, so there is no window for it to drift
   from what's actually pending (simpler than threading a flag through,
   and structurally can't go stale the way a cached HTTP client's copy
   can).

**Tradeoff, explicit:** `/ask`'s pre-flight check is one extra
`aget_state` round-trip per request that supplies an existing `thread_id`
(the common multi-turn case) — a few extra milliseconds against Postgres,
traded for closing a real data-loss/misattribution class of bug. Not
optimized further (e.g. caching "is this thread paused" between requests)
— premature until this measurably matters.

### SHOULD 1 — the end user must not see the draft

**Problem:** the `interrupt` SSE event forwarded `draft`/`confidence`/
`reasoning` on the SAME response stream `/ask`'s caller reads — directly
contradicting ADR-0023's own "Free-text leak guard" rule ("the SSE stream
... never forwards `.reasoning`... never into a response the end user
sees"), never acknowledged as a reversal.

**Decision:** the `interrupt` SSE event now carries ONLY `{thread_id,
interrupt_id, status: "under_review"}`. The full payload (draft answer,
confidence, reasoning, question) still lives inside `interrupt(...)`
itself as CHECKPOINTED STATE (`hitl_node`, graph/nodes.py) — unchanged,
since that's what a `/resume` decision has to be informed by. An operator
reads it via `graph.aget_state()`; the CLI `resume` command does exactly
that and prints question/draft/confidence/reasoning to stderr BEFORE
applying the decision (no `--show` flag needed — always shown, since a
decision made blind defeats the entire point of a human-review gate).

**Tradeoff, explicit:** this project has no distinct "operator" identity
from "whoever holds the shared `X-API-Key`" (ADR-0016) — restricting the
SSE payload doesn't yet ENFORCE an operator-vs-end-user boundary, it only
stops the CURRENT default channel (the `/ask` response) from leaking it.
An operator dashboard would need its own authenticated read path
(`graph.aget_state()` exposed some other way) to see the draft today — not
built here, named as the natural next step once this API has more than
one caller identity.

### SHOULD 3 — CLI exit codes

Each `resume`/`ask` failure mode now has its own exit code (see this
module's header table in `cli.py`): 7 (resume: unknown thread), 8 (resume:
not paused), 9 (ask: already paused) — previously 5 was reused for two
different `resume` conditions AND collided in spirit with `ask`'s own
`ToolCallError` code.

## How to reverse

- Drop the pause behavior entirely: `hitl_node` becomes an unconditional
  `return {}`, or remove the node and point `critic`'s edge straight back
  at `guard_out` (`build.py`) — no `GraphState` schema change either way,
  `hitl_node` adds no new state key of its own.
- Tune the threshold: `settings.critic_confidence_min`, one env var, no
  code change.
- Drop `/resume` entirely: delete the route and `_stream_resume`/
  `_require_paused_thread`; `/ask` still works unchanged (a paused run
  would just never be resumable via the API — the CLI's `resume` command
  would need the same removal to stay consistent).

## References

- Installed `langgraph/types.py`: `interrupt()`'s "resumes from the start
  of the node, re-executing all logic" docstring; `Command`'s
  `update`/`resume`/`goto` fields; `StateSnapshot`'s `.next`/`.interrupts`.
- Live-verified (this feature): a node's `Command(goto=...)` return
  overrides a static edge registered for the same node — no
  `add_conditional_edges` needed for `hitl -> guard_out`'s dynamic path.
- Live-verified: `stream_mode="updates"` emits a pause as
  `{"__interrupt__": (Interrupt(value=..., id=...),)}` — a distinct chunk,
  not folded into any node's own update.
- Round 2: `pydantic.json_schema.SkipJsonSchema` (pydantic 2.13, installed)
  — verified live that a field annotated with it is absent from
  `BaseModel.model_json_schema()`'s `properties`/`required` while remaining
  a real, settable field on the model (present in `.model_dump()`).
- Round 2, reproduced live before the fix: `graph.astream({"question":
  ...}, config=an-already-paused-thread-id)` starts a new run from `START`
  and overwrites the paused checkpoint — no error, no warning.
- ADR-0021 (`guard_out` as the final gate on every path, unchanged
  invariant) — ADR-0023 (the critic this feature reads
  `critic_confidence_min` against, and whose "Free-text leak guard" rule
  round 2's SHOULD 1 restores) — ADR-0024 (the Postgres checkpointer this
  feature depends on) — ADR-0019 (the classifier's fail-open reasoning
  round 2's BLOCKER 1 explicitly reuses) — ADR-0016 (the shared-API-key gap
  this feature's `/resume` inherits, not solves).
