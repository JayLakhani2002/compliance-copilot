# ADR-0024 — Durable state: Postgres checkpointer, thread_id, conversation history

**Status:** accepted 2026-08-29

## Context

Every `/ask` today runs the graph once and forgets: `guard_in ->
router/refuse -> retrieve -> answer -> critic -> guard_out` executes,
streams a `final` event, and the process holds nothing about that question
a second later (`graph/build.py`'s `build_graph()` compiled with no
checkpointer). Three things this project needs a durable state store for,
in the order they start mattering:

1. **Follow-up questions.** "and what about deployers?" only makes sense if
   the graph can see the prior turn's question and answer.
2. **Surviving a process restart.** If the API container restarts between
   turns, the conversation is lost today.
3. **Day 20's human-in-the-loop pause.** `interrupt()`/`Command(resume=...)`
   requires a checkpointer at all — ADR-0001 already verified LangGraph
   raises `RuntimeError: Cannot use Command(resume=...) without
   checkpointer` — so this is a prerequisite for that feature, not a
   nice-to-have alongside it.

Since ADR-0001 was written, `retrieve_node` became `async def` and the
CLI/evals call `graph.ainvoke`/`graph.astream` exclusively (Day 17's MCP
amendment) — there is no working synchronous entrypoint left once a run
reaches an async node. `router`/`critic` nodes and their `GraphState` keys
also landed (Day 18/ADR-0023), which matters for the per-turn reset (below).

## Options considered

1. **`InMemorySaver`** (`langgraph.checkpoint.memory`) — zero setup, but
   process-local: a restart loses every conversation, and it doesn't share
   state across replicas. Right tool for unit tests, wrong tool for the app.
2. **A SQLite-backed saver** — file-based durability with no extra service,
   but this project already runs Postgres for documents/chunks (ADR-0003)
   and Langfuse itself needs its own stack (ADR-0009) — a third storage
   engine for checkpoints alone would be a second thing to back up, monitor,
   and reason about for no real benefit over the database already running.
3. **`langgraph-checkpoint-postgres`'s `PostgresSaver`/`AsyncPostgresSaver`**
   (this ADR) — reuses the existing Postgres instance (ADR-0003/ADR-0010),
   the mechanism ADR-0001 already named as the target.
4. **No persistence** — leaves follow-up questions, restart-survival, and
   Day 20's HITL all unbuilt. Rejected: this is exactly the gap ADR-0001
   flagged as the reason LangGraph was chosen over a plain agent loop.

## Decision

**`AsyncPostgresSaver` everywhere the app runs the graph — the API and the
CLI both** (a deviation from ADR-0001's original hedge of "PostgresSaver /
AsyncPostgresSaver" and from the researcher brief's earlier "sync for the
CLI" draft): verified in the installed `langgraph/checkpoint/base/
__init__.py` that `BaseCheckpointSaver`'s async methods (`aget_tuple`,
`aput`, `aput_writes`) all raise `NotImplementedError` unless a subclass
overrides them, and the sync `PostgresSaver` doesn't. Since `retrieve_node`
is `async def` and every caller now drives the graph via `ainvoke`/
`astream`, the sync saver would break the moment a real run reached
`retrieve_node` — there is no code path left that needs the sync saver, so
using it anywhere would just be a second saver implementation to maintain
for zero benefit.

**Packages** (verified installed, matches ADR-0001's earlier PyPI check):
`langgraph-checkpoint-postgres` 3.1.2, `psycopg-pool` 3.3.1 (`uv add
langgraph-checkpoint-postgres psycopg-pool`) — the checkpointer's own
`AsyncConnectionPool` comes from `psycopg-pool`, built by the *caller*, not
the saver (verified against the installed `langgraph/checkpoint/postgres/
aio.py`'s `AsyncPostgresSaver.__init__(conn: AsyncConnection |
AsyncConnectionPool, ...)` — no pool kwargs live on the saver itself).

**`src/compliance_copilot/checkpointer.py`** (new, small, its own module
rather than folded into `db.py`): `db.py` is the sync SQLAlchemy engine used
by ingestion/retrieval; this is a different driver stack (async psycopg
pool) for a different job. `build_checkpointer()` is an async context
manager: builds `_checkpointer_dsn()` (the same `make_url(...).set(
drivername="postgresql")` URL surgery `tests/conftest.py` already uses,
since psycopg's own conninfo parser chokes on SQLAlchemy's `+psycopg`
driver suffix — verified live), opens an `AsyncConnectionPool(dsn,
kwargs={"autocommit": True, "row_factory": dict_row}, open=False)` +
`await pool.open()` (both required by the saver's own SQL, Context7-verified
against `libs/checkpoint-postgres/tests/test_async.py`'s fixture), wraps it
in `AsyncPostgresSaver`, runs `await saver.setup()` (idempotent — creates
checkpoint tables/runs migrations only if missing, per the installed
package's own docstring), yields it, and closes the pool on the way out.
Also holds `validate_thread_id()` — one place both `api.py`'s
`AskRequest` field validator and `cli.py`'s `ask`/`delete-thread` argument
parsing call, so the UUID4 rule is enforced identically everywhere.

**`build_graph(checkpointer=None)`** (`graph/build.py`): the compiled graph
now takes an optional checkpointer, forwarded to `.compile(checkpointer=
...)`. `None` (unchanged default) is today's stateless-per-call behaviour —
every existing caller (evals, most of `tests/test_graph.py`) is unaffected.
`lru_cache(maxsize=1)` still wraps it — keyed on the checkpointer argument
(identity-hashed, the default for a plain object), so a genuinely different
saver instance is correctly a cache miss, not a stale reused graph.

**API wiring** (`api.py`): `lifespan()` opens one `build_checkpointer()` for
the process's whole lifetime, stores it on `app.state.checkpointer`, and
closes it on shutdown. `get_checkpointer_dependency(request)` reads it off
`app.state` (falling back to `None` when `lifespan` hasn't run — e.g. a bare
`TestClient(app)` used without its `with` context manager, this project's
existing `tests/test_api.py` pattern) — same `app.dependency_overrides`
shape every other dependency here already has, so a test can swap in an
`InMemorySaver()` fixture without touching Postgres. `/ask`'s `AskRequest`
gains `thread_id: str | None`; absent means the route mints
`uuid.uuid4()`, present must pass `validate_thread_id` (422 otherwise).
`_stream_answer` yields a `thread` SSE event — `{"thread_id": ...}` — as the
literal FIRST event of every response, before the graph even starts (the
caller already knows the id by then), so a client that omitted `thread_id`
learns the server-issued one in time to send it back on the next call.
`config["configurable"] = {"thread_id": ...}` rides alongside the existing
Langfuse `session_id`.

**CLI wiring** (`cli.py`): `ask` gains `--thread-id` (same `validate_
thread_id` via an `argparse` `type=` callable) — absent mints a fresh one,
printed to stderr immediately (`thread: <uuid>`) so a second invocation can
continue the conversation; `build_checkpointer()` opens/closes around this
one command's one call, since a CLI invocation is a fresh OS process every
time (no long-lived pool to hold open the way the API's `lifespan` does). A
new `delete-thread <uuid>` command calls `checkpointer.adelete_thread(...)`.

**History** (`graph/state.py`, `graph/nodes.py`): `GraphState.history:
NotRequired[list[Turn]]` — deliberately a **plain field, not `Annotated[
list[Turn], operator.add]`** (a real deviation from the researcher brief,
caught by writing the durability test before trusting the design — see
"Why not `operator.add`" below). `Turn` is a small frozen dataclass
(`question`, `answer`) — round-trips through the checkpoint serde the same
generic way every other dataclass in this state already does (`GuardResult`,
`OutputVerdict`, ...). `guard_out_node` is the ONLY node that ever writes
`history`, on every path that returns rather than raises: it reads
`state.get("history", [])`, appends this turn, and returns the full
replacement list already capped to the last 3 turns (`_capped_history`,
`_HISTORY_MAX_TURNS`). `answer_node`'s `_build_messages` renders it as prior
`("human", question), ("ai", answer)` pairs, placed after the system prompt
and before the current turn's freshly-retrieved excerpts+question — the
system prompt never changes (stable prefix), history changes turn-to-turn
but is still more stable than this turn's excerpts, which change on every
single call regardless (ADR-0002/ADR-0007's prompt-caching note).

**Per-turn reset** (`graph/nodes.py`'s `_PER_TURN_RESET`, merged into every
one of `guard_in_node`'s return dicts, since it's the one node every turn
always visits first): `answer`, `refused`, `draft`, `citation_error`,
`attempts`, `output_guard`, `router`, `critic` are all explicitly reset
(`None`/`0`/`False`) at the start of every turn. State does not auto-reset
between checkpointed turns — a checkpoint is a snapshot of *everything* — so
without this, turn N's leftover values leak into turn N+1's routing
decisions.

## Why not `operator.add` (a deviation from the research, found by testing)

The researcher brief's §3 called for `Annotated[list[Turn], operator.add]`
plus "return an already-capped replacement list" from `guard_out_node`.
Reading the installed `langgraph/channels/binop.py`
(`BinaryOperatorAggregate.update`) shows why that combination is actually
wrong: a reducer-backed channel computes `self.value = self.operator(
self.value, returned_value)` — it always combines the OLD channel value
(the prior turn's checkpointed history) with whatever a node returns, it
never replaces. Returning an already-capped **full** list under `operator.
add` would concatenate it onto the existing value every turn instead of
replacing it — turn 2 would end up with turn 1's list *plus* the "capped"
list, silently growing past the cap. Since `guard_out_node` is the ONLY
writer of `history`, on any path, in any run, a reducer earns nothing here:
LangGraph's default merge for a plain key ("last write wins") already gives
exactly the needed semantics — replace with a smaller list. This was caught
by writing `test_history_capped_at_last_three_turns` before trusting the
design, not by inspection alone.

## Security & cost implications

- **`thread_id` is server-issued by default** (`uuid.uuid4()`), and a
  client-supplied one must pass `validate_thread_id` (syntactically a
  UUIDv4) — a guessable/sequential id would let a caller resume or read
  another session's checkpointed conversation. **This does NOT close the
  separate, still-open ADR-0016 gap**: this API has one shared `X-API-Key`
  across every caller, so any key holder can still supply any
  validly-*shaped* `thread_id` (including one they never issued themselves,
  if they can guess or observe it) and resume/read that thread. Per-caller
  thread ownership would need per-caller identity (ADR-0016's own "revisit
  if multiple distinct client identities" note) — out of scope for this
  feature, named here as a known limitation, not solved.
- **Checkpoints persist the (already `guard_in`-redacted) question,
  retrieved excerpts, and answers into Postgres** — same trust-boundary
  class ADR-0001 already flagged ("checkpointed state should be treated
  with the same sensitivity as application logs"). No new redaction work:
  `guard_in` still runs first on every path, before anything that reaches
  checkpointed state.
- **Erasure path**: `cli.py`'s `delete-thread <uuid>` calls `adelete_thread`
  — a real, minimal GDPR-flavoured "forget this conversation" command,
  from day one rather than bolted on later (echoing lesson 13's point about
  the documents table). `# ponytail:` this is a manual, one-thread-at-a-time
  command — no scheduled retention job, no bulk "delete everything older
  than N days" sweep. Add a retention job the day checkpoint table growth
  or a real deletion-request volume actually shows up; today's traffic is
  the author's own testing.
- **EU residency unchanged**: checkpoints land in the same Hetzner Postgres
  instance (ADR-0003/ADR-0010) already hosting documents/chunks — no new
  provider, no new residency question.
- **Cost**: `AsyncPostgresSaver` adds no per-call spend of its own (a local
  library against an already-running Postgres) — a few KB of serialized
  state per node, per turn, accumulating per thread. No index/VACUUM/
  retention tuning done yet — an operational follow-up ADR-0001 already
  named as unsolved by LangGraph itself, unchanged by this feature.
- **Startup now depends on Postgres being reachable**: `api.py`'s
  `lifespan` opens the checkpointer pool and calls `.setup()` before the
  app starts serving — a Postgres outage at startup means the app never
  comes up, rather than starting in some degraded checkpointer-less mode.
  Deliberate: fail loudly, before any LLM spend is even possible, matching
  the project's existing "DB down" failure posture (`docs/ARCHITECTURE.md`
  §7) rather than a new, quieter failure mode.

## How to reverse

- Drop the checkpointer entirely: `build_graph()`/`ask()` already default
  `checkpointer=None` — omit the argument everywhere it's currently passed
  (api.py's `lifespan`/`_stream_answer`, cli.py's `_run_ask`) and delete
  `checkpointer.py`. `GraphState.history` and the per-turn reset are inert
  without a checkpointer (there is no "prior turn" to reset from or append
  to in a stateless-per-call run) — safe to leave in place or delete either
  way, no other code depends on them.
- Swap Postgres for a different saver: only `checkpointer.py`'s
  `build_checkpointer()` needs to change (a different connection/setup
  shape) — `build_graph(checkpointer=...)`, `GraphState.history`, and the
  per-turn reset are saver-agnostic (`InMemorySaver` already proves this in
  `tests/test_graph.py`'s unit tests).
- Drop `thread_id` from the API: revert `AskRequest`, the `thread` SSE
  event, and `get_checkpointer_dependency`'s wiring in `api.py` — the graph
  itself doesn't require a `thread_id` to run (a `None`/absent one, or a
  checkpointer-less graph, is exactly today's pre-ADR-0024 behaviour).

## References

- `langgraph-checkpoint-postgres` 3.1.2, `psycopg-pool` 3.3.1 — installed
  versions, verified via `uv add` (2026-08-29), matching ADR-0001's earlier
  PyPI check.
- `AsyncPostgresSaver.__init__`, `.setup()`, `.adelete_thread()`: installed
  `.venv/lib/python3.12/site-packages/langgraph/checkpoint/postgres/aio.py`.
- `BinaryOperatorAggregate.update` (why `operator.add` would double-count a
  full-replacement return): installed `.venv/lib/python3.12/site-packages/
  langgraph/channels/binop.py`.
- `BaseCheckpointSaver`'s async methods raising `NotImplementedError` by
  default (why `AsyncPostgresSaver`, not the sync `PostgresSaver`, for
  every async caller): installed `.venv/lib/python3.12/site-packages/
  langgraph/checkpoint/base/__init__.py`.
- Pool/saver construction pattern (`AsyncConnectionPool(..., kwargs={
  "autocommit": True, "row_factory": dict_row})`, then `AsyncPostgresSaver`,
  then `await checkpointer.setup()`): Context7 `/langchain-ai/langgraph`,
  `libs/checkpoint-postgres/tests/test_async.py`.
- ADR-0001 (original checkpointer decision + `interrupt()`/`Command(
  resume=...)` requiring one), ADR-0016 (the shared-API-key thread gap this
  ADR names but doesn't close), ADR-0021 (`guard_out` as the one place every
  terminal path funnels through, which is why it's the one place `history`
  is written).
