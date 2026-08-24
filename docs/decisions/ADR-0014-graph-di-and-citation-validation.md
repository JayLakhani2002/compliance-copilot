# ADR-0014 — Graph dependency injection and citation validation

## Status
Accepted. 2026-08-24.

## Context
Day 6 builds the first compiled LangGraph graph (`retrieve` -> `answer`, `docs/ARCHITECTURE.md` §4). Two decisions came up building it, both consequential for later days:

1. `retrieve` and `answer` need a DB `Session`, an `Embeddings` provider, and an LLM client to do their work. None of these are serialisable, and ADR-0001 already commits this project to a Postgres checkpointer once human-in-the-loop (`interrupt()`) lands — so whatever holds these objects must never end up inside `GraphState`, or a future checkpointer would try to persist a live DB connection.
2. `answer` calls Claude for a structured `AnswerSchema` (citations included). The model can hallucinate a citation — an anchor that was never retrieved, a recital cited as if it were binding law, or a "verbatim" quote that isn't actually in the source text. Something has to decide what happens when that occurs.

## Decision 1: dependency injection via `context_schema`/`Runtime[GraphContext]`

### Options considered
1. **Closures / `functools.partial`** — wrap each node in a factory that closes over `session`/`embeddings`/`llm`.
2. **`config["configurable"]`** — pass dependencies through `RunnableConfig` at `.invoke(state, config=...)` time.
3. **`context_schema` / `Runtime[Context]`** (LangGraph 1.x) — declare a dataclass, pass it as `StateGraph(..., context_schema=GraphContext)`, and give node functions a second `runtime: Runtime[GraphContext]` parameter.

### Decision
Option 3. `GraphContext` (`src/compliance_copilot/graph/state.py`) holds `session`, `embeddings`, `llm`; `build_graph()` compiles with `context_schema=GraphContext`; `nodes.py`'s `retrieve_node`/`answer_node` read `runtime.context`.

### Why not the others
- **Closures**: work today (no checkpointer yet), but the graph object itself would need constructing fresh per request once a checkpointer is added, to avoid a stale closed-over `Session` living across requests. `Runtime[Context]` avoids that because context is supplied at `.invoke()` time, not at graph-construction time — the compiled graph can stay a single cached object (`build_graph()`'s `lru_cache`) forever.
- **`config["configurable"]`**: works, but it's a stringly-typed dict shared with LangGraph's own internal config keys — no static typing, easy to typo a key name with no error until runtime.
- `Runtime` is LangGraph 1.x's purpose-built mechanism for exactly this ("run-scoped data... run dependencies" per its own docstring, `langgraph/runtime.py`), and it keeps non-serialisable objects out of `GraphState` by construction — `context` is per-call, never checkpointed.

## Decision 2: citation validation — hard error, not a warning

### Options considered
1. **No check** — trust the model's structured output as-is.
2. **Warn-only** — log a warning but still return the answer.
3. **Hard error (`CitationError`)** — reject the entire answer if any citation fails validation.

### Decision
Option 3. `answer_node` (`src/compliance_copilot/graph/nodes.py`) checks, for every citation in the model's `AnswerSchema`: (a) its `(regulation, anchor)` pair exists among the *articles* `retrieve_node` actually fetched (recitals don't count — they're supporting context only, never a citation source, per ADR-0013), and (b) its `quote`, whitespace-collapsed and case-folded, is a substring of that article chunk's text under the same normalisation. Either failure raises `CitationError` — the caller (`cli.py`'s `ask` subcommand) prints `REFUSED: <message>` and exits non-zero rather than showing a half-validated answer.

### Why not the others
- **No check**: defeats the entire point of forcing structured output with citations — a compliance assistant that can silently cite a non-existent article is worse than one that visibly fails.
- **Warn-only**: this is a compliance tool; an unverifiable "citation" reaching a user who trusts it is the actual harm being guarded against (`docs/ARCHITECTURE.md` §4's "guard blocks, never swaps" principle). A log line nobody reads doesn't prevent that.
- Hard error is also the cheapest to build and the most honest about what Day 6 can and can't guarantee — see the cost below.

### Honest cost
A strict verbatim-quote substring check can reject a *correct* answer whose quote drifted slightly beyond whitespace/case (e.g. the model silently expands an abbreviation, or drops a trailing clause). That's a false rejection, not a false acceptance — the safer failure direction for a compliance tool, but a real cost: Day 6 has no retry loop, so a drifted quote today is a hard failure the user sees immediately, not a chance for the model to try again. Building a retry/critic loop that re-prompts the model on a `CitationError` instead of failing outright is explicitly Week 3's job (`docs/ARCHITECTURE.md` §4's `critic` node), not this ADR's.

## Security & cost implications
- **Security:** `CitationError`'s message lists only the offending citation and the allowed anchors — never the user's question — so logging or displaying it can't leak question content (a citation-shaped message is safe to show an operator or a user). The question and retrieved article/recital text still leave the trust boundary to Anthropic on every `answer_node` call, same as ADR-0002 already covers; PII redaction is a pre-graph concern (`guard_in`, not built yet) and out of scope here.
- **Cost:** no new cost driver — validation is local string comparison, not an extra LLM call. A rejected answer today means the user sees a refusal and the question has to be re-asked (no retry), which is a UX cost, not a spend cost.

## How to reverse
- Decision 1: swapping DI mechanisms is a change to `state.py`'s `GraphContext` + `build.py`'s `context_schema=` argument + each node's second parameter — the node bodies' actual logic doesn't change.
- Decision 2: the two checks live entirely inside `answer_node` (`src/compliance_copilot/graph/nodes.py`) — softening to a warning, or adding a retry loop, is a change to that one function, not a schema or graph-shape change.

## References
- LangGraph `Runtime[Context]` / `context_schema`, verified against Context7 `/langchain-ai/langgraph` source `libs/langgraph/tests/test_runtime.py::test_injected_runtime` and `libs/langgraph/langgraph/runtime.py`.
- `StateGraph(state_schema=..., context_schema=...)` constructor signature — confirmed directly from the installed package, `.venv/lib/python3.12/site-packages/langgraph/graph/state.py`.
- `ChatAnthropic.with_structured_output`'s `method` parameter and default (`"function_calling"`, not `"json_schema"`) — confirmed directly from the installed package, `.venv/lib/python3.12/site-packages/langchain_anthropic/chat_models.py`.
- ADR-0013 (retrieval strategy: articles-first) — the "articles are the only citation source" rule this ADR enforces mechanically.
- ADR-0001 (agent framework: LangGraph) — the checkpointer/`interrupt()` context that makes keeping non-serialisable objects out of `GraphState` matter now, not just in theory.
