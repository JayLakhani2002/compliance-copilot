# ADR-0023 — Router and critic: two cheap-LLM nodes, records-before-blocks

**Status:** accepted 2026-08-29

## Context

`art_3` is a real anchor in BOTH the AI Act and GDPR — each regulation
numbers its own Article 3. `retrieve_node` (graph/nodes.py, ADR-0007's
Day-17 amendment) always calls `search_regulation` with `regulation=None`
(search both laws), even though that tool already accepts a `regulation`
filter (`"ai_act" | "gdpr" | None`, ADR-0013) that nothing populates. A
cross-regulation collision is possible whenever a question is clearly about
one law but the retriever searches both, diluting the top-k with irrelevant
competing candidates from the other regulation.

Separately, `guard_out` (ADR-0021) checks *shape* — is there a citation,
does the schema validate, is there a scaffold/canary/PII leak — but nothing
today checks *substance*: does the cited quote actually SUPPORT the claim
being made, in the judgment sense `evals/judge.py`'s offline `JudgeVerdict`
already applies at eval time, per CI run. There is no per-request sibling of
that check.

## Options considered

1. **No router** — `retrieve_node` keeps searching both regulations always.
   Leaves the measured `art_3` collision risk open indefinitely; a cheap,
   scoped fix is left on the table.
2. **Regex/keyword router** — cheaper than an LLM call, but "Datenschutz,"
   "data protection," and "personenbezogene Daten" all mean GDPR-scope and
   share no substring a regex could anchor on; a question can imply scope
   without ever naming the regulation (same argument ADR-0019 already made
   for the classifier's paraphrase/multilingual gap).
3. **LLM router + LLM critic, both plain `StateGraph` nodes** (this ADR) —
   reuses the classifier guard's whole playbook (ADR-0019): cheap tier,
   `.with_structured_output`, temperature=0, short timeout, explicit
   fail-policy. The router's label is advice to a deterministic node, never
   agency — `retrieve_node` still runs the same fixed two-call retrieval it
   runs today, only with one more filter argument. The critic only records a
   verdict; it does not branch on it yet.
4. **Specialist subgraphs** (a GDPR-only pipeline, an AI-Act-only pipeline)
   — rejected: both would share the same system prompt, differing only in
   which excerpts get retrieved. That's one node with a filter, not two
   agents (lesson 18's own "don't go multi-agent" case).

## Decision

Option 3.

### Router (`src/compliance_copilot/router.py`)

- `RouterVerdict(regulation: Literal["ai_act","gdpr","both","out_of_scope"],
  reason: str, max_length=300)` — forced via
  `.with_structured_output(RouterVerdict, method="json_schema")`, the same
  pattern already proven for `Verdict`/`AnswerSchema`. No free-text field the
  model can hide an injected instruction in beyond the capped `reason`.
- `ROUTER_PROMPT`: minimal system prompt (classify only) + 4 short examples
  (one per label, including the cross-regulation case) — same "anchor the
  boundary without a growing example list" reasoning as `CLASSIFIER_PROMPT`.
- `make_router_llm()`: mirrors `make_classifier_llm()` exactly — same
  per-provider cheap-tier defaults (`gpt-4.1-nano` / `claude-haiku-4-5`),
  `settings.router_model` override, `settings.router_timeout_s` (default
  3.0s), `max_retries=0`.
- `route(text, llm) -> RouterVerdict | None`: `None` on ANY exception,
  logging only the exception class name — same fail-open shape as
  `classify()`.
- Graph wiring (`build.py`): `guard_in`'s clean branch now goes to `router`
  instead of `retrieve`; `route_after_router` sends `out_of_scope` straight
  to `refuse` (no retrieval/answer spend on a question about neither
  regulation), everything else (including a `None`/disabled router,
  normalised by `router_node` to a `both`-shaped verdict) to `retrieve`.
- `retrieve_node` reads `state.get("router")`: `"both"` or absent → no
  filter (`regulation=None`, today's behaviour); `"ai_act"`/`"gdpr"` →
  that filter, passed straight into `search_regulation`'s existing
  `regulation` parameter.
- `GraphContext.router: Any | None = None` (new field) — `None` disables the
  router entirely (`settings.router_enabled=False`); `router_node` then
  no-ops (`return {}`), leaving `state["router"]` absent, which
  `retrieve_node`/`route_after_router` both already treat as "no filter,
  never `out_of_scope`."

**Fail-open target: `both`, deliberately the OPPOSITE axis from the
classifier's fail-open.** `classify()` failing open means "allow" (don't
block the product on an outage). `route()` failing open means "search both
regulations" — `router_node` maps a `None` verdict (or a disabled router) to
the equivalent of `regulation=None`. Failing to `out_of_scope` on an outage
would turn a router BUG into a false REFUSAL — fail-CLOSED on availability,
the wrong direction per ADR-0019's own asymmetric-harm argument (a wrong
scope guess costs a slightly worse top-5; a false refusal costs the whole
answer). Failing to `ai_act`/`gdpr` arbitrarily risks a false narrow miss
with no basis to prefer one law over the other. `both`/`None` is the only
choice that can't make things worse than pre-router behaviour.

### Critic (`src/compliance_copilot/critic.py`)

- `CriticVerdict(faithful: bool, confidence: float [0,1], reasoning: str,
  max_length=300)` — a NEW schema, structurally similar to (but not
  importing) `evals.judge.JudgeVerdict`: `evals/` is a dev/CI-only tree
  (`pyproject.toml`'s `packages = ["src/compliance_copilot"]`), so importing
  it from shipped code would be a real layering violation. Deliberately
  narrower than the offline judge — no `relevant` field: `guard_out`'s
  `scope_unsupported` heuristic already gives a cheap, deterministic proxy
  for "did this even try to relevance-check," and ADR-0017 itself declined
  to gate on `relevant` even offline ("more phrasing-sensitive than the
  factual-grounding property this gate exists to guarantee") — the same
  reasoning applies with even less budget online. Adding `relevant` is a
  clean future extension, not this feature's scope.
- `CRITIC_SYSTEM_PROMPT` reuses `evals.judge.JUDGE_SYSTEM_PROMPT`'s wording
  almost verbatim (same rubric, adapted for `confidence` instead of
  `relevant`) — the critic checks the gap NEITHER `_validate_citations`
  (ADR-0014, verbatim-quote-exists) NOR `guard_out`'s `citation_not_retrieved`
  re-check (ADR-0021) can see: does the drafted answer's PROSE claim
  actually follow from what the quoted text SAYS (semantic support), not
  just whether the quote string is present.
- `make_critic_llm()`: same cheap-tier construction pattern as
  `make_router_llm()`/`make_classifier_llm()`.
- `critique(question, answer, contexts, llm) -> CriticVerdict`: NEVER raises
  past this function — on any exception, still returns a verdict, a
  PESSIMISTIC one (`faithful=False, confidence=0.0,
  reasoning="critic_error:<ExceptionClassName>"`), logged the same
  "exception class name only" way every guard already follows.
- `critic_node` (graph/nodes.py): runs only on `answer`'s success branch,
  between `answer` and `guard_out` — `route_after_answer`'s success case now
  returns `"critic"` (was `"guard_out"`); a single unconditional
  `add_edge("critic", "guard_out")` keeps `guard_out` the final gate on
  EVERY path (ADR-0021's invariant is unchanged, just one hop later on this
  one branch). Context is built from `state["articles"]` (already in
  memory, no DB round-trip) and CITED-only (same `(regulation, anchor)`
  keying `_validate_citations` uses) — mirrors `evals/run_answer_eval.py`'s
  own `_contexts_for` cited-only contract, keeping the online critic
  comparable to the offline judge on the same questions.
- `GraphContext.critic: Any | None = None` — same disable contract as
  `router` above (`settings.critic_enabled=False`).
- `settings.critic_confidence_min: float = 0.6` — a Day-20 placeholder, NOT
  read anywhere yet. Kept (not dropped as YAGNI) because Day 20 wiring a new
  conditional edge / `interrupt()` off this threshold needs no
  settings-schema change if the knob already exists; the alternative (adding
  it later) is strictly more churn for zero cost saved today.

**Fail policy is the OPPOSITE bias from the router/classifier's fail-open.**
Nothing branches on the critic's output yet (Day 20's job), so there is no
availability risk an outage could threaten. But there IS a calibration risk:
a silent gap (no score at all) or a falsely reassuring `faithful=True` would
corrupt Day 20's threshold-tuning work, which lesson 18 explicitly wants
done "on evidence, not intuition." So the critic fails PESSIMISTIC, not
open — a low, honest signal an operator can filter on, never a silently
missing one or a falsely confident one.

**Langfuse scores** (`api.py`'s `_stream_answer`, only when tracing is on):
`critic_faithful` (1.0/0.0) and `critic_confidence` (0-1) as two separate
named scores, matching the existing one-signal-per-`tracing.score()`-call
convention (`citation_valid`, `refused`, `output_blocked` are each their own
call, never combined).

### Free-text leak guard (both nodes)

`RouterVerdict.reason` and `CriticVerdict.reasoning` are short strings the
MODEL writes — free text, not a fixed enum, distinct from
`Verdict.category`'s closed literal set. Both are capped (`max_length=300`)
and treated the same as every other guard's "reasons only, never raw text"
rule: logged and traced, but the SSE stream (`api.py`) only ever forwards
`RouterVerdict.regulation` and `CriticVerdict.{faithful,confidence}` — never
`.reason`/`.reasoning` — so a model that ignores instructions and echoes a
fragment of the question into its own justification can leak into logs/
traces at worst, never into a response the end user sees.

## Why not the others

- **No router (option 1)** leaves a real, named, currently-unaddressed
  collision risk (`art_3`) with a known, cheap fix already sitting one field
  away (`search_regulation`'s existing `regulation` parameter).
- **Regex/keyword router (option 2)** cannot see meaning across languages or
  implicit scope — the identical argument ADR-0019 already proved
  empirically for the classifier's own paraphrase/multilingual gap.
- **Specialist subgraphs (option 4)** would duplicate the answer prompt for
  no reason — a filter on one retrieval call already captures the entire
  difference between an "AI-Act-only" and "GDPR-only" answer path.

## Security & cost

- Two more cheap-tier (`gpt-4.1-nano`/`claude-haiku-4-5`) calls per
  question — router always runs on a clean, non-flagged question (before
  retrieval); critic only runs on a genuinely validated answer (never on a
  refusal). At $0.10/$0.40 per MTok in/out and small prompts (router:
  ~200-400 in / ~30 out tokens; critic: ~800-1500 in / ~60 out tokens),
  added cost per answered question is well under $0.001 — consistent with
  ARCHITECTURE.md §9's existing cost ballpark, which already budgeted for
  this pair.
- Both calls see only the guard_in-cleaned, already-PII-redacted question
  (router) or that plus the drafted answer and its own cited excerpts
  (critic) — never raw PII, never the whole corpus. Same injection-surface
  mitigation as every guard so far: structured output only, no tools, no
  free-text field an injected instruction could land executable content in.
- Labels/scores only reach logs and traces, never the question or answer
  body — consistent with every prior guard node.

## How to reverse

- `ROUTER_ENABLED=false` / `CRITIC_ENABLED=false` — one env var each, no
  code change. `get_router_dependency()`/`get_critic_dependency()` (api.py)
  then return `None`, `router_node`/`critic_node` no-op, and the graph shape
  is unchanged (both nodes still exist, they just pass state through
  unmodified) — no edge changes required to turn either off.
- Fully removing either node from the graph (not just disabling it) is a
  `build.py`-only change: drop the node registration, point
  `route_after_guard`'s clean branch back at `"retrieve"` (router) or make
  `route_after_answer`'s success case return `"guard_out"` again (critic) —
  no `GraphState`/schema change required either way.

## References

- ADR-0013 (retrieval strategy) — `search_regulation`'s existing
  `regulation` filter, unused until this feature.
- ADR-0014 (graph DI, citation validation) — the `(regulation, anchor)`
  keying and cited-only context-building pattern the critic reuses.
- ADR-0019 (classifier guard) — the whole node-construction/fail-policy
  playbook both new nodes mirror.
- ADR-0021 (output guard) — the "`guard_out` is the final gate on every
  path" invariant, preserved with the critic inserted one hop earlier.
- `evals/judge.py` / `evals/run_answer_eval.py` — the offline judge and its
  cited-only context-building convention the critic's rubric/context
  mirrors without importing.
