# ADR-0031 — Fuzzy quote matching for citation validation

**Status:** accepted 2026-09-02

## Context

ADR-0014's `_validate_citations` requires a citation's `quote` to appear as
an exact substring of its cited excerpt after `_normalise` (whitespace
collapse, case fold, curly-quote fold). That check is deliberately strict
(ADR-0014: "guard blocks, never swaps") — but it has no tolerance for
cosmetic drift a real model reproduces even when the underlying quote is
genuine: a punctuation variant, a collapsed "...", a spaced-out hyphen, a
dropped trailing marker. Measured baseline (ADR-0022, 2026-08-26 red-team
run): **6/20 benign legal questions and golden item c01** failed with
`CitationError` despite being genuinely on-topic, correctly-cited answers —
tracked separately as `benign_citation_error_rate` precisely so this
guardrail-adjacent failure mode wouldn't inflate the red-team FPR number for
an unrelated reason.

## Options considered

1. **Status quo (verbatim-only)** — simplest, but leaves the measured 30%
   benign failure rate on the table; a compliance tool that refuses a
   correct, well-cited answer because of a stray semicolon is a real UX and
   trust cost, not just a metric.
2. **Sentence-index citations** — have the model cite `(anchor, sentence_n)`
   instead of a free-text quote, sidestepping the matching problem
   entirely. Rejected: a bigger schema/prompt change than this problem
   needs, and it trades one failure mode (quote drift) for another (the
   model miscounting sentence indices in a multi-sentence excerpt) with no
   existing measurement of which is worse.
3. **LLM re-judge** (a cheap model asked "is this quote faithful to the
   excerpt?") — real accuracy, but an LLM call on every citation check is
   the exact cost/latency `_validate_citations` was built to avoid (it's
   deterministic string logic today, ADR-0014). `critic_node` (ADR-0023)
   already does semantic-support judging post-hoc; duplicating that
   per-citation, pre-hoc, would double LLM spend for overlapping coverage.
4. **`rapidfuzz` (or another fuzzy-matching dependency)** — faster than
   stdlib `difflib` at scale, but citations here are 1-5 per answer against
   article-sized (a few KB) excerpts — nowhere near the volume that speed
   difference would matter, and it's a new dependency for a job stdlib
   already does.
5. **stdlib `difflib.SequenceMatcher` fallback with a high floor**, tried
   only after the exact check misses (unchanged fast path on a clean hit).

## Decision

Option 5. `guards/quotes.py`'s new `quote_matches(quote, excerpt) ->
QuoteMatch` is the ONE function both `_validate_citations` (graph/nodes.py)
and the MCP `cite` tool (mcp_server.py) call — same "one function, two
callers can't drift" contract `_normalise`/`_MIN_QUOTE_LENGTH` already
established when this module was split out of `graph.nodes` (ADR-0007's
Day-17 amendment).

**Fast path (unchanged):** `_normalise`d `quote` is a substring of
`_normalise`d `excerpt` -> match, `score=None`, zero fuzzy work.

**Fuzzy fallback (new, only on an exact miss) — TWO conditions, both must
hold** (round 2 revision; round 1 shipped only the first, see "Round 2"
below for why that wasn't enough):

1. **Similarity floor.** Slides a quote-length window across every offset
   of the excerpt (`_best_fuzzy_window`), scoring each with
   `difflib.SequenceMatcher.ratio()` (one matcher built once, `set_seq2`
   reused per window — `difflib`'s own documented pattern for scanning many
   candidates against one fixed sequence) — comparing the quote against the
   WHOLE excerpt in one `ratio()` call would always score low regardless of
   match quality, since `SequenceMatcher` scores by combined length;
   windowing at quote-length keeps quote and candidate apples-to-apples.
   The best score must be `>= settings.quote_similarity_min`.
2. **No added words.** Every content token (word/number run) in the quote
   must appear, in the same order, somewhere in that SAME winning window
   (`_is_ordered_subsequence`) — omissions are free (that's the cosmetic-
   drift tolerance this function exists for), but a token the window
   doesn't have at all, or only has behind where the check has already
   advanced, fails it outright, regardless of the ratio score.

**Tuned floor: 0.92.** A citation that passes only via the fuzzy path is
logged as a `quote_fuzzy_match` guardrail event (`anchor`, `score` — never
the quote text, same rule every guardrail event in `nodes.py` already
follows) so Langfuse shows how often the fallback fires.

### Also decided: a prompt fix, not a lower floor, for c01

Reproducing golden item c01 live (`gpt-4.1-mini`) surfaced a SECOND,
different failure shape the floor search below was built to distinguish
from cosmetic drift: the model spliced multiple non-contiguous list items
(`(a)...(d)...(e)`, silently dropping `(b)`/`(c)`) into ONE citation's
quote, and separately produced a quote eliding roughly a third of a long
sentence behind a single `...`. Both score well below any floor that also
rejects the adversarial fixtures below (0.50 and 0.28 respectively) —
correctly so: a large omission is textually indistinguishable from a
malicious splice, and a floor loose enough to accept it would also accept
the "spliced half-sentences" adversarial fixture. Lowering the floor to fix
c01 would have reopened exactly the hole ADR-0022's ASR gate exists to
catch.

The actual fix was `SYSTEM_PROMPT` (graph/nodes.py): an added instruction
that each quote must be ONE CONTIGUOUS span — no `...`, no joining list
items with a citation's own punctuation — and to add a SEPARATE citation
per point instead. Live re-run after this prompt change: the model split
its Art. 14 discussion into five short, contiguous per-item citations
(each an exact or near-exact match) and self-corrected its one remaining
over-long ellipsis quote on the existing retry-once pass (ADR-0015) — c01
now succeeds with `quote_similarity_min` unchanged at 0.92, not raised.
This is the right layer for that fix: `quote_matches`'s job is tolerating
drift in an otherwise-faithful quote, not reconstructing what the model
meant to quote.

### Floor search — measured, not asserted

Fixtures built directly against a realistic article-shaped excerpt
(`tests/test_guards_quotes.py::EXCERPT`, numbered sub-clause + `(1)`
marker + a compound sentence — chosen so a naive whole-excerpt
`ratio()` would already read low, the failure mode `_best_fuzzy_window`'s
windowing exists to avoid):

| Fixture | Type | Score |
|---|---|---|
| punctuation swap (comma -> semicolon) | drift, must pass | 0.971 |
| collapsed ellipsis (short elision) | drift, must pass | 0.976 |
| spaced-out hyphen ("high - risk") | drift, must pass | 0.978 |
| dropped trailing "(1)" marker | drift, must pass | 0.964 |
| right words, inverted meaning | adversarial, must fail | 0.693 |
| spliced half-sentences (two excerpts) | adversarial, must fail | 0.589 |
| quote from a different article | adversarial, must fail | 0.402 |
| fully fabricated quote | adversarial, must fail | 0.351 |

These fixtures only OMIT text relative to the excerpt (a punctuation swap,
a shortened elision, a dropped marker) — none of them ADD a word the
excerpt doesn't have. `tests/test_guards_quotes.py::
test_boundary_at_exact_threshold_is_inclusive` pins the `>=` (not `>`)
boundary behaviour directly, independent of any specific fixture's score.

**What this table does NOT prove, and round 1 wrongly claimed it did:**
round 1 read the 0.964-vs-0.693 gap above as "more than 0.04 of headroom
below every real drift case" and treated that as evidence the floor was
safe against adversarial input generally. It is not — every fixture above
was hand-built by the same person who set the floor, all of them either
purely omit text or replace a large-enough span that the ratio drops far
below the floor; none of them tested the specific shape "genuine quote,
plus one or a few ADDED words" — which is exactly the shape a `difflib`
ratio is worst at catching (a 3-15 character insertion into an
80-260 character quote is a small edit-distance fraction no matter what the
inserted words say). See "Round 2" below for what an adversarial prober
who wasn't the same person who tuned the floor actually found.

### Round 2 — reviewer found two floor-integrity holes and a window-scan bug

Round 1 review probed `quote_matches` directly (not the fixtures above) and
found three reproducible holes, all confirmed against the checked-in
round-1 code:

| Probe | Round-1 score | Round-1 verdict | Round-2 verdict |
|---|---|---|---|
| Negation flip: insert one "not" into an otherwise-genuine, 80+ char quote | 0.957–0.968 | **accepted (hole)** | **rejected** — "not" is not a token in the winning window |
| Fabricated clause appended to a genuine short quote (e.g. ", within 90 days") | 0.961–0.963 | **accepted (hole)**, up to ~8% of excerpt length | **rejected** — "within"/"90"/"days" aren't in the window |
| Fabricated clause prepended ("Notwithstanding any other provision, ") | 0.921 | **accepted (hole)** | **rejected** — "notwithstanding" etc. aren't in the window |
| Genuine drift quote (`dropped_marker` fixture), shifted 1-7 filler characters from its round-1-measured offset | 0.857–0.964 depending on offset | **inconsistent** (4/8 offsets flipped a genuine match to reject, purely a step-8 sampling artifact) | **0.964 at every offset** (step-1 scan) |
| Cross-regulation / different-article quote | 0.128–0.402 | rejected | rejected |
| Wrong-meaning + extra unrelated clause | 0.767 | rejected | rejected |

Root causes and fixes:

- **Negation flip / appended-or-prepended fabrication (the "two
  floor-integrity holes"):** a character-level `difflib` ratio rewards the
  genuine majority of a quote and is structurally blind to what a small
  insertion says — inserting "not", or tacking a few words onto either
  end, barely moves the ratio. The appended/prepended case had a second,
  sharper root cause: when a (genuine+fabricated) quote is longer than the
  excerpt itself, the window scan's too-short-excerpt clamp collapses to
  exactly ONE comparison — the whole excerpt vs. the whole quote — and
  `ratio()` alone rewards near-total alignment of the genuine prefix
  regardless of what's tacked onto the end. **Fix:** the second condition
  above (`_is_ordered_subsequence`) — an inserted/appended/prepended word
  is, definitionally, a token the winning window's own text does not have
  in that position, so it now fails outright no matter how high the ratio
  reads. This also closes the same hole for the degenerate
  quote-longer-than-excerpt window: that window is still computed the same
  way, but its text no longer goes unchecked.
- **Window-scan step-8 sampling bug:** `_best_fuzzy_window` (renamed from
  `_best_fuzzy_score`, round 1) used to slide in steps of 8 as an
  unmeasured performance guard — for a quote near `_MIN_QUOTE_LENGTH` (20
  chars), that grid could step clean over the one offset where the
  genuine match actually sits, arbitrarily rejecting a citation whose
  quote never changed, only its position in the excerpt. **Fix:** deleted
  the step entirely — every offset is scanned. Measured cost (this
  session's own timing, a ~300-char quote — the `SYSTEM_PROMPT`'s stated
  max): a genuine drift quote scores fast regardless of excerpt size (the
  `quick_ratio()` prefilter prunes almost every candidate once a good match
matched (accept) path: 53–93 ms on a 10KB excerpt — dominated by the
per-offset scan loop (~4,900 iterations), not by SequenceMatcher itself
(review round 2 re-measured this; an earlier 0.08–0.16 ms figure was wrong).
  sharing almost no vocabulary with the excerpt is the adversarial worst
  case for that same prefilter (a low `best_ratio` never rises enough to
  prune later candidates, so nearly every window pays for a real `ratio()`
  call): 423ms at 5KB (this ADR's stated real-world ceiling), 874ms at
  10KB. This only costs anything on a citation that was going to be
  REJECTED anyway (the fuzzy fallback only runs after an exact-match miss,
  and a genuine quote is found in well under a millisecond); `answer_timeout_s`
  /`request_timeout_s` (ADR-0028) already bound the request as a whole
  (up to 5 citations at ~0.4s worst-case each is still a fraction of the
  60s request budget). Documented as a real, measured cost — not a
  re-opened blocker — and worth a prefilter (token-overlap or
  `difflib.get_close_matches` over sentence units) if a future excerpt-size
  or citation-volume increase ever makes it one; not worth adding
  pre-emptively today (YAGNI).

The floor itself (0.92) did not move — every fix above is a second,
independent condition alongside it, not a floor adjustment. Re-running the
original drift fixtures confirms they still pass (they only omit text, per
the note above) and the original adversarial fixtures still fail (already
well below 0.92 on the ratio alone, so the subsequence condition changes
nothing for them).

## Why not the others

- **Status quo**: the measured 30% benign-question failure rate on real
  legal questions IS the cost being paid today, for zero security benefit —
  none of it was a caught hallucination, all six were genuine on-topic
  answers.
- **Sentence-index citations**: a real alternative worth revisiting if
  quote drift keeps recurring after this fix, but a bigger change with its
  own unmeasured failure mode, for a problem this smaller fix already
  measurably closes.
- **LLM re-judge**: `critic_node` already spends one LLM call per request on
  a strictly harder version of this question (semantic support, not just
  string similarity) — a second, cheaper LLM call duplicating part of that
  coverage isn't worth the added latency/cost for what stdlib string
  matching already handles.
- **rapidfuzz**: no problem this project actually has (article-sized
  excerpts, 1-5 citations) that stdlib `difflib` doesn't already solve fast
  enough — see "How to reverse" for the one-line swap if that ever changes.

## Security & cost implications

- **Security:** the floor stays real — every adversarial fixture tried
  (wrong-meaning-same-words, cross-excerpt splice, wrong article, pure
  fabrication) scores 0.69 or below, comfortably under 0.92 — and round 2's
  added subsequence condition closes the two floor-integrity holes round 1
  wrongly treated as merely theoretical (negation flip, appended/prepended
  fabrication — see "Round 2" above). **Residual risk, accurately scoped
  after round 2:** the subsequence check is evadable only if the added/
  changed word ALREADY exists, in the right relative position, somewhere
  else in the same winning window — e.g. a genuine "not" appearing earlier
  in the same clause for unrelated reasons, coincidentally letting a
  negation trick's token requirement be satisfied without the word actually
  belonging where the quote places it. Narrower than round 1's "near-miss
  paraphrase" framing (which round 2 proved was not narrow at all — see the
  probe table above) but not zero. **`critic_node` (ADR-0023) is NOT a
  reliable backstop for this residual, and this ADR should not have implied
  otherwise in round 1:** reading `critic.py`/`critic_node` directly
  (graph/nodes.py) shows `critique()` is called ONCE per turn over the
  WHOLE drafted answer text against all of its cited excerpts combined
  (`critic_node`'s own docstring: "Context is built from `state["articles"]`
  ... CITED-only") — it is an answer-level faithfulness judgment, not a
  per-citation one. A short, evasion-crafted clause riding on an otherwise-
  faithful multi-citation answer is exactly the case where an answer-level
  judge is least likely to move its overall verdict. The actual backstop
  for this narrowed residual is monitoring: every fuzzy-accepted citation
  logs `quote_fuzzy_match` with its score (never the quote text) — a real
  evasion attempt would still show up there as a fuzzy accept close to the
  floor, the same signal this ADR already relies on, just without a second
  automated layer behind it today.
- **Cost:** zero new LLM calls — `difflib` is pure stdlib, CPU-only, and
  only runs at all on an exact-match miss (the common case pays nothing
  extra). The Day-17 amendment's "no LLM-client imports in this module"
  property is preserved (`settings` has no LangChain dependency either).

## Measured before/after (2026-09-02/03, `gpt-4.1-mini`, real DB + retrieval)

**Answer eval** (`evals/run_answer_eval.py --faithfulness-min 0.8`, 10
golden questions):

| | Before (ADR-0022 baseline) | After (`quote_similarity_min=0.92` + the `SYSTEM_PROMPT` contiguous-quote fix below) |
|---|---|---|
| faithfulness | 0.900 (c01 counted as a failure) | **1.000** |
| relevancy | (not separately tracked at baseline) | 0.900 |
| citation_error_rate | 0.100 (1/10, c01) | **0.000** |
| c01 | `CitationError`, no answer produced | Answers; judged faithful (relevancy flagged — it covers AI Act Art. 14/86 fully but omits the GDPR Art. 22(1) half of the two-part question, a coverage gap, not a citation-validity one) |

First run of this eval hit a transient `OpenAITimeoutError` on the judge's
own call (infra flake, not this change) — re-run produced the numbers
above.

**Red-team** (`evals/run_redteam.py --subset all --asr-max 0.05
--fpr-max 0.10`, 40 attacks + 20 benign):

| | Before (ADR-0022, 2026-08-26) | After |
|---|---|---|
| ASR | 0.000 (0/40) | **0.000 (0/40)** — unchanged, no attack fixture scores near the 0.92 floor |
| FPR | 0.000 (0/20 — guard refusals only, unaffected by this change) | **0.000 (0/20)** — unchanged |
| benign_citation_error_rate | 0.300 (6/20) | **0.050 (1/20)** — the number this ADR exists to move |

Cost of this session's two measurement runs: ~$0.020 (answer eval, 20 LLM
calls) + ~$0.0245 (red-team, 37 classifier + 24 answer calls) ≈ $0.045
total, in line with ADR-0017/ADR-0022's existing per-run cost estimates.

## How to reverse

- **Disable the fuzzy fallback entirely** (verbatim-only, ADR-0014's
  original behaviour): set `settings.quote_similarity_min = 1.0` — no
  `difflib` score reaches exactly 1.0 short of a match the exact substring
  check would already have caught, so this one env var is the kill switch,
  no code change.
- **Retune the floor**: one field, `Settings.quote_similarity_min`
  (`settings.py`) — re-run `tests/test_guards_quotes.py` and both eval
  scripts above after any change, same discipline this ADR's own floor
  search followed.
- **Swap the algorithm**: `_best_fuzzy_window`/`_is_ordered_subsequence`/
  `quote_matches` are the only functions involved (`guards/quotes.py`) —
  replacing the windowed `difflib` scan (or the subsequence check) with
  `rapidfuzz` or a different similarity/coverage measure is a change to
  those functions only; both callers (`_validate_citations`,
  `cite`) go through `quote_matches`'s same `QuoteMatch` return shape either
  way.
- **Revert the `SYSTEM_PROMPT` contiguous-quote instruction**: one paragraph
  in `graph/nodes.py`'s `SYSTEM_PROMPT` constant — removing it returns to
  ADR-0015's original quoting instructions; the fuzzy floor is independent
  of it either way (removing the prompt change would just reopen the c01
  splicing/large-ellipsis failure this ADR's floor correctly still rejects).

## References

- ADR-0014 (citation validation is a hard error — unchanged: fuzzy matching
  is a wider acceptance criterion, not a softer failure mode).
- ADR-0015 (retry-once edge — unchanged; the fuzzy fallback runs on every
  attempt, same as the exact check always did).
- ADR-0022 (red-team ASR/FPR gate — source of the measured 6/20 baseline
  and the `benign_citation_error_rate` metric this ADR's fix targets).
- `difflib.SequenceMatcher` — stdlib, `ratio()`/`quick_ratio()`/
  `set_seq2()` behaviour confirmed against the installed CPython stdlib
  source (`.venv`'s Python 3.12 `difflib.py`), not assumed.
