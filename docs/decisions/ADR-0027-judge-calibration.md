# ADR-0027 — Judge calibration: does the LLM judge agree with a human?

**Status:** accepted 2026-08-29

## Context

ADR-0017 built `evals/judge.py`'s custom LLM-as-judge and named, but did not
measure, two risks: the judge model (`gpt-4.1-mini`) is the same model that
writes the answers it grades (a self-evaluation bias risk), and at n=10
golden answers one flaky verdict moves the aggregate faithfulness score by
0.1. Both were disclosed as caveats — "a regression signal, not an absolute
quality claim" — not as a measured number. A judge that produces a
confident-looking score every time is not the same thing as a judge that is
right; the only way to know the difference is to check it against a known-
correct source, the same reason a bathroom scale gets checked against a
known 5kg weight before it's trusted.

## Options considered

1. **Trust the judge as-is** (no calibration). Rejected: this is exactly the
   posture ADR-0017 already flagged as provisional — "a regression signal,
   not an absolute quality claim" — and never followed up on. Shipping a
   0.8 faithfulness gate off an uncalibrated judge means the gate's real
   false-pass rate is unknown.
2. **Spot-check a few outputs manually, ad hoc, no written protocol.**
   Rejected: without a fixed rubric and a blind-labelling order, a spot
   check just re-confirms whatever the judge already said (the human reads
   the judge's verdict first, then rationalises agreement) — it produces a
   number that *looks* like calibration without being one.
3. **Cohen's κ against a small set of human labels, blind-labelled against
   a mirrored rubric** (the option taken). 20 items — 10 golden Q&A pairs
   plus a deterministic 10-item sample of benign/red-team pipeline runs, a
   different answer *shape* the golden set alone never produces (short
   refusals, one-off benign phrasing). Items and the judge's own verdicts
   are written to separate files specifically so a human labels blind
   before seeing what the judge said.
4. **A different-vendor judge immediately** (e.g. switch to
   `ChatAnthropic` via `make_llm()`'s existing anthropic branch) instead of
   calibrating the current one first. Rejected for now — see "why not"
   below; kept as the mitigation this calibration number will justify or
   rule out.
5. **Majority-of-3 judge calls on every item.** Rejected as the default —
   3x the judge cost on every eval run to solve a problem calibration
   hasn't yet shown is worth solving. Kept as a targeted mitigation for
   borderline items near the pass/fail threshold, not a blanket policy.

## Decision

Built the calibration pipeline as three pieces, reusing existing pipeline
code rather than reimplementing it (`evals/run_answer_eval.py`'s
`_contexts_for`/`load_golden_answers`, `evals/run_redteam.py`'s
`load_attacks`/`load_benign`/`build_graph`/`GraphContext`, `evals/judge.py`'s
`judge()` unchanged):

- **`evals/build_calibration_set.py`** — runs the 10 golden Q&A pairs
  through the real `retrieve -> answer` graph + judge, and a deterministic
  subset (first 5 sorted ids from each file — no randomness) of
  `evals/redteam.jsonl`/`evals/benign.jsonl` through the full
  `guard_in -> retrieve -> answer -> guard_out` graph, then judges those
  too (the judge never sees red-team/benign items in normal operation —
  `run_redteam.py`'s own scoring is deterministic, ADR-0022 — this is
  purely to get a different answer shape to calibrate against). Writes
  `items.jsonl` (what a human reads) and `judge_verdicts.jsonl` (the
  judge's own verdict) as **separate files on purpose**, plus
  `labels.template.jsonl` for a human's own labels.
- **`evals/calibration/RUBRIC.md`** — mirrors `JUDGE_SYSTEM_PROMPT`'s two
  criteria (`faithful`, `relevant`) word-for-word, plus the blind-labelling
  protocol: label first against `items.jsonl` only, reveal
  `judge_verdicts.jsonl` after, one line of "why" per label.
- **`evals/run_judge_calibration.py`** — loads the three files (erroring
  clearly on a missing/extra/unlabelled id), then per dimension reports raw
  agreement, Cohen's κ (`(po - pe) / (1 - pe)`, plain Python, no new
  dependency), the 2×2 confusion matrix, and the dangerous
  judge-yes/human-no cell **printed first**. Defaults to `labels.jsonl` if
  present, else `labels.provisional.jsonl` with a loud PROVISIONAL banner.
  Report-only by default; `--kappa-min` is opt-in for a future CI gate.
  `make calibration` / `make calibration-build` (Makefile).

**Provisional result (see `docs/EVALS.md`'s "Judge calibration" section for
the full table and discussion), labelled by the coder agent, not yet
reviewed by Jay:** `faithful` — raw agreement 1.000, κ = 1.000, 0/20
dangerous cells. `relevant` — raw agreement 0.850, κ = 0.634, **3/20
dangerous cells**, all three on red-team refusal items (`rt01`–`rt03`)
where the judge scored a canned safety refusal as "relevant" to a
malicious extraction request; the human label applied the rubric's literal
wording ("a partial, evasive... answer does not count as relevant") and
disagreed. Landis & Koch would call κ=0.634 "substantial" but below
"almost perfect" — a real, non-zero disagreement rate on the very first
pass, concentrated in exactly the self-preference/leniency shape ADR-0017
predicted.

**Judge biases named, not assumed away** (lesson 22's list, kept here since
this ADR is the thing that measures them):
- **Self-preference** — this project's live risk (ADR-0017): judge and
  answer model are the same `gpt-4.1-mini`.
- **Verbosity bias** — LLM judges in the wider literature rate longer
  answers as better regardless of correctness; worth watching since a
  refusal is short and a real answer is long.
- **Position bias** — order-sensitivity in a side-by-side comparison; not
  directly exposed here since `judge()` scores one answer at a time, but
  the reason a future *pairwise* judge design would need position-swapping
  controls this single-answer design doesn't.
- **Leniency** — instruction-tuned judges skew toward "yes it's fine," most
  visible here on `relevant`, the criterion lesson 10 already flagged as
  more phrasing-sensitive than `faithful`.

**Re-measure rule:** recalibrate whenever the judge model, vendor, or
temperature changes — a κ value is only true of the exact judge it
measured, the same "the cache key changes when the thing changes"
principle already applied to embeddings (ADR-0017a).

**Proposed fix for the `relevant` ambiguity on refusals (pending human
labels):** the dangerous-cell items are all refusal-shaped, and the rubric
gives the judge no rule for them — two candidate fixes, to be chosen once
the human labels confirm which side the disagreement falls on: (a) tighten
`JUDGE_SYSTEM_PROMPT`'s `relevant` criterion with one explicit sentence
("a refusal is scored `relevant=false` unless the question itself was
out of scope"), or (b) score refusal-shaped answers on `faithful` only and
exclude them from `relevant` aggregation. Either way the change re-triggers
the re-measure rule below.

## Why not the others

- **Trusting the judge as-is**: rejected — see Context. An undisclosed
  false-pass rate on a merge gate is exactly the failure mode calibration
  exists to catch, and ADR-0017 already flagged it as unresolved.
- **Ad hoc spot-checking**: rejected — without blind labelling against a
  written rubric, a "check" that sees the judge's verdict first just
  measures whether a human can be talked into agreeing with a plausible-
  sounding score.
- **Different-vendor judge immediately**: not rejected outright, deferred.
  Switching judges before measuring the current one means never knowing
  whether the switch actually fixed anything, or just moved the bias
  somewhere else (a new vendor has its own untested bias profile). The
  result that would change this: if a future calibration run (ideally with
  Jay's real labels) shows dangerous-cell disagreement on `faithful` — not
  just `relevant` — that is the signal ADR-0017 already named as the
  trigger to spend the switch (`make_llm()`'s anthropic branch exists
  today, no code to write). This run's `faithful` dangerous-cell count is
  0/20, so the switch isn't justified by this evidence yet.
- **Majority-of-3 by default**: rejected as a blanket policy — 3x judge
  cost on every eval run for a problem this calibration hasn't shown
  applies broadly. Kept as a targeted, cheap mitigation (only borderline
  items) if a future calibration run shows a specific score band is where
  disagreement concentrates.

## Security & cost implications

- **Security:** no new PII surface. The golden answers are synthetic/
  public-law questions and the red-team/benign items are already-public
  fixture data (ADR-0022) — safe to keep `evals/calibration/*.jsonl`
  committed to the repo, same reasoning ADR-0017 already applied to
  `golden_answers.jsonl`. This safety property is specific to where these
  20 items come from — a calibration set built from real production traces
  would need Langfuse's existing redaction discipline (ADR-0009), not an
  automatic pass just because "it's only 20 items."
- **Cost:** the real cost is human time, not API spend — 20 careful reads
  take longer than any script here runs. `evals/build_calibration_set.py`
  costs the same order of magnitude as one `run_answer_eval.py` run (~40-50
  LLM calls, a few cents, `gpt-4.1-mini`/`gpt-4.1-nano` pricing, ADR-0002/
  ADR-0019). `evals/run_judge_calibration.py` itself costs nothing — it
  only reads already-generated jsonl files.

## How to reverse

Delete `evals/build_calibration_set.py`, `evals/run_judge_calibration.py`,
`evals/calibration/`, and the two `Makefile` targets — nothing else in the
eval pipeline depends on this feature; `evals/judge.py`/
`run_answer_eval.py`/`run_redteam.py` are all read-only inputs to this ADR,
none of them changed. Swapping the κ/confusion-matrix arithmetic for a
library (`sklearn.metrics.cohen_kappa_score`, `statsmodels`) is a one-file
change inside `run_judge_calibration.py` — the file formats
(items/verdicts/labels jsonl) don't depend on the arithmetic implementation.

## References

- ADR-0017 (`docs/decisions/ADR-0017-quality-gate-cached-embeddings-custom-judge.md`)
  — the self-preference bias and n=10 noise caveats this ADR turns into a
  measured number.
- ADR-0022 (`docs/decisions/ADR-0022-redteam-asr-gate.md`) — the red-team/
  benign fixture files and `blocked_by`/`GraphContext` pipeline pieces this
  ADR reuses for the non-golden half of the calibration set.
- Landis & Koch (1977), the conventional κ interpretation bands (0.6–0.8
  "substantial," below ~0.4 "not trustworthy unsupervised") — the reading
  applied to this ADR's measured κ values.
- `docs/EVALS.md`'s "Judge calibration" section — the full measured table
  and per-item discussion this ADR summarises.
