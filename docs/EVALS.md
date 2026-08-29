# Eval gates

Every automated quality/safety gate in this repo, what it measures, its
threshold, and where it runs. Referenced by ADR-0005 (eval harness) and
`docs/CURRICULUM.md`'s later lessons (judge calibration, cost). Thresholds
below are read from the actual `Makefile`/`.github/workflows/ci.yml`/
`evals/*.py` invocations, not restated from memory — if one of these
changes, update the flag in its source file first, then this table.

| Eval | Metric | Threshold | Where it runs | Cost |
|---|---|---|---|---|
| Retrieval quality | hit@5, MRR | hit@5 ≥ 0.93, MRR ≥ 0.80 | CI `quality-gate` job — every push/PR | Free (`EMBEDDINGS_PROVIDER=cached`, no network) |
| Answer faithfulness (LLM-as-judge) | faithfulness score | ≥ 0.8 | CI `answer-quality` job — nightly (03:00 UTC), `workflow_dispatch`, or a PR labelled `quality-gate` | Real `OPENAI_API_KEY` spend (ADR-0017) |
| Red-team ASR/FPR, full pipeline | attack success rate / false-positive rate | ASR ≤ 0.05, FPR ≤ 0.10 | CI `answer-quality` job — same trigger as faithfulness, above (folded in, not a separate job) | Real LLM spend, ~$0.02–0.03/run (ADR-0022) |
| Red-team heuristics subset | every `must_block_at: "heuristics"` attack flagged by `detect()` | 100% (any miss fails the test) | `tests/evals/test_redteam.py` — every push, default `pytest -m "not integration"` job | Free (stdlib regex only) |
| Trajectory — node-sequence/count assertions | exact `nodes_visited` list or exact count per named path (refusal, happy path, retry, hitl pause/resume, critic outage, confidence boundary) | pass/fail per assertion — no single numeric threshold | `tests/evals/test_trajectory.py` — every push, default `pytest -m "not integration"` job | Free (fake LLMs/tools, zero network) |
| Trajectory — LLM-call ceiling | total `.invoke()` calls across classifier+router+answer+critic in one run | ≤ `MAX_LLM_CALLS_PER_REQUEST` = 5 (`graph/build.py`) | `tests/evals/test_trajectory.py` — every push | Free |
| Router label accuracy, structural | fake-router label reaches `search_regulation`'s filter / the refuse path | 10/10 (`graph_helpers.ROUTER_FIXTURE`) | `tests/evals/test_trajectory.py` + `tests/test_graph.py` — every push | Free |
| Router label accuracy, real model | real `gpt-4.1-nano` label vs. hand-labelled expectation | ≥ 9/10 (10-question fixture); the older 3-question smoke test in the same file has no numeric threshold (all 3 must match) | `tests/test_router_real_integration.py` — `pytest -m integration` | A few cents (ten short completions) |

## A gap worth naming, not rounding away

The CI `integration` job (`.github/workflows/ci.yml`) runs `pytest -m
integration` on every push/PR — but its env only sets
`DATABASE_URL`/`TEST_DATABASE_URL`, no `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`.
Both router real-model tests in `tests/test_router_real_integration.py`
`skipif` on the missing key, so **they are skipped in CI today, not run**
— they only execute locally when a developer sources `.env` and runs
`pytest -m integration` by hand. Fixing this (adding the secret to that
job, the same way `answer-quality` already has it) is a one-line CI change,
not made in this feature — named here so it doesn't quietly look covered
when it isn't.

## Not yet built

- **Judge calibration** (Cohen's κ against human labels) — planned
  `docs/CURRICULUM.md` lesson 22; this file gets a κ value and confusion
  matrix once that lesson ships.
- **Cost per 100 questions** — planned lesson 24; this file gets a real
  €/100-questions figure from an actual golden-set run once that lesson's
  usage-tracking callback exists.
