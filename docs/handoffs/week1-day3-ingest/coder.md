# Handoff — Week 1 Day 3: EUR-Lex ingestion (coder)

Branch: `feature/ingest-eurlex`. Task: fetch + parse the AI Act and GDPR from
Cellar into article/recital chunks. Pure — no DB writes, no embeddings
(Day 4).

## Files
- `src/compliance_copilot/ingest/__init__.py` — package docstring.
- `src/compliance_copilot/ingest/eurlex.py` — `REGULATIONS`, `ArticleChunk`,
  `fetch_xhtml`, `parse_articles`, `ingest_regulation`.
- `src/compliance_copilot/cli.py` — added `ingest --regulation
  ai_act|gdpr|all --dry-run` (existing `init-db` untouched).
- `tests/fixtures/eurlex_sample.xhtml` — real AI Act markup, Recitals 1-2 +
  Articles 1-3 (Article 3's definitions table truncated to its first 3
  entries, still real markup), 23 KB.
- `tests/test_ingest_eurlex.py` — unit, no network, 5 tests.
- `tests/test_ingest_eurlex_integration.py` — `integration` marker, skipped
  unless `RUN_NETWORK_TESTS=1`.
- `pyproject.toml` / `uv.lock` — added `httpx`, `selectolax`.

## Library choices + doc sources (verified via Context7 before writing)
- **httpx** `0.28.1` — `/encode/httpx`. `httpx.Client(headers=..., timeout=...,
  follow_redirects=True)`, `.get(url)`, check `response.status_code`.
  https://github.com/encode/httpx/blob/master/docs/advanced/clients.md,
  .../docs/advanced/timeouts.md, .../docs/quickstart.md
- **selectolax** `0.4.11` — `/rushter/selectolax` (Lexbor engine).
  `HTMLParser(html).css('div.eli-subdivision')` / `.css_first('#id')`,
  `.text(deep=True, separator=' ', strip=True)`, `.decompose()` to remove a
  child node before re-extracting the parent's text.
  https://github.com/rushter/selectolax/blob/master/README.rst,
  .../examples/walkthrough.ipynb
  Chose selectolax over lxml/bs4: id/class CSS selectors work directly
  (confirmed above), Lexbor is fast, and nothing in this parse needs
  lxml's XPath or bs4's more permissive-but-slower tree — no reason to
  reach further down the dependency list.
- **hashlib** — stdlib, `sha256(text.encode()).hexdigest()`.

## Real HTML structure (inspected live before writing the parser, per task instructions — fetched AI Act into data/raw, printed around id="art_3" and id="rct_1")
Each article/recital is `<div class="eli-subdivision" id="art_N">` /
`id="rct_N"`. An article's div additionally holds:
- `<p class="oj-ti-art">Article N</p>` — label, stripped from body text.
- `<div class="eli-title" id="art_N.tit_1"><p class="oj-sti-art">Title</p></div>`
  — stripped from body text, kept separately as `.title`.
- Body paragraphs as `<div id="NNN.NNN"><p class="oj-normal">1.   ...</p></div>`,
  and enumerated sub-items (definitions, list items) as
  `<table><tr><td>(1)</td><td>text</td></tr></table>` rather than plain `<p>`
  tags — this is why the parser strips label/title by `.decompose()` and
  takes `.text(deep=True)` on the whole div rather than assuming a flat list
  of `<p>` children.

Recitals have the same `eli-subdivision` div but no label/title, just the
`(N)` + text table — number comes from the `id`, not from parsing "(N)" out
of the text.

Article id `art_N` is matched with `re.fullmatch(r"art_(\d+)", id)` so it
doesn't also match the title's own nested id `art_N.tit_1`.

## Test results
- `uv run ruff check .` — clean.
- `uv run ruff format --check .` — clean (one file auto-formatted during dev).
- `uv run pytest -m "not integration"` — 11 passed, 2 skipped (the 2 existing
  DB integration tests, no DATABASE_URL set — pre-existing, unrelated).
- CLI dry-run against live Cellar: `ai_act: 113 articles, 180 recitals`,
  `gdpr: 99 articles, 173 recitals` — matches ADR-0012's verified counts
  exactly.
- `RUN_NETWORK_TESTS=1 uv run pytest -m integration -k eurlex` — 2 passed
  (real fetch, both regulations, counts asserted as above).

## Open issues / notes for next agent
- **Not added to CI**: `RUN_NETWORK_TESTS` is deliberately not referenced in
  `.github/workflows/ci.yml` — keeps CI deterministic/offline per task
  instructions. Someone should decide later whether a scheduled (not
  per-PR) job re-runs this against live Cellar to catch upstream structure
  drift.
- **Pre-existing gotcha, not this task's bug**: running the integration
  command exactly as specified in the task brief —
  `RUN_NETWORK_TESTS=1 DATABASE_URL= uv run pytest -m integration -k eurlex`
  — fails at *collection*, not at the eurlex tests: `DATABASE_URL=` (empty
  string) makes `compliance_copilot/settings.py` read an empty
  `database_url`, and `db.py` builds a module-level SQLAlchemy engine at
  import time (`create_engine(settings.database_url)`), so importing
  `tests/test_db_integration.py`/`test_db_models.py` raises
  `ArgumentError` before pytest's `-k` filter ever runs. Worked around
  locally with `--ignore=tests/test_db_integration.py
  --ignore=tests/test_db_models.py` (or just omit `DATABASE_URL=`, which
  falls back to the real default in settings.py and collects fine). Not
  fixed here — out of this task's scope (`db.py`/`settings.py` weren't
  touched) — but worth a follow-up: engine creation should probably be
  lazy, not module-level, so an unrelated empty env var doesn't break
  collection of unrelated test files.
- `ArticleChunk.regulation` holds the short `REGULATIONS` key (`"ai_act"` /
  `"gdpr"`), not the display title — matches what `--regulation` and
  `REGULATIONS` already use elsewhere; `REGULATIONS[key]["title"]` is where
  the display string lives if Day 4's DB-write step wants it for
  `Document.title`.
- `data/raw/32024R1689.xhtml` (1.26 MB) and `data/raw/32016R0679.xhtml`
  (0.8 MB) are on disk locally from verification runs — gitignored, not
  committed, will regenerate on any future `ingest_regulation()` call.

## Next step
Day 4: `embeddings.py` + wire `ingest_regulation()` output into
`Document`/`Chunk` DB writes (`src/compliance_copilot/db.py`) with HNSW
index, per `docs/CURRICULUM.md`.
