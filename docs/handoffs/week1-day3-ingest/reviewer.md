# Handoff — Week 1 Day 3: EUR-Lex ingestion (reviewer)

Branch: `feature/ingest-eurlex`. Reviewed `git diff develop...feature/ingest-eurlex`
in full (11 files, +928/-5): `src/compliance_copilot/ingest/eurlex.py`,
`ingest/__init__.py`, `cli.py`, both test files, the fixture, pyproject/uv.lock,
ADR-0012/lesson-03 additions.

## Verdict: **APPROVE**

No blockers. One major (recital count isn't sanity-checked), rest are minor/nits.
Correctness is solid: parser produces exactly the expected counts, no empty/short
articles, no duplicate anchors, numbering preserved.

## Article-length stats (live run against cached `data/raw/32024R1689.xhtml`, via
throwaway script, not committed)

- articles: 113, recitals: 180 (matches ADR-0012's verified counts, matches
  `expected_articles` sanity check)
- lengths: min=149, median=1862, max=17079 chars
- 3 shortest: Art. 87 (149 chars, single cross-reference sentence to Directive
  (EU) 2019/1937), Art. 94 (204 chars), Art. 64 (211 chars) — all genuinely
  short articles in the real text, not truncation artifacts (read the full
  sentences, they're complete and grammatical)
- 2 longest: Art. 5 (11,080 chars — the prohibited-practices list), Art. 3
  (17,079 chars — Definitions)
- 0 articles with text < 40 chars or empty — no parser-bug signal
- 0 duplicate anchor ids
- Article 3 (Definitions): title correctly extracted as `"Definitions"`,
  body text opens with `"For the purposes of this Regulation, the following
  definitions apply: (1) 'AI system' means..."` — the `(1)`/`(2)`/... numbered
  list markers are preserved inline (68 `(N) '...'` definition entries found
  via regex over the parsed text), confirming the definitions table's
  numbering is NOT lost despite each entry being a separate `<table>` (per
  coder's note that sub-items render as tables, not `<p>` siblings) — the
  `.text(deep=True)` walk over the whole `div` picks up every nested table's
  text in document order regardless of tag type, so the table-vs-`<p>`
  structural quirk doesn't lose or reorder anything.
- Paragraph-numbering / letter-list preservation spot-checked directly:
  `_normalise_whitespace("1.\xa0\xa0\xa0The purpose...")` → `"1. The purpose..."`;
  `_normalise_whitespace("(a)\xa0\xa0\xa0harmonised rules")` → `"(a) harmonised
  rules"`. Python's `\s` (Unicode mode, the default for `str` patterns) does
  match `\xa0` (NBSP), so the `&nbsp;&nbsp;&nbsp;` EUR-Lex uses after "1." /
  "(a)" collapses to a single space and the numbering itself is kept, not
  stripped. This was worth checking explicitly — an easy way for this exact
  regex to have accidentally swallowed the numbering if `\s` hadn't matched
  NBSP, and it's exactly the failure mode the task description warned about.
- No letter-suffixed article ids (e.g. `art_6a`) present in either cached
  source file (`grep -oE 'id="art_[0-9]*[a-zA-Z]+' data/raw/*.xhtml` → no
  matches in either), so `_ARTICLE_ID_RE = re.compile(r"art_(\d+)")` doesn't
  currently silently drop any article — but see Finding 1, this is a latent
  gap, not a verified-safe design choice.

Test runs: `uv run pytest -m "not integration" -q` → 11 passed, 2 skipped
(pre-existing DB tests, unrelated). `RUN_NETWORK_TESTS=1 uv run pytest -m
integration -k eurlex -q --ignore=tests/test_db_integration.py` → 2 passed
(both regulations' real counts, served from the already-cached xhtml so this
run didn't actually re-hit the network, but exercises the same code path).

## Findings

### 1. [Major] Recital count has no sanity check — only articles do
`src/compliance_copilot/ingest/eurlex.py:187-207` (`ingest_regulation`) only
validates `len(articles) == meta["expected_articles"]`. `REGULATIONS` has no
`expected_recitals` key, so if EUR-Lex changes the recital markup (a
different id scheme, or `rct_N` disappears while `art_N` stays intact) the
production ingest path (`ingest_regulation`, the only thing the CLI calls)
would silently return 0 or partial recitals with no error — exactly the
"parser silently returns 0" failure mode `docs/lessons/03_corpus_and_chunking.md`
names as the reason for a count-sanity check in the first place. The live
recital counts (180 / 173) are currently only asserted in the *integration
test* (`tests/test_ingest_eurlex_integration.py`), which isn't run by default
CI and isn't what an operator running `ingest --dry-run` in practice relies
on for the "did this actually work" signal.
**Fix:** add `"expected_recitals": 180` / `173` to `REGULATIONS`, check both
counts in `ingest_regulation`, matching the existing article check's shape.

### 2. [Minor] `fetch_xhtml` cache write isn't atomic — plausible cache
poisoning via a partial/corrupted file
`src/compliance_copilot/ingest/eurlex.py:104` —
`cache_path.write_text(response.text, encoding="utf-8")` writes directly to
the final cache path. If the process is killed mid-write (OOM, disk full,
SIGKILL during a 1.2 MB write), a partial file is left on disk, and the next
call's `if cache_path.exists(): return cache_path.read_text(...)`
(line 96-97) treats it as a fully valid cache hit — no size/well-formedness
check at read time. In the `ingest_regulation` call path this eventually
gets caught by the article-count check (finding 1's gap notwithstanding),
but `fetch_xhtml` is itself a public function another caller could use
directly and get truncated XHTML silently.
**Fix:** write to a temp file in the same directory and `os.replace()` it
into place (atomic on POSIX and Windows) — small diff, closes the gap
completely rather than relying on a downstream count check to catch it.

### 3. [Minor] Network/connection errors aren't normalized like HTTP-status
errors
`src/compliance_copilot/ingest/eurlex.py:98-103` — a non-200 response is
wrapped in a friendly `RuntimeError` naming the CELEX id and URL, but
`client.get(url)` itself (line 98) can raise `httpx.ConnectError`,
`httpx.TimeoutException`, etc., and those propagate raw and uncaught — an
operator hitting a DNS failure or Cellar outage gets an unguarded httpx
traceback instead of the same kind of message. Not a hard bug (exceptions
still surface, nothing is swallowed), but inconsistent error handling for
what's conceptually the same "fetch failed" case.
**Fix (optional, low priority):** wrap the `client.get(url)` call in
`try/except httpx.HTTPError` and re-raise as the same `RuntimeError` shape,
or leave as-is and note it's an accepted gap — either is defensible for a
Day 3 scope.

### 4. [Minor] Fixture typo contradicts its own "real markup" claim
`tests/fixtures/eurlex_sample.xhtml` — Article 1's title renders as
`Subject matter\`` (stray trailing backtick inside
`<p class="oj-sti-art">Subject matter\`</p>`). The real AI Act Article 1
title is "Subject matter" with no backtick. `docs/handoffs/week1-day3-ingest/coder.md`
describes the fixture as "real AI Act markup" (only calling out the
Article 3 definitions-table truncation as an intentional edit), so this
reads as an accidental edit slipping into what's meant to be a byte-faithful
excerpt. No test currently checks Article 1's title text so it wasn't
caught. Doesn't affect production code (fixture is test-only) — flagging for
fidelity, not correctness.
**Fix:** remove the stray backtick from the fixture.

### 5. [Nit] `REQUEST_HEADERS`'s User-Agent comment overclaims
`src/compliance_copilot/ingest/eurlex.py:33` — the comment above
`REQUEST_HEADERS` says "a real contact URL so an EU Publications Office
admin could see who's hitting the endpoint" but the User-Agent string is a
GitHub repo URL, not a contact email/address as `docs/lessons/03` (and the
polite-crawler norm it invokes) actually calls for. Not a functional issue —
the User-Agent header is present and does identify the crawler, which is
the actual requirement — just a slight comment/practice mismatch. Fix only
if convenient (e.g. append a contact email query param or note), not worth
a follow-up on its own.

## Robustness checklist (task item 2)
- Non-200 handling: present, raises `RuntimeError` with celex/status/url
  (eurlex.py:99-102). Good.
- Cache poisoning: gap — finding 2 above (no atomicity, no read-time
  validation). The `ingest_regulation` count check is a partial mitigation
  but doesn't cover recitals (finding 1) or direct `fetch_xhtml` callers.
- User-Agent: present, identifies the crawler and a repo URL (finding 5,
  nit only).
- Timeout: set (`timeout=60`, eurlex.py:98). Good — no unbounded hang risk.
- Network errors: partial gap — finding 3, connection/timeout exceptions
  aren't normalized the way HTTP-status errors are.

## Security (task item 3)
- Nothing executes HTML: selectolax is used purely for CSS-selector
  text/attribute extraction (`.css`, `.css_first`, `.text()`,
  `.attributes.get("id")`) — no `eval`, no template rendering of the parsed
  content, no HTML re-serialization that could later be interpreted as
  markup by a downstream consumer. Confirmed by reading every selectolax
  call site in `eurlex.py`.
- No secrets: only public EUR-Lex/Cellar URLs and CELEX ids; no API keys or
  credentials anywhere in this diff.
- No PII: source corpus is public EU legal text; nothing user-supplied
  enters this module (it's a pure fetch+parse of a fixed CELEX id set).
- Cache files (`data/raw/*.xhtml`) confirmed gitignored
  (`git check-ignore -v` matches `.gitignore:26:data/raw/`) — not committed,
  consistent with `docs/handoffs/week1-day3-ingest/coder.md`'s note that
  they're local-only.

## Teaching (task item 4)
Header comments present and accurate in every changed source file
(`ingest/eurlex.py`, `ingest/__init__.py`, `cli.py`) — each states the
file's purpose and place in the architecture, matching what the code
actually does. Why-comments are present at the non-obvious decision points
(Cellar vs. eur-lex.europa.eu, selectolax vs. lxml/bs4, cache-first fetch,
`.decompose()` before `.text()`, `fullmatch` to avoid matching
`art_N.tit_1`, required `--dry-run` flag). All spot-checked against the
actual code behavior — no comment found that misdescribes what its code
does.

## Docs claims spot-checked against Context7 (task item 5)
All confirmed correct, no fabricated API surface:
- `httpx.Client(headers=..., timeout=..., follow_redirects=True)` — all
  three are real `Client.__init__` kwargs (`/encode/httpx` docs); default
  `follow_redirects` is `False`, so passing `True` explicitly is required
  and correctly done (eurlex.py:98).
- `client.get(url)` / `response.status_code` — standard httpx `Response`
  usage, matches docs.
- selectolax `HTMLParser(html).css(selector)` / `.css_first(selector)` /
  `.text(deep=True, separator=..., strip=...)` / `.decompose()` — all
  confirmed against `/rushter/selectolax` README + walkthrough examples;
  `decompose()` specifically confirmed as the documented way to remove a
  node from the tree before re-extracting a parent's text, which is exactly
  how it's used here (strip the title/label divs before taking the body's
  `.text()`).
- No dataclass/pydantic API claims to check here beyond stdlib
  `@dataclass` on `ArticleChunk` (eurlex.py:69) — plain stdlib dataclass
  usage, no exotic features, nothing to verify against Context7.

## `db.py` module-level engine (task item 6)
Confirmed pre-existing and untouched by this diff
(`git diff develop...feature/ingest-eurlex -- src/compliance_copilot/db.py
src/compliance_copilot/settings.py` is empty). Reproduced the coder's
report: `DATABASE_URL= uv run pytest -m integration -k eurlex -q` fails at
*collection* with `sqlalchemy.exc.ArgumentError: Could not parse SQLAlchemy
URL from given URL string`, because `db.py:90` does
`engine = create_engine(settings.database_url)` at import time, and
`test_db_integration.py`/`test_db_models.py` import `compliance_copilot.db`
regardless of `-k eurlex` filtering — pytest resolves imports for every
collected file before applying `-k`.

**Recommendation: make the engine lazy (module-level `@lru_cache`-wrapped
function, e.g. `get_engine()`), not keep as-is.** Severity: **minor**, but
worth fixing soon — not urgent enough to block this PR (it's pre-existing,
out of this diff's scope, and every *actual* call path in this repo today
sets a real `DATABASE_URL` or accepts the working default), but it's a
correctness footgun that will keep surprising people running scoped `-k`
test commands (as literally happened to the coder while following the
task's own instructions), and the fix is genuinely small: wrap
`create_engine(...)` in a cached function, replace the five or so
`from compliance_copilot.db import engine` usages with `get_engine()` calls.
Low blast radius, no schema/behavior change. File as a follow-up task rather
than doing it in this PR (out of scope per `CLAUDE.md`'s one-feature-per-branch
rule) — but flag it now so it doesn't get lost.

## Scope/YAGNI vs CURRICULUM Day 3 (task item 7)
Correctly scoped. This PR is fetch+parse+dry-run only — no DB writes, no
embeddings, both explicitly deferred to Day 4 per
`docs/handoffs/week1-day3-ingest/coder.md`'s "Next step" and matching
`docs/CURRICULUM.md`'s day split. `ingest/eurlex.py` has no side effects
beyond the on-disk XHTML cache (no DB session opened, no embedding client
constructed) — genuinely pure, testable with a fixture and no network, as
the module docstring claims. The CLI's `--dry-run` being `required=True`
rather than silently defaulting is a reasonable, minimal way to make "no
other mode exists yet" explicit at the call site without building unused
flags ahead of need. `argparse` over click/typer for "a couple of
subcommands" is appropriately minimal — no unrequested abstraction. Nothing
in this diff reaches ahead into Day 4 work; nothing is under-built for what
Day 3 actually needs either.
