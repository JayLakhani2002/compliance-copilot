# ADR-0006: Guardrails — layered custom guardrails (input + output)

## Status
Accepted. 2026-08-23.

## Context
A legal-adjacent RAG system has two distinct failure classes to guard against: **untrustworthy input** (prompt injection — text crafted to make the model ignore its instructions; PII the user pastes in; off-topic questions) and **untrustworthy output** (an answer that makes a claim with no supporting citation, or that doesn't match the expected response schema). Market research shows guardrails/safety named explicitly in ~10% of ads but "with senior framing" — e.g. "prompt injection mitigation," "Governance, Guardrails, Datenschutz (DSGVO) und AI-Act" — meaning depth here matters more than the raw naming percentage suggests for the target audience.

## Options considered
1. **Layered custom guardrails**, split at input and output:
   - **Input** (`guard_in` node, `docs/ARCHITECTURE.md` §4): prompt-injection heuristics (pattern/regex checks for known injection phrasing) + a Haiku classifier (ADR-0002) for less obvious injection attempts; PII detection/redaction via **Microsoft Presidio**; a topic/scope check (is this actually a question about the AI Act or GDPR).
   - **Output** (`guard_out` node): Pydantic structured output (the LLM's answer is validated against a schema, not accepted as free text); a citation-must-exist check (every claim's cited article must actually appear in this run's retrieved chunks — not just look like a citation); a refusal policy (what the system says when it declines to answer).
2. **Guardrails AI** (a Python library/DSL for defining and enforcing output "guards" like schema validation, PII checks, etc., built around composable validators) — capable, but adds a dependency with its own DSL/config surface to learn and debug, for functionality this project can express directly in Pydantic + a few focused checks.
3. **NeMo Guardrails** (NVIDIA's dialogue-rails framework, using a "Colang" config language for defining conversational rails) — powerful for complex multi-turn dialogue policies, but heavier than what a mostly-single-turn Q&A system needs, and introduces a second configuration language (Colang) alongside Python, which works against the project's teaching goal ("write for a reader who knows basic Python").
4. **Llama Guard** (Meta's safety-classifier model, used as a moderation layer) — a real, credible option, but treated here as a **future comparison eval** rather than the primary mechanism: swapping in a fixed pretrained safety classifier is a good add-on experiment (does it catch things the custom Haiku classifier misses?) but shouldn't be the primary guardrail on day one, since the project's teaching value is in building and explaining the guardrail logic, not in deferring entirely to an off-the-shelf classifier.

## Decision
**Layered custom guardrails**, as described in option 1, built directly in Python + Pydantic + Presidio + a Haiku classifier call — no guardrail-DSL library. Guardrails AI and NeMo Guardrails are explicitly **not** used as the primary mechanism. Llama Guard is noted as a candidate **comparison eval** to add later (does a pretrained safety classifier catch injection attempts the custom heuristics/Haiku classifier miss, and vice versa) — not part of the core `guard_in`/`guard_out` path.

**Presidio verified detail:** `presidio-analyzer` (PyPI, version 2.2.364, verified 2026-08-23) is the PII-detection half of Microsoft's Presidio project; pairing it with `presidio-anonymizer` (not separately verified in this pass, but is Presidio's companion redaction package) gives detect-then-redact. Presidio's versioning uses a high, frequently-incrementing patch number (a CI-driven build-number scheme), which is expected — it is not a sign of an unstable/abandoned package, just a different release cadence than semantic-versioned libraries in this stack.

## Why not the others
- **Guardrails AI**: rejected as the *primary* mechanism because this project's guardrail logic (a handful of specific checks: injection heuristics, PII redaction, topic scope, citation existence, schema validation) is simple enough to write directly and explain line-by-line to a reader who knows basic Python — adding a DSL/config layer on top would obscure rather than clarify the logic, working against the project's explicit teaching goal.
- **NeMo Guardrails**: rejected for the same "extra config language" reason, amplified — Colang is a genuinely separate thing to learn, and this project is mostly single-turn Q&A rather than the complex multi-turn dialogue-policy scenarios NeMo Guardrails is built for.
- **Llama Guard**: not rejected, deferred — it's a legitimate heavier dependency (a model to run/call) that adds real value as a *second opinion* eval, but committing to it as the primary layer would mean the guardrail behavior is "whatever Llama Guard decides" rather than something built and explained as part of this project.

## Security & cost implications
- **Security:** this ADR *is* the security control for the input/output trust boundaries described in `docs/ARCHITECTURE.md` §6 — `guard_in` is explicitly called out there as the highest-severity bug surface in the codebase, since every node downstream of it assumes its input has already been checked. The citation-must-exist check in `guard_out` is the specific control against the "no citation found" failure mode in `docs/ARCHITECTURE.md` §7 (default behavior: refuse rather than emit an uncited claim).
- **Cost:** the Haiku classifier call in `guard_in` is an extra LLM call per request beyond the router/answer/critic calls already counted in ADR-0002's cost model — small (classification-sized prompt, Haiku pricing), but worth remembering when tallying total calls/request (four Haiku/Sonnet calls per request end-to-end: injection classifier, router, answer, critic). Presidio runs locally (no per-call API cost), so PII redaction adds latency, not spend.

## How to reverse
Each guardrail is a small, independently swappable function/check — replacing the injection heuristic+classifier with Llama Guard, or replacing Presidio with a different PII library, touches only that one function's implementation, not the `guard_in`/`guard_out` node interface (which is "take state in, return possibly-modified state or a refusal, plus a guardrail-event log entry" regardless of what's inside).

## References
- `presidio-analyzer`, PyPI: 2.2.364 — https://pypi.org/project/presidio-analyzer/ (verified 2026-08-23); docs: https://microsoft.github.io/presidio/
- Market data on guardrails naming and framing ("Governance, Guardrails, Datenschutz (DSGVO) und AI-Act"): `docs/research/market_research.md`
