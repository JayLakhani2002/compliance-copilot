# src/compliance_copilot/api.py — the HTTP surface for the graph
# (docs/ARCHITECTURE.md §5-6, ADR-0008, ADR-0016). Two real routes: `/ask`,
# an API-key-gated, rate-limited, streaming (SSE) wrapper around
# `graph.astream(...)`, and `/resume` (ADR-0025), which continues a run
# `/ask` paused via `hitl_node`'s `interrupt()`. `/healthz` is
# unauthenticated and does no DB/LLM work, for a container orchestrator's
# liveness probe.
#
# Why one module, not a package: this is still a handful of routes sharing
# one dependency set and one SSE-framing convention — splitting
# routes/deps/schemas into separate files would be indirection with nothing
# on the other side of it yet (ponytail). Split it the day a third
# meaningfully different surface (not just another route on the same graph)
# actually lands.
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from langchain_core.embeddings import Embeddings
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from compliance_copilot import embeddings as embeddings_module
from compliance_copilot import tracing
from compliance_copilot.checkpointer import build_checkpointer, validate_thread_id
from compliance_copilot.critic import make_critic_llm
from compliance_copilot.db import get_session
from compliance_copilot.graph import CitationError, GraphContext, make_mcp_tools
from compliance_copilot.graph.build import build_graph
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.guards.classifier import make_classifier_llm
from compliance_copilot.guards.output import OutputGuardError
from compliance_copilot.logging_filter import install_pii_scrub
from compliance_copilot.router import make_router_llm
from compliance_copilot.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request schema — the trust boundary (ADR-0006 §"input"): nothing past this
# validation has to assume an unbounded or malformed question. `extra=
# "forbid"` rejects any field the client wasn't asked for, rather than
# silently ignoring it (a client sending a stray field is a signal something
# is wrong, not something to shrug off).
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=settings.max_question_chars)
    # ADR-0024: absent (the common case — a conversation's first turn) means
    # the server mints a fresh `uuid.uuid4()` (the `ask` route below) and
    # returns it in the first SSE event so the client can continue the
    # conversation by sending it back here on the next call. A
    # client-supplied value must look like something the server itself
    # would have issued — `validate_thread_id` (checkpointer.py) rejects
    # anything else with a 422, since a guessable/sequential id would let a
    # caller resume or read another session's checkpointed conversation
    # (ADR-0024's security note — this does NOT close the separate,
    # still-open ADR-0016 gap: the shared API key means any key holder can
    # still supply any validly-SHAPED thread_id and resume ANY thread).
    thread_id: str | None = Field(default=None)

    @field_validator("thread_id")
    @classmethod
    def _thread_id_must_be_uuid4(cls, v: str | None) -> str | None:
        return v if v is None else validate_thread_id(v)


# ---------------------------------------------------------------------------
# ADR-0025: `/resume`'s request body — same trust-boundary posture as
# `AskRequest` above (`extra="forbid"`, a bounded `edited_answer` length).
# `edited_answer` is required exactly when `decision == "edit"` (enforced by
# the `model_validator` below, not just a docstring convention) — the
# operator's own text still has to reach `guard_out` unchanged by this
# schema, this validator only decides whether a value is REQUIRED, never
# rewrites it.
# ---------------------------------------------------------------------------
class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Not `str | None` with a server-issued default like `AskRequest.
    # thread_id`: resuming a run that doesn't exist yet makes no sense, so
    # the client must already have one (from the `interrupt` SSE event or
    # the CLI's "under review" line).
    thread_id: str
    # ADR-0025 round 2 (BLOCKER 2): required — the `interrupt_id` `/ask`'s
    # `interrupt` SSE event already returned. `_require_paused_thread`
    # compares this against the ACTUAL pending interrupt's `.id` (from
    # `graph.aget_state`) before applying anything: a mismatch means the
    # caller is resolving a STALE pause (a later `/ask` call already
    # re-paused this same thread on a different question, or a previous
    # `/resume` already resolved this exact one) — 409, never silently
    # applied to the wrong draft.
    interrupt_id: str
    decision: Literal["approve", "edit", "reject"]
    # Reuses `max_question_chars` (no separate "max answer length" setting
    # exists yet, and an edited answer is the same order of magnitude of
    # text) — same "bound cost/abuse at the trust boundary" reasoning
    # `AskRequest.question` already applies.
    edited_answer: str | None = Field(default=None, max_length=settings.max_question_chars)

    @field_validator("thread_id")
    @classmethod
    def _thread_id_must_be_uuid4(cls, v: str) -> str:
        return validate_thread_id(v)

    @model_validator(mode="after")
    def _edited_answer_matches_decision(self) -> ResumeRequest:
        if self.decision == "edit" and not self.edited_answer:
            raise ValueError("edited_answer is required when decision is 'edit'")
        if self.decision != "edit" and self.edited_answer is not None:
            raise ValueError("edited_answer is only allowed when decision is 'edit'")
        return self


# ---------------------------------------------------------------------------
# Auth — API-key header, constant-time compare (ADR-0016: why not OAuth/JWT
# yet — a single-tenant portfolio API has no user identity to federate, so a
# shared secret is the right amount of mechanism today).
# `auto_error=False`: FastAPI's built-in auto-401 only ever returns a fixed
# generic message — this app needs to distinguish "no key" (401) from "wrong
# key" (403) from "server has no key configured" (503), so the check is
# manual.
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    if settings.api_key is None:
        # Never "auth disabled" — an unconfigured key must fail closed, not
        # open (project rule: security defaults to refusing, not permitting).
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API_KEY not configured")
    if key is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header")
    # secrets.compare_digest, not `==`: a naive string compare can return
    # faster the sooner it hits a mismatched byte, which leaks how many
    # leading characters were right to anyone timing the response — a
    # timing attack (ADR-0016).
    if not secrets.compare_digest(key, settings.api_key):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid X-API-Key")


# ---------------------------------------------------------------------------
# Dependency factories for the LLM/embeddings clients — mirrors db.py's
# `get_session` shape so tests can override them the same way
# (`app.dependency_overrides`). `lru_cache(maxsize=1)`: build once per
# process (a `ChatOpenAI`/`OpenAIEmbeddings` client is safe to reuse across
# requests, same reasoning as build.py's cached `build_graph()`), not once
# per request.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_llm_dependency() -> Any:
    return make_llm()


@lru_cache(maxsize=1)
def get_embeddings_dependency() -> Embeddings:
    return embeddings_module.get_embeddings()


@lru_cache(maxsize=1)
def get_classifier_dependency() -> Any | None:
    """ADR-0019: `None` when `settings.classifier_enabled` is `False` —
    `guard_in_node` treats a `None` classifier as "layer 2 disabled",
    skipping straight past it (same as before this feature existed). Built
    once at `lifespan()` startup (below), same `lru_cache(maxsize=1)`
    "build once per process" reasoning as `get_llm_dependency` above."""
    if not settings.classifier_enabled:
        return None
    return make_classifier_llm()


@lru_cache(maxsize=1)
def get_router_dependency() -> Any | None:
    """ADR-0023: same "disabled means None" contract as
    `get_classifier_dependency` above — `router_node` no-ops when this is
    `None`."""
    if not settings.router_enabled:
        return None
    return make_router_llm()


@lru_cache(maxsize=1)
def get_critic_dependency() -> Any | None:
    """ADR-0023: same "disabled means None" contract as
    `get_classifier_dependency` above — `critic_node` no-ops when this is
    `None`."""
    if not settings.critic_enabled:
        return None
    return make_critic_llm()


# ADR-0007 Day-17 amendment: `make_mcp_tools()` is `async def` (it awaits
# `MultiServerMCPClient.get_tools()`), so it can't be wrapped in a plain
# `@lru_cache` the way the sync dependencies above are — a tiny hand-rolled
# async-singleton cache instead. `_UNSET`, not `None`, marks "not built
# yet": `None` is itself a valid cached value (`settings.mcp_enabled=False`,
# make_mcp_tools()'s own "how to reverse" lever), so it must be
# distinguishable from "haven't tried yet".
_UNSET = object()
_tools_cache: dict[str, Any] | None | object = _UNSET


def get_checkpointer_dependency(request: Request) -> Any | None:
    """ADR-0024: the `AsyncPostgresSaver` built once in `lifespan()` below
    and stored on `app.state.checkpointer` — read via a dependency (not a
    module-level global) so tests can override it the same way as every
    other dependency here (`app.dependency_overrides`), e.g. with a
    fixture-scoped `InMemorySaver()` for a Postgres-free unit test of
    multi-turn history.

    `getattr(..., None)`, not `request.app.state.checkpointer` directly:
    `TestClient(app)` used WITHOUT its `with` context manager (this
    project's existing `tests/test_api.py` fixture) never runs `lifespan()`
    at all, so `app.state.checkpointer` is simply never set in that case —
    `None` reproduces this feature's pre-existing behaviour exactly
    (`build_graph(checkpointer=None)`: a `thread_id` is accepted but nothing
    durable happens with it), rather than an `AttributeError` breaking every
    existing test that doesn't care about persistence."""
    return getattr(request.app.state, "checkpointer", None)


async def get_tools_dependency() -> dict[str, Any] | None:
    """Built once per process (warmed up in `lifespan()` below, same
    "build once at startup" reasoning as `build_graph()`/
    `get_classifier_dependency()`) — the tool LIST is cheap to hold onto;
    `MultiServerMCPClient`'s per-call sessions mean each tool's own
    `.ainvoke()` still opens/tears down its own subprocess+session, so
    reusing this list across requests doesn't hold any connection open."""
    global _tools_cache
    if _tools_cache is _UNSET:
        _tools_cache = await make_mcp_tools()
    return _tools_cache


# ---------------------------------------------------------------------------
# Rate limiting — slowapi, keyed by API key when present, falling back to
# remote address when it's absent (an unauthenticated or wrong-key caller
# still gets IP-based throttling — see below for why this needs the
# request to reach the limiter check *before* `require_api_key` runs).
#
# Round-1 review finding: the original version used a per-route
# `@limiter.limit(...)` decorator. Verified in installed
# `slowapi/extension.py`'s `__limit_decorator`: that decorator wraps the
# endpoint *function*, so its rate-limit check only runs when the function
# is actually called — and FastAPI only calls a route's function after
# `solve_dependencies()` succeeds (verified in installed
# `fastapi/routing.py`'s `get_request_handler`). `require_api_key` raises
# `HTTPException` directly from a dependency, so a 401/403 short-circuits
# before the wrapped function (and therefore the rate check) ever runs —
# confirmed live: 30 rapid no-key requests produced 30 401s and zero 429s.
# Fix: `SlowAPIMiddleware` (installed `slowapi/middleware.py`) is real ASGI
# middleware — it runs before routing/dependency resolution for every
# non-exempt request, so it rate-limits a request regardless of whether
# auth later rejects it. `default_limits=[...]` (not `@limiter.limit`) is
# what a route gets checked against when no per-route decorator exists;
# verified in installed `slowapi/middleware.py`'s `_should_exempt`, which
# skips the middleware's check when a route already has a `@limiter.limit`
# decorator ("there is a decorator for this route, we let the decorator
# handle it") — so the decorator and the middleware are mutually exclusive
# paths, not stackable.
#
# `settings.rate_limit` is still passed as a callable (not the bare
# string), so it's re-read on every request rather than baked in at import
# time (unchanged reasoning: installed `slowapi/wrappers.py`'s
# `LimitGroup.__iter__` calls a callable `limit_value` fresh each check).
# ADR-0016: in-memory storage — single process only, see the `ponytail:`
# note below.
# ---------------------------------------------------------------------------
def _rate_limit_key(request: Request) -> str:
    return request.headers.get("X-API-Key") or get_remote_address(request)


# ponytail: in-memory limiter — resets on process restart, doesn't share
# state across multiple workers/replicas. Fine for this single-process
# deployment; swap `storage_uri="redis://..."` (slowapi's documented
# storage_uri kwarg) the day this runs as more than one process.
limiter = Limiter(key_func=_rate_limit_key, default_limits=[lambda: settings.rate_limit])


# ---------------------------------------------------------------------------
# Body-size cap (round-1 review finding #2): neither Starlette nor FastAPI
# caps request body size by default (verified: no such option in installed
# `starlette.applications.Starlette.__init__` or `fastapi.FastAPI.__init__`
# signatures) — an attacker with a valid key could send an arbitrarily
# large body that gets fully buffered into memory before Pydantic's
# `max_length` ever runs. `Content-Length` is checked here, before the body
# is read, so an oversized request is rejected without ever touching it.
# A body sent chunked (`Transfer-Encoding: chunked`, no `Content-Length`)
# is rejected too — 411, the standard HTTP code for "you must send
# Content-Length" (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/411)
# — rather than trusting an absent length. The reverse proxy (Caddy,
# ADR-0010) will add a second, earlier cap once this deploys behind one;
# this middleware is what the app relies on until then.
# ---------------------------------------------------------------------------
class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is None:
            if request.headers.get("transfer-encoding", "").lower() == "chunked":
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            return await call_next(request)
        try:
            length = int(content_length)
        except ValueError:
            return JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
        if length > settings.max_body_bytes:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)


def _sse(event: str, data: dict) -> str:
    """WHATWG SSE framing: `event:`/`data:` lines, blank line terminates
    and dispatches the event (see ADR-0016's references). A 5-line helper
    replaces `sse-starlette` entirely — verified via that package's own
    source that this is all it does for this project's single-path,
    no-reconnect streaming need (ADR-0016)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _run_graph_and_stream(
    graph: Any,
    initial_input: dict[str, Any] | Command,
    *,
    context: GraphContext,
    config: dict,
    thread_id: str,
) -> AsyncIterator[str]:
    """The shared `astream(..., stream_mode='updates')` -> SSE-event loop
    behind both `/ask` (`initial_input={"question": ...}`) and `/resume`
    (ADR-0025, `initial_input=Command(resume=...)`) — `Pregel.astream`'s
    installed signature accepts either as its first argument, so a fresh run
    and a resumed one are the same loop from this point on: the same node
    names, the same event shapes, the same exception handling. Never yields
    the question or full chunk text — only node names, article/recital
    anchors, attempt counts, and (on failure) a citation-error message built
    solely from anchors (see `CitationError`'s own docstring, state.py).

    ADR-0021: `guard_out` runs on every path AFTER `refuse`/`answer`/`hitl`
    (build.py), so the `final` event — and which `answer`/`refused` it
    carries — has to reflect whatever `guard_out` decided, not the state at
    the moment an earlier node fired. `current_answer`/`current_refused`
    track the latest values across node updates (`stream_mode='updates'`
    only ever hands this loop the KEYS one node changed, not the full
    state); `final` is emitted once, at `guard_out`, using whichever values
    are current by then.

    ADR-0025: a `{"__interrupt__": (Interrupt(...), ...)}` chunk (verified
    against the installed `langgraph` pregel loop, matching Context7's
    documented shape) is `hitl_node` pausing on a low-confidence critic
    score — emits `interrupt` and ends the stream with no `final` event,
    the same "stream ends without a final answer" contract a citation/
    output-guard failure already gives via `error` below.

    ADR-0025 round 2 (SHOULD 1): the `interrupt` SSE event carries ONLY
    `{thread_id, interrupt_id, status: "under_review"}` — never the draft
    answer, confidence, or reasoning. This is the same channel `/ask`'s
    caller reads (`docs/ARCHITECTURE.md` §5: the end user gets "under
    review", nothing more); an operator reviews the full payload separately
    (`graph.aget_state()`, which the CLI `resume` command prints before
    applying a decision) — a distinct identity from "whoever holds the
    shared API key", in intent if not yet in enforcement (ADR-0016's still-
    open gap). The full payload still lives inside `interrupt(...)` itself
    (`hitl_node`, graph/nodes.py) — this is a narrower SSE projection of
    it, not a narrower checkpoint.

    ADR-0028: the whole `graph.astream(...)` loop below runs inside
    `asyncio.timeout(settings.request_timeout_s)` — ONE deadline for the
    whole request (every node this run visits), distinct from any single
    LLM call's own per-call timeout (`answer_timeout_s` etc., settings.py).
    A `TimeoutError` raised inside (stdlib — `asyncio.TimeoutError` has been
    the same class as the builtin since Python 3.11) is caught below and
    turned into a typed `{"type": "timeout"}` error event, same "never a
    stack trace, never a silently-dropped connection" convention every
    other `error` event here already follows. Shared by `/resume` for
    free: `_stream_resume` calls this same function, so a resumed run gets
    the same deadline with no separate wiring.

    Resuming (`initial_input` is a `Command`) starts this SAME loop
    part-way through the graph — `answer_node`/`critic_node` already ran
    (and were streamed) in the EARLIER call that paused, so THIS stream
    never re-emits their updates. `current_answer`/`current_refused` are
    seeded from the paused snapshot's own values (verified live: an
    `approve` resume, which returns `Command(goto="guard_out")` with no
    `update`, streams only `{"hitl": None}` then `guard_out`'s own update —
    neither carries `answer` — so without this seed `current_answer` would
    still be `None` when `final` tries to `.model_dump()` it)."""
    started = time.monotonic()
    trace_emitted = False
    current_answer = None
    current_refused = False
    # ADR-0028: mirrors `current_refused`'s seeding above — a resume whose
    # paused turn had already degraded (can't happen today, `hitl_node`
    # never pauses on a degraded turn since `critic_node` no-ops on it, but
    # seeding it the same way costs nothing and avoids a silent assumption).
    current_degraded = False
    if isinstance(initial_input, Command):
        snapshot = await graph.aget_state(config)
        current_answer = snapshot.values.get("answer")
        current_refused = snapshot.values.get("refused", False)
        current_degraded = snapshot.values.get("degraded", False)
    # ADR-0028 round 2 (reviewer BLOCKER 1): a plain `async with
    # asyncio.timeout(...): async for update in graph.astream(...): ...
    # yield ...` wraps a `yield` INSIDE the timeout scope — when the
    # generator is parked at that `yield` waiting on a SLOW CONSUMER (a
    # slow SSE client, socket backpressure), `Task.cancel()` still fires at
    # the deadline (it targets the Task, not a specific frame), but the
    # `CancelledError` lands wherever the Task is actually suspended right
    # then — the consumer's own await (Starlette's `await send(...)`), not
    # here. `except TimeoutError:` below never runs, no `{"type":
    # "timeout"}` event is ever sent, and the un-`aclose()`d generator gets
    # `GeneratorExit` thrown into it later at GC/shutdown — inside a scope
    # whose deadline already elapsed, which raised an actual
    # `RuntimeError: async generator ignored GeneratorExit` in testing
    # (reviewer-reproduced, not theoretical). Fix: put `asyncio.timeout()`
    # around ONLY the "get the next update" step (`anext`) each iteration —
    # never around a `yield`.
    #
    # `used` (not a fixed `deadline = start + budget`, which was ROUND 2's
    # first, still-wrong attempt): a plain wall-clock deadline keeps
    # ticking during the gap between one `yield` and the consumer's next
    # pull too — it can't tell "the graph was slow" apart from "the
    # consumer was slow to ask for the next chunk", so it silently
    # re-introduces the exact bug this fix exists to remove. `used`
    # instead accumulates ONLY the measured duration of each individual
    # `anext()` call (the `finally` below) — time spent between calls
    # (however long a slow consumer takes) is never added to it. A slow
    # CLIENT therefore never counts against this budget (that's ASGI/
    # proxy territory, a separate concern) — only slow GRAPH work does,
    # cumulatively across every `anext` call, never reset per chunk.
    used = 0.0
    astream_iter = graph.astream(
        initial_input, context=context, config=config, stream_mode="updates"
    ).__aiter__()
    try:
        while True:
            remaining = settings.request_timeout_s - used
            if remaining <= 0:
                raise TimeoutError
            started_wait = time.monotonic()
            try:
                async with asyncio.timeout(remaining):
                    update = await anext(astream_iter)
            except StopAsyncIteration:
                break
            finally:
                used += time.monotonic() - started_wait
            if "__interrupt__" in update:
                payload = update["__interrupt__"][0]
                draft = payload.value
                # Full draft/confidence/reasoning logged server-side only
                # (an operator reading logs/traces, not the SSE response) —
                # the SSE event below is deliberately narrower (SHOULD 1).
                logger.info(
                    "node=hitl interrupted confidence=%s elapsed_ms=%d",
                    draft["confidence"],
                    int((time.monotonic() - started) * 1000),
                )
                yield _sse(
                    "interrupt",
                    {
                        "thread_id": thread_id,
                        "interrupt_id": payload.id,
                        "status": "under_review",
                    },
                )
                tracing.score("interrupted", 1.0, tracing.current_trace_id(config))
                return
            for node_name, node_update in update.items():
                elapsed_ms = int((time.monotonic() - started) * 1000)
                # ADR-0023: `router_node`/`critic_node` return `{}` when
                # disabled (`GraphContext.router`/`.critic` is `None`) —
                # verified live that LangGraph normalises an empty-dict node
                # return to `None` in the `stream_mode="updates"` payload, so
                # this is what keeps `node_update.get(...)` below safe rather
                # than raising `AttributeError` on `None`.
                node_update = node_update or {}
                if node_name == "guard_in":
                    guard = node_update["guard"]
                    # ADR-0020: entity TYPE names only, e.g. ["PERSON",
                    # "EMAIL_ADDRESS"] — `node_update` only carries
                    # `pii_entities` at all when `guard_in_node` actually
                    # redacted something (state.py's NotRequired key), so
                    # `.get(..., ())` is what makes this key ALWAYS present
                    # in the event (empty list = nothing found), matching
                    # `final`'s "refused" convention below.
                    pii_entities = node_update.get("pii_entities", ())
                    logger.info(
                        "node=%s flagged=%s score=%s elapsed_ms=%d",
                        node_name,
                        guard.flagged,
                        guard.score,
                        elapsed_ms,
                    )
                    yield _sse(
                        "node",
                        {
                            "node": node_name,
                            "flagged": guard.flagged,
                            "score": guard.score,
                            "reasons": list(guard.reasons),
                            "pii": list(pii_entities),
                        },
                    )
                    # `guard_in` is the real first node now (ADR-0018) — the
                    # earliest-trace-id emission (see the "retrieve" branch's
                    # comment, unchanged below) moves here so a refused
                    # request (which never reaches "retrieve") still gets a
                    # `trace` event.
                    if not trace_emitted:
                        trace_id = tracing.current_trace_id(config)
                        if trace_id is not None:
                            yield _sse("trace", {"trace_id": trace_id})
                        trace_emitted = True
                elif node_name == "router":
                    # ADR-0023: absent when the router is disabled
                    # (`GraphContext.router=None`) — no event at all in that
                    # case, same "invisible when off" behaviour every other
                    # optional guard already has. `verdict.reason` is free
                    # model text (router.py's `RouterVerdict`) — logged only,
                    # never forwarded in the SSE payload (ADR-0023's leak
                    # guard: only policy-approved fields reach the client).
                    verdict = node_update.get("router")
                    if verdict is not None:
                        logger.info(
                            "node=%s regulation=%s elapsed_ms=%d",
                            node_name,
                            verdict.regulation,
                            elapsed_ms,
                        )
                        yield _sse("node", {"node": node_name, "regulation": verdict.regulation})
                elif node_name == "refuse":
                    # No preceding "node" event for `refuse` itself — the
                    # `guard_in` event above already told the client this
                    # was flagged. `final` no longer fires here (ADR-0021):
                    # `guard_out` runs next on this same path and is what
                    # actually emits it, once it's had its say.
                    current_answer = node_update["answer"]
                    current_refused = True
                    logger.info("node=%s elapsed_ms=%d", node_name, elapsed_ms)
                    tracing.score("refused", 1.0, tracing.current_trace_id(config))
                elif node_name == "retrieve":
                    articles = node_update.get("articles") or []
                    recitals = node_update.get("recitals") or []
                    logger.info(
                        "node=%s articles=%d recitals=%d elapsed_ms=%d",
                        node_name,
                        len(articles),
                        len(recitals),
                        elapsed_ms,
                    )
                    yield _sse(
                        "node",
                        {
                            "node": node_name,
                            "articles": [a.anchor for a in articles],
                            "recitals": [r.anchor for r in recitals],
                        },
                    )
                elif node_name == "answer":
                    attempts = node_update.get("attempts")
                    citation_error = node_update.get("citation_error") is not None
                    logger.info(
                        "node=%s attempt=%s citation_error=%s elapsed_ms=%d",
                        node_name,
                        attempts,
                        citation_error,
                        elapsed_ms,
                    )
                    yield _sse(
                        "node",
                        {
                            "node": node_name,
                            "attempt": attempts,
                            "citation_error": citation_error,
                        },
                    )
                    answer = node_update.get("answer")
                    if answer is not None:
                        # `final` no longer fires here either (ADR-0021,
                        # same reasoning as the `refuse` branch above) —
                        # `guard_out` runs next and emits it.
                        current_answer = answer
                        current_refused = False
                        # ADR-0028: `answer_node` sets "degraded" only on
                        # its outage-fallback branch — absent (a normal
                        # validated answer) defaults False here the same
                        # way every other optional key in this loop does.
                        current_degraded = node_update.get("degraded", False)
                        if current_degraded:
                            # A degraded turn didn't validate any
                            # citations — the "counted, not hidden" signal
                            # lesson 23 calls for, distinct from the
                            # citation_valid score below.
                            tracing.score("degraded", 1.0, tracing.current_trace_id(config))
                        else:
                            # First quality signal on the dashboard (ADR-0009
                            # amendment, Day 10 builds eval scores on top of
                            # this): a validated answer scores 1.0.
                            tracing.score("citation_valid", 1.0, tracing.current_trace_id(config))
                elif node_name == "critic":
                    # ADR-0023: absent when the critic is disabled
                    # (`GraphContext.critic=None`) or on a refused path (the
                    # critic never runs there — build.py only wires it onto
                    # `answer`'s success branch). `verdict.reasoning` is free
                    # model text, same leak-guard rule as `router` above —
                    # logged and scored, never forwarded in the SSE payload.
                    verdict = node_update.get("critic")
                    if verdict is not None:
                        logger.info(
                            "node=%s faithful=%s confidence=%s elapsed_ms=%d",
                            node_name,
                            verdict.faithful,
                            verdict.confidence,
                            elapsed_ms,
                        )
                        yield _sse(
                            "node",
                            {
                                "node": node_name,
                                "faithful": verdict.faithful,
                                "confidence": verdict.confidence,
                            },
                        )
                        trace_id = tracing.current_trace_id(config)
                        tracing.score("critic_faithful", 1.0 if verdict.faithful else 0.0, trace_id)
                        tracing.score("critic_confidence", verdict.confidence, trace_id)
                        if verdict.error:
                            # ADR-0025 round 2 (BLOCKER 1): a critic-tier
                            # OUTAGE, distinct from a genuine low-confidence
                            # verdict — `hitl_node` will NOT pause for this
                            # (see its own docstring), so this score is the
                            # visible record that LLM-judge coverage was
                            # lost for this request, mirroring the router's
                            # own fail-open logging (router.py).
                            tracing.score("critic_unavailable", 1.0, trace_id)
                elif node_name == "hitl":
                    # ADR-0025: only reached on a RESUME (a fresh run either
                    # pauses via `__interrupt__` above, or passes through
                    # with no node update at all — `hitl_node`'s no-op
                    # `return {}}` normalises to `None`/`{}` the same way
                    # `router`/`critic` already do when disabled). `approve`
                    # returns `Command(goto="guard_out")` with no `update`
                    # (verified live: streams as `{"hitl": None}`) — leaves
                    # `current_answer`/`current_refused` at whatever they
                    # were seeded to above. `edit`/`reject` DO write
                    # `answer` (and `reject` writes `refused=True`) —
                    # captured here the same way `refuse`'s block above
                    # does, no new event vocabulary needed.
                    if node_update.get("answer") is not None:
                        current_answer = node_update["answer"]
                    if "refused" in node_update:
                        current_refused = node_update["refused"]
                    logger.info("node=%s resumed elapsed_ms=%d", node_name, elapsed_ms)
                elif node_name == "guard_out":
                    # ADR-0021: the final gate, reached on every path. Its
                    # own update only carries "answer"/"refused" when it
                    # REWROTE them (a policy-violation block) — otherwise
                    # `current_answer`/`current_refused` are already
                    # whatever `refuse`/`answer` set above, which is exactly
                    # "pass through unchanged".
                    verdict = node_update["output_guard"]
                    if node_update.get("answer") is not None:
                        current_answer = node_update["answer"]
                        current_refused = node_update.get("refused", True)
                        # ADR-0028: `guard_out_node`'s rewrite branch
                        # (a policy-violation block) never sets
                        # "degraded" — defaulting False here is what
                        # correctly clears a stale True from the
                        # `answer` branch above once a degraded fallback
                        # gets rewritten into a plain refusal.
                        current_degraded = node_update.get("degraded", False)
                    logger.info(
                        "node=%s ok=%s reason=%s elapsed_ms=%d",
                        node_name,
                        verdict.ok,
                        verdict.reason,
                        elapsed_ms,
                    )
                    yield _sse(
                        "node", {"node": node_name, "ok": verdict.ok, "reason": verdict.reason}
                    )
                    if not verdict.ok:
                        tracing.score("output_blocked", 1.0, tracing.current_trace_id(config))
                    yield _sse(
                        "final",
                        {
                            **current_answer.model_dump(),
                            "refused": current_refused,
                            "degraded": current_degraded,
                        },
                    )
    except TimeoutError:
        # ADR-0028 round 2: either `anext(astream_iter)`'s own
        # `asyncio.timeout(remaining)` expired (that one node call was
        # slow), or `remaining <= 0` was already true at the top of an
        # iteration (the SUM of every prior `anext()`'s own measured
        # duration — `used` — already exhausted the budget, even though no
        # single call individually did) — both raise `TimeoutError` from a
        # point that's INSIDE this `try`, never from a `yield`, so this
        # clause always actually runs (round 1's first-attempt bug: a slow
        # CONSUMER, not slow graph work, could previously starve this
        # clause entirely). Same "type only, never a stack trace"
        # convention as every other `error` event here.
        logger.warning("request_timeout elapsed_ms=%d", int((time.monotonic() - started) * 1000))
        yield _sse("error", {"type": "timeout"})
        tracing.score("timeout", 1.0, tracing.current_trace_id(config))
    except OperationalError:
        # ADR-0028: the DB is unreachable — a distinct, actionable signal
        # from the generic `internal_error` catch-all below (an operator
        # reading `dependency_unavailable` knows to check Postgres, not to
        # start reading a stack trace for a code bug). Never logged with
        # `.exception()` here: `OperationalError`'s own message can embed
        # the connection string (the "no PII in logs" hard rule already
        # applies to secrets too), so only the elapsed time is logged.
        logger.error(
            "dependency_unavailable elapsed_ms=%d", int((time.monotonic() - started) * 1000)
        )
        yield _sse("error", {"type": "dependency_unavailable"})
        tracing.score("dependency_unavailable", 1.0, tracing.current_trace_id(config))
    except OutputGuardError as exc:
        # `guard_out_node` (graph/nodes.py) raised — an internal invariant
        # broke (a citation `answer_node` claims it already validated, or a
        # refusal that somehow fails its own fixed-text checks), not a
        # policy violation. `str(exc)` is the reason code only (guards/
        # output.py's `OutputGuardError` docstring) — safe to send, never
        # the answer text.
        logger.error(
            "guard_out invariant failed reason=%s elapsed_ms=%d",
            str(exc),
            int((time.monotonic() - started) * 1000),
        )
        yield _sse("error", {"type": "output_guard_error", "reason": str(exc)})
        tracing.score("output_guard_error", 1.0, tracing.current_trace_id(config))
    except CitationError as exc:
        # Reached only after `fail_node` (build.py) raises — the retry
        # loop's two attempts both failed citation validation. `str(exc)`
        # is safe to send: `CitationError`'s message is built only from
        # citation/anchor data, never the question (state.py's docstring).
        logger.info("citation_error elapsed_ms=%d", int((time.monotonic() - started) * 1000))
        yield _sse("error", {"type": "citation_error", "message": str(exc)})
        tracing.score("citation_valid", 0.0, tracing.current_trace_id(config))
    except Exception:
        # Anything else (LLM/DB failure, a bug) — log server-side with a
        # stack trace but tell the client nothing beyond "something broke".
        # No question text is passed to `logger.exception` here, satisfying
        # the "no PII in logs" hard rule.
        logger.exception(
            "internal_error during /ask stream, elapsed_ms=%d",
            int((time.monotonic() - started) * 1000),
        )
        yield _sse("error", {"type": "internal_error"})
    finally:
        # ADR-0028 round 2: `async for`/manual iteration never auto-closes
        # an async generator on early exit (a `return`, a `break`, or an
        # exception escaping the loop) — without this, `astream_iter`
        # (LangGraph's own `Pregel.astream` generator) sits open until GC
        # or `loop.shutdown_asyncgens()` eventually throws `GeneratorExit`
        # into it, which reviewer testing showed can surface as a real
        # `RuntimeError: async generator ignored GeneratorExit` at
        # shutdown. Runs on EVERY exit path (normal completion, the
        # `return` on `__interrupt__`, and all five `except` clauses
        # above) — `aclose()` on an already-exhausted iterator is a no-op.
        await astream_iter.aclose()


def _build_run_config(thread_id: str) -> dict:
    """One config per request: `run_config()` builds a fresh CallbackHandler
    (or `[]` when tracing is disabled, tracing.py) and a request-scoped
    session id, so concurrent requests never share a handler instance (see
    tracing.current_trace_id's docstring on why that matters).
    `configurable.thread_id` (ADR-0024) rides alongside it — LangGraph reads
    this key straight off the same config dict, no separate `config=`
    argument needed. Shared by `/ask` and `/resume` (ADR-0025) — identical
    shape either way."""
    config = tracing.run_config(session_id=uuid.uuid4().hex)
    config["configurable"] = {"thread_id": thread_id}
    return config


async def _stream_answer(
    question: str,
    *,
    session: Session,
    embeddings: Embeddings,
    llm: Any,
    classifier: Any | None,
    router: Any | None,
    critic: Any | None,
    tools: dict[str, Any] | None,
    checkpointer: Any | None,
    thread_id: str,
) -> AsyncIterator[str]:
    """The `/ask` response body. `classifier`: ADR-0019's layer-2 guard,
    `None` when disabled (`get_classifier_dependency` above). `router`/
    `critic`: ADR-0023's two cheap-LLM calls, `None` when disabled — same
    contract. `checkpointer` is `None` when `lifespan()` hasn't run (e.g. a
    bare `TestClient(app)`, see `get_checkpointer_dependency`'s docstring),
    which reproduces today's stateless-per-call behaviour exactly. The
    `thread` event fires FIRST, before the graph even starts, since the
    caller already knows `thread_id` by then (server-issued or client-
    supplied, validated) — this is what lets a client that omitted
    `thread_id` learn the one the server picked, in time to send it back on
    the next call (or to resume a pause, ADR-0025)."""
    yield _sse("thread", {"thread_id": thread_id})
    graph = build_graph(checkpointer=checkpointer)
    context = GraphContext(
        session=session,
        embeddings=embeddings,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools,
    )
    config = _build_run_config(thread_id)
    async for chunk in _run_graph_and_stream(
        graph, {"question": question}, context=context, config=config, thread_id=thread_id
    ):
        yield chunk


async def _require_paused_thread(graph: Any, thread_id: str, interrupt_id: str) -> None:
    """ADR-0025: `POST /resume`'s pre-flight check. `graph.aget_state(...)`
    on a thread_id with no checkpointed state at all returns an EMPTY
    snapshot (`values == {}`) rather than raising — verified live against
    the installed `langgraph` package — so "no values" is what actually
    distinguishes 404 (never existed) from 409 (exists, but isn't currently
    paused: `snapshot.next`/`snapshot.interrupts` both empty means no
    pending `interrupt()` to resume). A graph compiled with no checkpointer
    at all (`checkpointer=None`, e.g. `lifespan()` hasn't run) raises
    `ValueError` from `aget_state` itself — no persistence means no thread
    can be "known" either, so that's the same 404.

    ADR-0025 round 2 (BLOCKER 2): also checks `interrupt_id` — the caller's
    claimed pending-interrupt id — against `snapshot.interrupts[0].id`, the
    ACTUAL one `aget_state` reports right now. A mismatch is 409, same as
    "not paused": it means the caller is holding a STALE reference (a later
    `/ask` re-paused this thread on a different question, or a previous
    `/resume` already resolved this exact interrupt) — never silently
    applied to a draft the caller never actually reviewed."""
    try:
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown thread_id") from None
    if not snapshot.values:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown thread_id")
    if not snapshot.next or not snapshot.interrupts:
        raise HTTPException(status.HTTP_409_CONFLICT, "thread is not paused")
    if snapshot.interrupts[0].id != interrupt_id:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "interrupt_id does not match the pending review"
        )


async def _reject_if_paused(graph: Any, thread_id: str) -> None:
    """ADR-0025 round 2 (BLOCKER 2): `/ask`'s pre-flight check on a
    CLIENT-SUPPLIED (existing) `thread_id` — reproduced live before this
    fix: calling `graph.astream({"question": ...}, config=same_thread_id)`
    on a thread currently paused at `hitl` is NOT rejected by LangGraph —
    it happily starts a brand-new run from `START`, and the new run's
    checkpoint OVERWRITES the paused one (the original draft, critic
    verdict, and interrupt vanish with no error). A review-gating feature
    cannot let a second `/ask` silently supersede a pending review — 409,
    the same status `/resume` already uses for "not currently paused"
    (the mirror-image condition). `ValueError` (no checkpointer at all)
    means nothing could possibly be paused — nothing to reject."""
    try:
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    except ValueError:
        return
    if snapshot.next and snapshot.interrupts:
        raise HTTPException(status.HTTP_409_CONFLICT, "thread is awaiting review; call /resume")


async def _stream_resume(
    decision: str,
    edited_answer: str | None,
    *,
    session: Session,
    embeddings: Embeddings,
    llm: Any,
    classifier: Any | None,
    router: Any | None,
    critic: Any | None,
    tools: dict[str, Any] | None,
    checkpointer: Any | None,
    thread_id: str,
) -> AsyncIterator[str]:
    """The `/resume` response body (ADR-0025): resumes a run `hitl_node`
    paused, via `Command(resume=...)` — `hitl_node` re-executes from the top
    (LangGraph's documented resume semantics), reads the decision back out
    of `interrupt()`'s return value, and routes to `guard_out` either way
    (approve/edit unchanged or rewritten answer; reject the fixed refusal).
    From there this streams the remainder (`guard_out` -> `final`) through
    the SAME `_run_graph_and_stream` loop `/ask` uses — identical event
    shapes, no separate "resume" event vocabulary to maintain.

    No `thread` event here (unlike `_stream_answer`): the caller already
    knows `thread_id` — they supplied it in the request body.

    `GraphContext` is rebuilt fresh from the request's own dependencies,
    exactly like `/ask` — it is NOT part of checkpointed state (state.py's
    module docstring: a DB session/LLM client isn't serialisable or meant to
    be persisted), so a resume needs its own live dependencies the same way
    a fresh run does, even though the guard_out-only remainder rarely uses
    most of them. Pre-flight validation (`_require_paused_thread`) already
    ran in the route handler, BEFORE this generator was ever constructed —
    an `HTTPException` raised from inside a `StreamingResponse` body would
    arrive after a 200 and headers were already sent, not as the 404/409 a
    client needs to see."""
    graph = build_graph(checkpointer=checkpointer)
    context = GraphContext(
        session=session,
        embeddings=embeddings,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools,
    )
    config = _build_run_config(thread_id)
    # ADR-0025 deliverable D: minimal tracing signal for which decision an
    # operator made — mirrors the existing one-signal-per-`tracing.score()`
    # call convention (`refused`, `citation_valid`, ...).
    tracing.score(f"hitl_{decision}", 1.0, tracing.current_trace_id(config))
    resume_value = {"decision": decision, "edited_answer": edited_answer}
    async for chunk in _run_graph_and_stream(
        graph, Command(resume=resume_value), context=context, config=config, thread_id=thread_id
    ):
        yield chunk


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ADR-0020: the logging backstop (defence-in-depth only — the primary
    # PII control is guard_in's redaction, see logging_filter.py's module
    # docstring) — installed once at process startup, before any request
    # can log anything.
    install_pii_scrub()
    # ADR-0024: the durable-state pool+saver, opened once for the process's
    # whole lifetime (`build_checkpointer()`'s `async with` closes the pool
    # on the way out of THIS function, i.e. on shutdown) and stored on
    # `app.state` so `get_checkpointer_dependency` above can read it per
    # request. A Postgres outage at startup means this raises and the app
    # never comes up — deliberate, same "fail before any LLM spend" posture
    # ADR-0001/the persistence lesson already call for, not a new one.
    async with build_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer
        # Build the graph once at startup rather than on the first request
        # — `build_graph()` is itself `@lru_cache(maxsize=1)` (build.py), so
        # this call just moves the one-time build earlier; it's a no-op if
        # a request already triggered it first.
        build_graph(checkpointer=checkpointer)
        # Same reasoning for the classifier client (ADR-0019) — a
        # request-time first-call cost would otherwise land on whichever
        # caller happens to be first, instead of on startup where a slow
        # LLM client construction belongs.
        get_classifier_dependency()
        # ADR-0023: same reasoning again for the router/critic clients.
        get_router_dependency()
        get_critic_dependency()
        # ADR-0007 Day-17 amendment: same reasoning again — spawning the
        # MCP server subprocess and loading its tools happens once here, at
        # startup, not on whichever request happens to arrive first.
        await get_tools_dependency()
        yield
    # `shutdown()`, not `flush()`: this runs once as the process exits, so it
    # should also stop Langfuse's background consumer threads (tracing.py) —
    # a no-op when tracing is disabled.
    # to_thread: shutdown() flushes over the network with bounded retries —
    # keep that off the event loop so other in-flight responses can finish.
    await asyncio.to_thread(tracing.shutdown)


# docs_url/redoc_url=None: no public API docs surface yet (ADR-0016) — this
# is a single-endpoint internal API, not something meant to be browsed.
app = FastAPI(title="Compliance Copilot", docs_url=None, redoc_url=None, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# SlowAPIMiddleware (not the `@limiter.limit` decorator, see the "Rate
# limiting" comment above) — real ASGI middleware, runs before routing/auth
# for every request, so unauthenticated/wrong-key traffic is throttled too.
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
# No CORSMiddleware added anywhere in this module — FastAPI has no CORS
# unless explicitly configured, so the absence of that middleware IS the
# default-off state (ADR-0016).


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default handler for this exception includes an `"input"`
    key per error — the raw value that failed validation, i.e. the
    question itself for a too-short/too-long `AskRequest.question`. That
    would echo the question back in a 422 body, which the trust-boundary
    rule (ADR-0006: error bodies never echo the question) forbids. Strip
    it down to `type`/`loc`/`msg` only."""
    errors = [{"type": e["type"], "loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/healthz")
@limiter.exempt
async def healthz() -> dict:
    """No auth, no DB, no LLM call, no rate limit (`@limiter.exempt`) — a
    liveness probe must not depend on anything that can fail independently
    of the process being alive, or be throttled alongside real traffic."""
    return {"status": "ok"}


@app.get("/readyz")
@limiter.exempt
async def readyz(session: Session = Depends(get_session)) -> JSONResponse:
    """ADR-0028: **readiness**, not liveness — "should traffic be routed to
    this instance right now," a different question from `/healthz`'s
    "should this container be restarted." Runs a trivial `SELECT 1` through
    the same `get_session` dependency every real request uses, so a DB
    outage this route can't reach is exactly the DB outage `/ask` couldn't
    reach either. No auth (same reasoning as `/healthz` — an orchestrator's
    probe, not a client of the product) and no rate limit (`@limiter.exempt`,
    mirroring `/healthz`)."""
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.post("/ask")
async def ask(
    req: AskRequest,
    _auth: None = Depends(require_api_key),
    session: Session = Depends(get_session),
    embeddings: Embeddings = Depends(get_embeddings_dependency),
    llm: Any = Depends(get_llm_dependency),
    classifier: Any | None = Depends(get_classifier_dependency),
    router: Any | None = Depends(get_router_dependency),
    critic: Any | None = Depends(get_critic_dependency),
    tools: dict[str, Any] | None = Depends(get_tools_dependency),
    checkpointer: Any | None = Depends(get_checkpointer_dependency),
) -> StreamingResponse:
    # ADR-0024: mint a fresh thread on the client's first turn; a
    # client-supplied `thread_id` already passed `AskRequest`'s UUID4
    # validator above.
    thread_id = req.thread_id or str(uuid.uuid4())
    if req.thread_id:
        # ADR-0025 round 2 (BLOCKER 2): only a CLIENT-SUPPLIED thread_id can
        # possibly be paused already — a freshly-minted uuid4 has no prior
        # state, skip the round-trip for that (the common) case. Validated
        # BEFORE the streaming response starts, same reasoning `/resume`'s
        # `_require_paused_thread` call already documents.
        await _reject_if_paused(build_graph(checkpointer=checkpointer), thread_id)
    generator = _stream_answer(
        req.question,
        session=session,
        embeddings=embeddings,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        # Anti-buffering: `no-cache` tells any HTTP cache not to store this;
        # `X-Accel-Buffering: no` is an Nginx-specific hint to not buffer
        # the response before forwarding it — harmless on Caddy (this
        # project's deploy target, docs/ARCHITECTURE.md §6), future-proofs
        # an Nginx swap. Same two headers `sse-starlette` sets by default
        # (ADR-0016), reused here without the dependency.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/resume")
async def resume(
    req: ResumeRequest,
    _auth: None = Depends(require_api_key),
    session: Session = Depends(get_session),
    embeddings: Embeddings = Depends(get_embeddings_dependency),
    llm: Any = Depends(get_llm_dependency),
    classifier: Any | None = Depends(get_classifier_dependency),
    router: Any | None = Depends(get_router_dependency),
    critic: Any | None = Depends(get_critic_dependency),
    tools: dict[str, Any] | None = Depends(get_tools_dependency),
    checkpointer: Any | None = Depends(get_checkpointer_dependency),
) -> StreamingResponse:
    """ADR-0025: resumes a run `hitl_node` paused on a low-confidence critic
    score (the `interrupt` SSE event `/ask` emitted, carrying `thread_id`).
    Same auth (`X-API-Key`) and rate limit (`SlowAPIMiddleware`, ADR-0016)
    as `/ask` — every dependency here mirrors that route's, since resuming
    can run the same LLM-backed nodes a fresh run can (`guard_out` is
    deterministic, but the graph is rebuilt the same way regardless).

    404 unknown `thread_id` / 409 not currently paused / 409 `interrupt_id`
    mismatch (round 2, BLOCKER 2 — a stale reference to an already-resolved
    or already-superseded pause), all raised by `_require_paused_thread`
    before anything else runs.

    **Known gap (ADR-0016, still open — not solved here):** this API has
    one shared `X-API-Key` across every caller, so any key holder who
    knows/guesses a valid `thread_id`+`interrupt_id` pair can resume it —
    there is no binding between the key that started a run and the key
    allowed to resolve its pause. Accepted, not fixed, per ADR-0016/
    ADR-0024's own precedent for the same class of gap."""
    # Validated HERE, before the streaming response starts (not inside the
    # generator, see `_stream_resume`'s docstring) — a 404/409 has to be a
    # real HTTP status, not something raised after a 200 already went out.
    await _require_paused_thread(
        build_graph(checkpointer=checkpointer), req.thread_id, req.interrupt_id
    )
    generator = _stream_resume(
        req.decision,
        req.edited_answer,
        session=session,
        embeddings=embeddings,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools,
        checkpointer=checkpointer,
        thread_id=req.thread_id,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
