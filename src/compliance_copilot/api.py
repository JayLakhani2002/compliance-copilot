# src/compliance_copilot/api.py — the HTTP surface for the graph
# (docs/ARCHITECTURE.md §5-6, ADR-0008, ADR-0016). One route, `/ask`: an
# API-key-gated, rate-limited, streaming (SSE) wrapper around
# `graph.astream(...)` (ADR-0001's `retrieve -> answer` graph, unchanged).
# `/healthz` is unauthenticated and does no DB/LLM work, for a container
# orchestrator's liveness probe.
#
# Why one module, not a package: today's surface is one real endpoint plus
# a health check — splitting routes/deps/schemas into separate files would
# be indirection with nothing on the other side of it yet (ponytail). Split
# it the day a second real endpoint (e.g. the HITL resume endpoint
# docs/ARCHITECTURE.md §5 sketches) actually lands.
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
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from compliance_copilot import embeddings as embeddings_module
from compliance_copilot import tracing
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
) -> AsyncIterator[str]:
    """The `/ask` response body: runs the compiled graph via `astream(...,
    stream_mode='updates')`, translating each node's partial state into an
    SSE event. Never yields the question or full chunk text — only node
    names, article/recital anchors, attempt counts, and (on failure) a
    citation-error message built solely from anchors (see
    `CitationError`'s own docstring, state.py).

    `classifier`: ADR-0019's layer-2 guard, `None` when disabled
    (`get_classifier_dependency` above). `router`/`critic`: ADR-0023's two
    new cheap-LLM calls, `None` when disabled (`get_router_dependency`/
    `get_critic_dependency` above) — same contract.

    ADR-0021: `guard_out` now runs on every path AFTER `refuse`/`answer`
    (build.py), so the `final` event — and which `answer`/`refused` it
    carries — has to reflect whatever `guard_out` decided, not the state at
    the moment `refuse`/`answer` fired. `current_answer`/`current_refused`
    track the latest values across node updates (`stream_mode='updates'`
    only ever hands this loop the KEYS one node changed, not the full
    state); `final` is emitted once, at `guard_out`, using whichever values
    are current by then."""
    graph = build_graph()
    context = GraphContext(
        session=session,
        embeddings=embeddings,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools,
    )
    started = time.monotonic()
    # One config per request: `run_config()` builds a fresh CallbackHandler
    # (or `[]` when tracing is disabled, tracing.py) and a request-scoped
    # session id, so concurrent `/ask` calls never share a handler instance
    # (see tracing.current_trace_id's docstring on why that matters).
    config = tracing.run_config(session_id=uuid.uuid4().hex)
    trace_emitted = False
    current_answer = None
    current_refused = False
    try:
        async for update in graph.astream(
            {"question": question}, context=context, config=config, stream_mode="updates"
        ):
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
                        {"node": node_name, "attempt": attempts, "citation_error": citation_error},
                    )
                    answer = node_update.get("answer")
                    if answer is not None:
                        # `final` no longer fires here either (ADR-0021,
                        # same reasoning as the `refuse` branch above) —
                        # `guard_out` runs next and emits it.
                        current_answer = answer
                        current_refused = False
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
                    yield _sse("final", {**current_answer.model_dump(), "refused": current_refused})
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ADR-0020: the logging backstop (defence-in-depth only — the primary
    # PII control is guard_in's redaction, see logging_filter.py's module
    # docstring) — installed once at process startup, before any request
    # can log anything.
    install_pii_scrub()
    # Build the graph once at startup rather than on the first request —
    # `build_graph()` is itself `@lru_cache(maxsize=1)` (build.py), so this
    # call just moves the one-time build earlier; it's a no-op if a request
    # already triggered it first.
    build_graph()
    # Same reasoning for the classifier client (ADR-0019) — a request-time
    # first-call cost would otherwise land on whichever caller happens to
    # be first, instead of on startup where a slow LLM client construction
    # belongs.
    get_classifier_dependency()
    # ADR-0023: same reasoning again for the router/critic clients.
    get_router_dependency()
    get_critic_dependency()
    # ADR-0007 Day-17 amendment: same reasoning again — spawning the MCP
    # server subprocess and loading its tools happens once here, at
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
) -> StreamingResponse:
    generator = _stream_answer(
        req.question,
        session=session,
        embeddings=embeddings,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools,
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
