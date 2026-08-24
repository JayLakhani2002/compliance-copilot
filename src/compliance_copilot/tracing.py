# src/compliance_copilot/tracing.py — optional Langfuse tracing for the
# `retrieve -> answer` graph (ADR-0009, amended 2026-08-24). Sits beside
# `settings.py` in the dependency graph: `api.py` and `cli.py` call
# `run_config()` to get a LangChain `config` dict to pass into
# `graph.astream()`/`graph.invoke()`.
#
# Disabled-by-default contract: no `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
# in the environment (true for Jay today — no Langfuse Cloud account yet, and
# true in CI) means every function here is a no-op — zero network calls, zero
# imports of `langfuse`/OpenTelemetry. That last part matters: `langfuse`
# pulls in the OpenTelemetry SDK + OTLP exporter (a real, if small, import
# cost), so `get_callbacks()` imports it lazily, INSIDE the function, rather
# than at module level — importing this module never pays that cost unless
# tracing is actually turned on.
from __future__ import annotations

import os
from typing import Any

from compliance_copilot.settings import settings


def tracing_enabled() -> bool:
    """Same reasoning as `settings.py`'s comment on OPENAI_API_KEY/
    ANTHROPIC_API_KEY: read the env var directly rather than declaring a
    settings field, so a bare `Settings()` repr/log never has a Langfuse
    secret sitting in it. Both keys required — Langfuse's own client needs
    both to authenticate; a public key alone would construct a client that
    silently no-ops on every call anyway (see get_callbacks' docstring)."""
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def get_callbacks() -> list[Any]:
    """One `CallbackHandler` per call when enabled, `[]` otherwise.

    Verified against installed `langfuse==4.14.5` source
    (`.venv/.../langfuse/langchain/CallbackHandler.py`): constructing
    `CallbackHandler()` with no keys does NOT raise — it logs
    `WARNING:langfuse:Authentication error: Langfuse client initialized
    without public_key. Client will be disabled. ...` and returns a disabled
    client that no-ops on every call. Gating on `tracing_enabled()` here
    anyway (rather than relying on that fallback) avoids constructing the
    handler/OTel machinery at all when tracing is intentionally off, and
    keeps "why is nothing in the dashboard" traceable to an obvious empty
    list instead of a silently-disabled object three layers down."""
    if not tracing_enabled():
        return []
    from langfuse.langchain import CallbackHandler  # noqa: PLC0415 — see module docstring

    return [CallbackHandler()]


def run_config(*, tags: list[str] | None = None, session_id: str | None = None) -> dict:
    """Builds the LangChain `config` dict for `graph.astream()`/`.invoke()`.

    Session/tags mechanism — verified against installed source, NOT the
    `propagate_attributes()` context manager the researcher flagged as the
    only confirmed API: `CallbackHandler.on_chain_start` (same file, ~line
    496) reads `metadata["langfuse_session_id"]` and `metadata["langfuse_
    tags"]` directly off `config["metadata"]` and attaches them to the root
    trace. That's a plain dict key, not a context manager wrapping the whole
    call — simpler, and it's what the installed 4.14.5 code actually does,
    so it's the one used here (brief: "pick the mechanism that's actually in
    the source").

    `tags` defaults to `[llm_provider, "env:<env>"]` plus `GIT_SHA` if the
    deploy sets it (Railway/most CI set this automatically; reading the env
    var instead of shelling out to `git rev-parse` keeps this function free
    of a subprocess call and working in a container with no `.git` dir)."""
    if tags is None:
        tags = [settings.llm_provider, f"env:{settings.env}"]
        git_sha = os.environ.get("GIT_SHA")
        if git_sha:
            tags = [*tags, f"sha:{git_sha}"]
    metadata: dict[str, Any] = {"langfuse_tags": tags}
    if session_id is not None:
        metadata["langfuse_session_id"] = session_id
    return {"callbacks": get_callbacks(), "metadata": metadata}


def current_trace_id(config: dict) -> str | None:
    """Reads the trace id off the handler(s) actually used for one run.

    Deviates from the brief's zero-arg `current_trace_id() -> str | None`:
    verified `Langfuse.get_current_trace_id()` (installed
    `langfuse/_client/client.py`) only returns non-None while an OTel span is
    the *current* active span in context — by the time `api.py`'s
    `_stream_answer` wants to emit the `trace` SSE event, `graph.astream()`
    has already returned and no span is active, so that accessor would
    always give None here. `CallbackHandler.last_trace_id` (same installed
    file, set at `on_chain_start`/`on_llm_start`) is an attribute on the handler
    INSTANCE, set as each run completes — reading it off the specific
    handler this request built (via its `config`) is correct per-request;
    a hypothetical no-arg version would need a shared module-level
    "current handler" global, which would race under FastAPI's concurrent
    requests. Returns `None` when tracing is disabled (empty callbacks) or
    no trace has completed yet."""
    for callback in config.get("callbacks", []):
        trace_id = getattr(callback, "last_trace_id", None)
        if trace_id is not None:
            return trace_id
    return None


def flush() -> None:
    """No-op when disabled. Otherwise a blocking call that force-sends any
    batched spans — call after a request whose trace_id you need visible
    immediately (verified `Langfuse.flush()`, installed client.py)."""
    if not tracing_enabled():
        return
    from langfuse import get_client

    get_client().flush()


def shutdown() -> None:
    """Like `flush()` but also stops the background consumer threads and
    unregisters the `atexit` handler — call once, from the FastAPI lifespan's
    shutdown phase (verified `Langfuse.shutdown()`, installed client.py),
    not per-request."""
    if not tracing_enabled():
        return
    from langfuse import get_client

    get_client().shutdown()


def score(name: str, value: float, trace_id: str | None) -> None:
    """First quality signal on the Langfuse dashboard (Day 10's eval work
    builds on this): a `citation_valid` 0.0/1.0 score attached to the trace
    that produced (or failed to produce) a validated answer. No-op when
    tracing is disabled or there's no trace to attach to (verified
    `Langfuse.create_score(trace_id=..., name=..., value=..., data_type=
    "NUMERIC", ...)`, installed client.py)."""
    if not tracing_enabled() or trace_id is None:
        return
    from langfuse import get_client

    get_client().create_score(name=name, value=value, trace_id=trace_id, data_type="NUMERIC")


# Masking (`Langfuse(mask=fn)`, set once at process startup) is deliberately
# NOT wired here — see ADR-0009's amendment and ADR-0006 (guard_in's Presidio
# redaction is the primary defense; masking here would only be a
# defense-in-depth backstop, and guard_in is the feature that owns "what
# reaches a trace/log", not this one).
