# ADR-0007: Tools via MCP — Python MCP server consumed through `langchain-mcp-adapters`

## Status
Accepted. 2026-08-23.

## Context
The graph's `retrieve` node needs to call retrieval operations (`search_regulation`, `get_article`, `cite`) without embedding that logic directly inside a LangGraph node — separating "what tools exist" from "how the agent orchestrates" is both a cleaner architecture and a specific, named market signal: MCP (Model Context Protocol — a standard protocol for exposing tools/resources to an LLM application, independent of which agent framework consumes them) is named in 20% of sampled ads, described in the market research as "already mainstream," and is a concrete way to show interoperability rather than framework-specific tool definitions.

## Options considered
1. **A Python MCP server** (built with the official `mcp` SDK) exposing `search_regulation`, `get_article`, `cite` as MCP tools, consumed by the LangGraph agent through **`langchain-mcp-adapters`** (a LangChain-maintained package that wraps MCP tools as LangChain-compatible `BaseTool` objects).
2. **Plain LangChain tools only** (define `search_regulation` etc. as regular `@tool`-decorated Python functions, no MCP involved) — simpler, no protocol/process boundary to manage, but has no interoperability story: the tools would only ever be callable from this one LangChain/LangGraph codebase, and MCP's whole value (a tool server other MCP clients — not just this project's agent — could also call) would be forgone. Also forgoes the market signal named above.

## Decision
**A Python MCP server**, built with the official `mcp` SDK (PyPI package `mcp`), exposing three tools: `search_regulation` (vector search over the corpus, ADR-0003/0004), `get_article` (fetch a specific article by regulation+number), and `cite` (format a citation for a given article/recital reference). The LangGraph agent's `retrieve` node consumes these through **`langchain-mcp-adapters`**'s `MultiServerMCPClient`, which loads MCP tools as LangChain `BaseTool` objects that a LangGraph node/`ToolNode` can call directly.

**Verified breaking detail — this is the one place this research surfaced something that would have been built wrong if guessed from training data:** the official `mcp` Python SDK is currently at a **major version 2 (`mcp` 2.0.0 on PyPI, verified 2026-08-23)**, described in its own PyPI metadata as *"a major revision supporting the latest Model Context Protocol specification and designed to address architectural improvements from the previous version."* As part of that revision, **the class historically called `FastMCP` has been renamed to `MCPServer`**, importable as `from mcp.server.mcpserver import MCPServer`. Source-code comments in the SDK itself mark this explicitly (`# The @mcp.tool() decorator on the MCPServer class (formerly FastMCP)`, `# MCPServer.run() with transport overloads (formerly FastMCP)`). The decorator API is otherwise unchanged and still intuitive: `mcp = MCPServer("Demo")`, then `@mcp.tool()`, `@mcp.resource("uri://{param}")`, `@mcp.prompt()`, and `mcp.run(transport="stdio" | "sse" | "streamable-http")`. **This ADR's server should be written against `mcp.server.mcpserver.MCPServer`, not `mcp.server.fastmcp.FastMCP`** — the latter import path reflects the SDK's v1 layout and is what a training-data-only implementation would likely produce incorrectly here. (Whether v1's `fastmcp` import path survives as a backward-compatible alias in `mcp` 2.x was not separately confirmed — treat it as unavailable and use the current path.)

`langchain-mcp-adapters` usage verified against its own README (Context7, `langchain-ai/langchain-mcp-adapters`): `MultiServerMCPClient({...server configs...})`, then `tools = await client.get_tools()`, then bind those tools into a LangGraph `StateGraph` using `langgraph.prebuilt.ToolNode` and `tools_condition` for the conditional edge that routes to tool execution — this is the pattern the `retrieve` node (or a small subgraph around it) should follow.

## Why not the others
- **Plain LangChain tools only**: rejected because it forgoes both the interoperability MCP is specifically designed for and the market signal (MCP named in 20% of ads, "already mainstream" per the research). A tool server reachable via a standard protocol is also just a cleaner separation of concerns — the retrieval implementation can be tested, versioned, and (in principle) reused by a completely different client without touching agent code.

## Security & cost implications
- **Security:** the MCP server is the only path from the graph to Postgres for retrieval (`docs/ARCHITECTURE.md` §6, boundary #3) — tool inputs must be schema-validated (Pydantic, matching the tool's declared input schema) before they reach any database query, so a compromised or malfunctioning graph node cannot pass through an arbitrary string as if it were a safe parameter. Running the MCP server as a **separate process/container** (not in-process with the API) also means an MCP-server crash or hang doesn't take down the whole API process — it surfaces as the "MCP server unreachable" failure mode in `docs/ARCHITECTURE.md` §7, handled explicitly (fail closed, 503) rather than silently.
- **Cost:** no direct API cost — MCP itself is a protocol, not a hosted service; the only cost implication is one more container/process to run (`docs/ARCHITECTURE.md` §9's VPS sizing already accounts for it).

## How to reverse
The MCP server's tool implementations (the actual retrieval/citation logic) are ordinary Python functions decorated for MCP exposure — removing MCP would mean re-exposing those same functions as plain `@tool`-decorated LangChain tools instead, a mechanical change at the exposure layer, not a rewrite of the retrieval logic itself. On the client side, swapping `langchain-mcp-adapters`'s `MultiServerMCPClient` for direct in-process tool objects is similarly localized to wherever the graph's tool list is constructed.

## Amendment (2026-08-26) — implementation

Installing this ADR's two packages together (`mcp` + `langchain-mcp-adapters`)
resolves differently from what the original decision assumed:
`langchain-mcp-adapters` 0.3.2 pins `mcp<2.0.0,>=1.24.0`, so with nothing else
constraining `mcp` upward, `uv`'s resolver picks the highest 1.x release,
**`mcp==1.29.1`** — never the 2.0.0+ this ADR's "verified breaking detail"
named. That release ships the **pre-rename `FastMCP` class**
(`mcp.server.fastmcp.FastMCP`), not `MCPServer`
(`mcp.server.mcpserver.MCPServer`, only reachable on `mcp>=2.0.0`). `src/
compliance_copilot/mcp_server.py` is written against `FastMCP` for this
reason — the ADR's decision (a Python MCP server exposing three read-only
tools, consumed via `langchain-mcp-adapters`) is unchanged; only the SDK
class name and import path are corrected here. If `langchain-mcp-adapters`
is ever dropped, `mcp>=2.0.0`'s `MCPServer` becomes installable again and
this file would need updating to match — a mechanical rename, not a
redesign.

Implementation details verified against the installed `mcp` 1.29.1 wheel
(`.venv/lib/python3.12/site-packages/mcp/`), not guessed:
- **Sync tools**: all three tools (`search_regulation`, `get_article`,
  `cite`) are plain `def` functions — FastMCP auto-runs sync tools in a
  threadpool so they never block the event loop; no `async def` needed.
- **Per-call sessions**: each tool opens its own `Session(engine)` inside
  the function body. The one thing built once is an `AppContext(engine,
  embeddings)` yielded by an `@asynccontextmanager` `app_lifespan(server:
  FastMCP)`, read back via `ctx.request_context.lifespan_context` — the
  `Context` parameter is detected by its type annotation (not by name or
  position), and excluded from the tool's argument schema automatically.
- **Validation bounds**: `Field(min_length=/max_length=/ge=/le=/pattern=)`
  on tool parameters becomes the tool's JSON input schema, and IS enforced
  by the SDK for a real protocol `call_tool` (confirmed by reading
  `FuncMetadata.call_fn_with_arg_validation`'s `model_validate` call) —
  but a plain Python function call (this project's own unit tests bypass
  the protocol layer entirely) skips that validation, so each tool also
  re-checks its own bounds manually and raises `ValueError` — defense in
  depth at the actual trust boundary (this ADR's boundary #3), not
  redundant ceremony.
- **`cite`'s verbatim-quote check** imports `_normalise`/`_MIN_QUOTE_LENGTH`
  directly from `graph.nodes` rather than extracting them into a new
  shared module: `langchain-anthropic`/`langchain-openai` (what `nodes.py`
  otherwise pulls in) are already hard project dependencies, so this
  import adds no new package and is the smaller diff than a new
  `guards/quotes.py` module — the alternative this ADR's amendment brief
  raised as a fallback in case that import "drags in LLM deps" it
  wouldn't otherwise carry.
- **Structured output shape**: a `list[dict]` return (`search_regulation`)
  is wrapped by the SDK as `{"result": [...]}` in `structuredContent`; a
  `dict[str, Any]` return (`get_article`, `cite`) is NOT wrapped — returned
  as-is. (A bare `-> dict` annotation, with no type args, produces no
  structured output schema at all — confirmed by reading
  `func_metadata.py`'s type-dispatch branches — so both dict-returning
  tools are annotated `dict[str, Any]`, not bare `dict`.)
- **Transport**: `mcp.run(transport="stdio" | "sse" | "streamable-http")` —
  `sse` is legacy, unused here. `settings.mcp_transport` defaults to
  `"stdio"` (dev/CI/`MultiServerMCPClient`'s spawn model); `"streamable-
  http"` is for the Compose-internal `mcp-server` container only, host/port
  from `settings.mcp_host`/`mcp_port` (default `127.0.0.1:8001`) — still no
  auth story on that transport, so it must never cross the Compose bridge
  network (unchanged from this ADR's original security section).
- **Testing**: unit tests call the tool functions directly with a
  hand-built fake `Context` (no MCP protocol, no DB). Integration tests use
  `mcp.shared.memory.create_connected_server_and_client_session` (a real
  in-process protocol round-trip, no subprocess) for most assertions, plus
  exactly one real `mcp.client.stdio.stdio_client` +
  `StdioServerParameters` subprocess round-trip — the actual Day-17
  production transport — restricted to DB-only tool calls
  (`list_tools`, `get_article`) since `FakeEmbeddings` can't cross a
  process boundary and the test suite must not require a real
  `OPENAI_API_KEY`/network call.

## Amendment (2026-08-28) — Day 17: client side

`retrieve_node` (graph/nodes.py) is now the MCP *client* — the decoupling
this ADR named on paper is real in the running system as of this amendment.

**Adapter and session model.** `langchain-mcp-adapters` 0.3.2's
`MultiServerMCPClient` (`langchain_mcp_adapters.client`), configured with a
single `"copilot"` stdio connection (`build.py`'s `_mcp_connection()`) —
`command="uv"`, `args=["run", "--frozen", "python", "-m",
"compliance_copilot.mcp_server"]`, `env=dict(os.environ)` (full passthrough,
not a curated allowlist — the stdio SDK only inherits a small OS-dependent
safe subset by default, which would drop `DATABASE_URL`/`OPENAI_API_KEY`/
`PATH`/`uv` itself). `build.make_mcp_tools()` calls
`await client.get_tools()` once and returns `{tool.name: tool}`, threaded
into `GraphContext.tools` (state.py) the same way `session`/`embeddings`/
`llm`/`classifier` already are. **Per-call sessions** (the adapter's
documented default, confirmed live): building the tool list is cheap and
holds no connection open, but every `tool.ainvoke(...)` call still opens
and tears down its own subprocess+session — so `make_mcp_tools()` is safe
to call once (API startup, one CLI/eval invocation) and reuse across
requests without holding anything open.

**How a tool call actually returns data — verified live, not guessed.**
`langchain-mcp-adapters` builds every MCP tool as a LangChain
`StructuredTool` with `response_format="content_and_artifact"`. Calling
`tool.ainvoke(plain_args_dict)` (no `tool_call_id`) returns ONLY the
tool's human-readable text content — the parsed `structuredContent` (the
actual dict/list our tools return) is silently dropped
(`langchain_core.tools.base._format_output`: `tool_call_id is None` short-
circuits straight to `return content`). The fix, confirmed against a real
in-process server: invoke with a `ToolCall`-shaped input instead —
`{"type": "tool_call", "name": ..., "args": ..., "id": ...}` — which makes
LangChain build a real `ToolMessage` carrying `.artifact["structured_content"]`.
`nodes.py`'s `_call_tool` helper does exactly this. A tool's own reported
failure (e.g. `get_article`'s `ValueError: not found`) does NOT raise —
FastMCP converts it into `ToolMessage(status="error")` (`handle_tool_errors`
defaults `True`); `_call_tool` checks `result.status` explicitly rather than
relying on an exception.

**Two-hop retrieval, and a real capability change.** `search_regulation`
returns a ranked, 300-char-snippet list — too short to answer from or
validate citations against. `retrieve_node` calls `get_article` once per
unique `(regulation, anchor)` from that ranking to get each match's full,
all-parts-joined text, which is what actually reaches the prompt and the
citation check. `search_regulation` only searches `kind="article"`
(mcp_server.py, ADR-0013's original design for this tool) — no MCP tool
exposes recital search, so **`state["recitals"]` is now always `[]`**, a
real behaviour change from the pre-MCP `retrieve()` call (which fetched
supporting recitals separately). Flagged here rather than silently
absorbed: extending `search_regulation`'s schema to add a `kind` parameter
was considered and rejected for this amendment's scope — the server's tool
contract was already reviewed/approved on Day 16, and changing it wasn't
needed for the client-wiring work this amendment covers. Revisit if
recital context turns out to matter for answer quality.

**Error/retry/timeout policy** (`nodes.py`'s `_call_tool` +
`ToolCallError`, state.py): a per-call `asyncio.wait_for` timeout
(`settings.mcp_tool_timeout_s`, default 30s); a bounded retry
(`_MAX_TOOL_ATTEMPTS=2`, i.e. one retry) ONLY for `TimeoutError` and
transient transport errors (`OSError`/`mcp.shared.exceptions.McpError`) —
never for a tool's own reported `status="error"` or a malformed result
shape, since retrying a validation failure just burns the timeout budget
on a call that can never succeed. Every failure path raises a typed
`ToolCallError` (never a silent fallback to direct retrieval) — this
surfaces through api.py's existing generic `except Exception` handler as
the `internal_error` SSE event with no code change needed there, and
through the CLI as an `INTERNAL:` message (exit code 5). Logs the tool
name, latency, and error CLASS only — never the call's arguments, which
carry the (already-redacted) question.

**Sync/async, verified not assumed.** A real installed-`langgraph` (1.2.11)
smoke test confirmed sync `graph.invoke()` raises `TypeError: No
synchronous function provided to "retrieve"` the moment a run actually
reaches an async node — there is no working sync path once `retrieve_node`
is `async def`. Every caller (`build.ask()`, `cli.py`'s `ask` command,
`api.py`'s already-async `_stream_answer`, `evals/run_redteam.py`'s
`_run_full`, `evals/run_answer_eval.py`) now drives the graph via
`ainvoke`/`astream`, wrapped in one `asyncio.run(...)` per process
entrypoint. The one exception: `evals/run_redteam.py`'s
`_run_heuristics_subset` stays on sync `graph.invoke()` unchanged — a
second live smoke test confirmed sync invoke works fine as long as the run
never actually reaches the async node, which is true by construction for
attacks tagged `must_block_at: "heuristics"` (blocked at `guard_in`,
`retrieve` never scheduled).

**Security & cost.** The MCP server is a stdio subprocess the API process itself spawns on the same host — no new network listener, no auth surface added (streamable-http stays loopback-only per the Day-16 amendment). The child inherits the parent's full environment (`env=dict(os.environ)` in `build.py`) because the stdio SDK otherwise starts it with a near-empty env and `uv run` needs PATH/HOME/VIRTUAL_ENV as well as DATABASE_URL/OPENAI_API_KEY; this is the same trust domain as the API process, so nothing new is exposed, but a curated allowlist is the upgrade if the server is ever run as a separate service. Tool-call logs record tool name, latency and error class only — never arguments, which carry the (redacted) question. Tool results are data: they are rendered into the prompt exactly like direct retrieval was, never executed or treated as instructions, and citations are still validated against the retrieved set (ADR-0014). Cost: zero new LLM calls — the retrieval call is deterministic, so the only addition is a local process round-trip of a few milliseconds per query plus the subprocess start-up on first use.

**How to reverse.** `settings.mcp_enabled` (`False`) makes
`make_mcp_tools()` return `None` without spawning the server subprocess —
`GraphContext.tools` stays empty and a real question reaching
`retrieve_node` still raises `ToolCallError` immediately (fail loudly,
never a silent fallback to the old direct `retrieve()` import — that
import path no longer exists in nodes.py at all). Reversing the MCP
integration itself means restoring the direct `retriever.retrieve()` call
in `retrieve_node` and dropping the `tools`/`ToolCallError` machinery —
mechanical, since `retriever.py` itself was never touched by this
amendment.

**Testing.** Unit (`tests/test_graph.py`, `tests/fake_mcp_tools.py`): fake
tool doubles matching the exact `ToolCall`-in/`ToolMessage`-out calling
convention above (not a plain dict-in/dict-out shortcut — that would test
a contract the real tools don't have). Integration
(`tests/test_graph_mcp_integration.py`): an in-process real MCP session
(`mcp.shared.memory.create_connected_server_and_client_session` +
`load_mcp_tools`) driving the full compiled graph against the test DB
fixture corpus, plus one real stdio-subprocess run through
`MultiServerMCPClient` (gated behind `RUN_NETWORK_TESTS=1` +
`OPENAI_API_KEY`, same convention as
`tests/test_search_real_embeddings_integration.py`, since a real
`search_regulation` call needs real embeddings that can't cross the
subprocess boundary). `tests/test_graph_real_integration.py`/
`tests/test_tracing_real_integration.py` (real LLM + real DB) also now
spawn a real MCP server subprocess rather than a fake tool double, since
their whole point is a real end-to-end run.

## References
- `mcp` (Model Context Protocol Python SDK), PyPI: 2.0.0 — https://pypi.org/project/mcp/ (verified 2026-08-23); docs: https://py.sdk.modelcontextprotocol.io/, repo: https://github.com/modelcontextprotocol/python-sdk
- `FastMCP` → `MCPServer` rename: Context7 `/modelcontextprotocol/python-sdk`, sources `examples/snippets/servers/mcpserver_quickstart.py`, `examples/mcpserver/readme-quickstart.py`, `src/mcp/server/mcpserver/server.py` (verified 2026-08-23)
- `langchain-mcp-adapters`, PyPI: 0.3.2 — https://pypi.org/project/langchain-mcp-adapters/ (verified 2026-08-23); usage pattern verified against Context7 `/langchain-ai/langchain-mcp-adapters`, source `README.md`
- Market data on MCP naming frequency ("already mainstream" read): `docs/research/market_research.md`
