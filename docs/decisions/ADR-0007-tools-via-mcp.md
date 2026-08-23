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

## References
- `mcp` (Model Context Protocol Python SDK), PyPI: 2.0.0 — https://pypi.org/project/mcp/ (verified 2026-08-23); docs: https://py.sdk.modelcontextprotocol.io/, repo: https://github.com/modelcontextprotocol/python-sdk
- `FastMCP` → `MCPServer` rename: Context7 `/modelcontextprotocol/python-sdk`, sources `examples/snippets/servers/mcpserver_quickstart.py`, `examples/mcpserver/readme-quickstart.py`, `src/mcp/server/mcpserver/server.py` (verified 2026-08-23)
- `langchain-mcp-adapters`, PyPI: 0.3.2 — https://pypi.org/project/langchain-mcp-adapters/ (verified 2026-08-23); usage pattern verified against Context7 `/langchain-ai/langchain-mcp-adapters`, source `README.md`
- Market data on MCP naming frequency ("already mainstream" read): `docs/research/market_research.md`
