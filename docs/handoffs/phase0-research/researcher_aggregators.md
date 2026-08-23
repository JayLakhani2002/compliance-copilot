# Researcher handoff — aggregators/south-north boards

Count: 15 ads collected, all opened in full (not snippet-only), verbatim requirement lines pulled from each.

Sources that worked: arbeitnow.com (best yield — but ~50% of search-result URLs were already 410 Gone/archived by fetch time, had to re-search for live IDs), join.com (worked once past archived listings — same 410 churn), aijobs.net/aijobs.net redirect (ICEYE Munich ad worked well).

Sources that failed/blocked: welcometothejungle.com (403 Forbidden on WebFetch), Cognigy careers (JS-rendered, empty on fetch), Helsing jobs page (429 rate limited), jobs.lever.co/Qonto (403), jobs.ashbyhq.com/AlephAlpha (JS-rendered, no static content), jobs.earlybird.com Aleph Alpha listings (both roles found were already closed — "no longer accepting applications"), Firecrawl MCP scrape hit free-tier rate limit after ~3 calls and stayed blocked rest of session (had to fall back to WebFetch only).

3 surprising observations:
1. Job-board link rot is severe and fast: the majority of URLs surfaced by web search for arbeitnow.com/join.com were already expired (HTTP 410) within what search cache implied was a live listing — real "currently open" yield was roughly 1 in 3 candidate URLs found.
2. German-language fluency (C1, sometimes "native") is a hard requirement on a large share of these ads even for "AI/GenAI/LLM Engineer" titles (arcode Systems, Talents2Germany, Pentadoc, Anthropic Applied AI Engineer) — not just for customer-facing consulting roles.
3. LangChain/LangGraph and RAG+vector-DB fluency appear as baseline expectations even at small startups (Alpas, Manex AI, Nejo), while explicit evaluation/observability tooling (Langfuse, LLM tracing, "designed and run LLM evaluations") is named specifically in a handful of ads (Nejo, Manex AI) rather than universally — suggesting eval/observability is a differentiator, not yet table stakes.

Files:
- Raw ads: /Users/jay/Documents/Projects/Langchain Project/docs/research/raw/aggregators_south_north.md
