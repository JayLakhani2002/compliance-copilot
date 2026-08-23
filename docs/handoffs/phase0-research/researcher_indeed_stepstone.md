# Handoff: researcher — indeed_stepstone

Ads collected: 20 (15 Indeed + 5 StepStone), all seen 2026-08-23, all posted within ~60 days.

Sources: Indeed MCP (search_jobs + get_job_details) worked fully — searched Berlin/Munich/Hamburg/Frankfurt/Stuttgart across AI/LLM/GenAI/ML titles, opened every ad's full description before recording. StepStone via Firecrawl search hit the free-tier rate limit immediately (search + scrape both blocked), so fell back to WebSearch (site:stepstone.de) to find listing pages, then WebFetch on individual /stellenangebote--...-inline.html URLs, which rendered full text cleanly.

Rejected/skipped: several Indeed hits were plausible titles but not GenAI/LLM-relevant on inspection (e.g. sensmore's ML Engineer is LiDAR/camera perception for robotics, Zalando's Senior ML Engineer is classic recommendation-systems MLOps) — dropped rather than force-fit. Many more Indeed results were >60 days old (Feb–May 2026 postings) and excluded per the freshness rule; one StepStone listing (Novoferm, from an early search hit) returned HTTP 410 Gone — closed, excluded.

Three surprising observations:
1. MCP (Model Context Protocol) already appears as an explicit named requirement in production job ads (Infosys, N26, Ecoza, commercetools) — not just LangChain/RAG basics anymore; agent-to-agent (A2A) orchestration is also explicitly named at N26.
2. German-language fluency (usually C1) is an explicit hard requirement in the large majority of ads, even at English-first companies like Salesforce and N26 — this is a real gating filter, not a formality.
3. LangGraph/CrewAI-style multi-agent orchestration and "evals as a first-class system" (commercetools' phrasing) show up more often than raw fine-tuning — production hiring is optimizing for agent reliability/evaluation over model training skills.
