# ADR-0010: Hosting — Docker Compose on a Hetzner Cloud VPS (Germany), Caddy for TLS, Terraform stub for AWS

## Status
Accepted. 2026-08-23.

## Context
The system needs to actually run somewhere reachable over the internet, with TLS, for the API (ADR-0008), the MCP server (ADR-0007), Postgres (ADR-0003), and now — per ADR-0009's finding — a five-service Langfuse observability stack. EU data residency (`docs/ARCHITECTURE.md` §8) is a named project goal, and the project's default constraints (`CLAUDE.md`) are 3–4h/day over 6 weeks, solo — meaning the hosting choice needs to be operable by one person in spare time, not a platform requiring a dedicated ops rotation.

## Options considered
1. **Docker Compose on a single Hetzner Cloud VPS, physically in Germany**, all services (`api`, `postgres`, `langfuse` + its dependencies, `mcp-server`) as containers behind **Caddy** (a reverse proxy that automates TLS certificate issuance/renewal via Let's Encrypt with minimal config) — with a **Terraform stub for AWS `eu-central-1`** as a stretch goal (not a full production migration, just infrastructure-as-code groundwork showing the path).
2. **Kubernetes** — the standard answer for "how do I run containers at scale," but explicitly overkill for a single-VPS, low-traffic portfolio deployment; the project brief itself calls this out and suggests mentioning **Helm** (a Kubernetes package manager) as a named upgrade path rather than building on K8s from day one.
3. **Railway EU** — a managed PaaS with an EU region option; simpler day-one ops than a bare VPS (no server management), but a weaker "I control my own EU-residency story end to end" narrative than a VPS the project fully owns and configures, which matters given how central data residency is to this project's premise.

## Decision
**Docker Compose on a Hetzner Cloud VPS located in Germany**, running all services as containers on one box, with **Caddy** as the reverse proxy/TLS terminator (Caddy's automatic HTTPS means no manual certificate management — a real ops-time savings for a solo 3–4h/day build). A **Terraform stub targeting AWS `eu-central-1`** is included as a stretch artifact — enough infrastructure-as-code to demonstrate the pattern (and to pair with ADR-0002/0004's Bedrock `eu-central-1` production path for inference/embeddings) without committing to a full cloud migration inside the 6-week timeline. **Kubernetes is explicitly not used**, with **Helm** named as the documented upgrade path if/when the project ever needs multi-node scaling.

**Sizing note carried over from ADR-0009:** because the full Langfuse self-host stack needs ClickHouse + Redis + MinIO in addition to this project's own Postgres (ADR-0003/0009), the single VPS needs to be sized for **five-plus stateful services** running concurrently, not just one Postgres instance — `docs/ARCHITECTURE.md` §9 estimates an 8GB-RAM-class box (e.g., Hetzner CX32/CPX31-tier) as the realistic minimum, not the smallest/cheapest tier, specifically because of ClickHouse's memory footprint.

## Why not the others
- **Kubernetes**: rejected explicitly as overkill for this project's scale — the project brief itself frames this as a deliberate, named-and-explained simplification (Helm as the stated upgrade path), not an oversight, which is itself worth stating plainly in a portfolio context (knowing when *not* to reach for K8s is also a signal).
- **Railway EU**: rejected on narrative grounds, not technical ones — a bare VPS the project configures end-to-end (Caddy, Docker Compose, firewall, backups) is a stronger demonstration of "I understand and control the EU-residency story" than a PaaS that handles those concerns invisibly, even though the PaaS would objectively be less ops work.

## Security & cost implications
- **Security:** running everything on one VPS means the internal Docker network is the only thing standing between services that should not be publicly reachable (Postgres, ClickHouse, Redis, MinIO, the MCP server) and the internet — only Caddy (and through it, the API) should be exposed on public ports; every other container should be reachable only over the Compose-internal network. Caddy's automatic TLS removes a whole class of "expired/misconfigured certificate" incidents that a manually-managed cert setup would risk.
- **Cost:** see `docs/ARCHITECTURE.md` §9 for the full breakdown — roughly €20–30/month fixed VPS+domain cost dominates at this project's traffic scale, with LLM/embedding API spend (ADR-0002/0004) as the variable component. A single-VPS Docker Compose setup is materially cheaper than either a managed Kubernetes cluster or a PaaS charging per-service.

## How to reverse
Docker Compose service definitions translate reasonably directly to Kubernetes manifests/Helm charts (each Compose service becomes roughly one Deployment + Service) if the project ever needs to scale beyond one box — this is the documented upgrade path rather than a full rewrite. Moving from Hetzner to AWS `eu-central-1` is the explicit purpose of the Terraform stub — it exists specifically so that migration path is partially pre-built rather than starting from zero if/when it's needed.

## References
- Project defaults (3–4h/day, 6 weeks, solo, EU deploy): `CLAUDE.md`
- Hosting options ranked against market signal (Kubernetes 38%, Docker 34%, Terraform 18% named in ads — infra is "table stakes," per the research read): `docs/research/market_research.md`
- Langfuse self-host service count driving VPS sizing: see ADR-0009 references

## Planner amendment — 2026-08-23
Per ADR-0009 amendment, Langfuse runs in Langfuse Cloud (EU region) for v1.0, so the VPS hosts `api`, `postgres`, `mcp-server`, `caddy` only → 4 GB-class Hetzner box is sufficient. Re-size to 8 GB only if the Week-6 self-host stretch is taken.
