# src/compliance_copilot/__init__.py
#
# Package root for Compliance Copilot: an agentic RAG system that answers
# questions over the EU AI Act and GDPR, with retrieval evals and guardrails
# gating every change in CI (see docs/ARCHITECTURE.md for the full design).
#
# This file currently only exposes __version__, used by tests/test_smoke.py
# to confirm the package installs and imports correctly. Real modules
# (ingestion, retrieval, agent graph, API) land feature-by-feature per
# docs/CURRICULUM.md — see docs/PROGRESS.md for what's built so far.

__version__ = "0.1.0"
