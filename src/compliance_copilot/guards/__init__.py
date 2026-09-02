# src/compliance_copilot/guards/__init__.py — package surface for
# `guard_in`'s three layers: heuristics (ADR-0018), the LLM classifier
# (ADR-0019), and PII redaction (ADR-0020). Callers import from
# `compliance_copilot.guards` rather than reaching into the submodules
# directly — same reasoning as `graph/__init__.py`'s package surface: the
# module split can change without breaking callers.
from compliance_copilot.guards.injection import GuardResult, detect, normalise
from compliance_copilot.guards.pii import RedactionResult, detect_language, redact

__all__ = [
    "GuardResult",
    "RedactionResult",
    "detect",
    "detect_language",
    "normalise",
    "redact",
]
