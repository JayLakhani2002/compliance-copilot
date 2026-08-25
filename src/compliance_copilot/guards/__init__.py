# src/compliance_copilot/guards/__init__.py — package surface for the
# input-side prompt-injection detector (docs/ARCHITECTURE.md §4's `guard_in`
# node, ADR-0018). Callers import from `compliance_copilot.guards` rather
# than reaching into `guards.injection` directly — same reasoning as
# `graph/__init__.py`'s package surface: the module split can change
# without breaking callers.
from compliance_copilot.guards.injection import GuardResult, detect, normalise

__all__ = ["GuardResult", "detect", "normalise"]
