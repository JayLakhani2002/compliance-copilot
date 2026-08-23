# tests/test_smoke.py — smoke test: does the package even import?
# This is the first thing CI runs; if it fails, the environment/packaging
# is broken and no feature test further down is worth trusting.

import compliance_copilot


def test_package_imports_and_has_version():
    assert isinstance(compliance_copilot.__version__, str)
    assert compliance_copilot.__version__
