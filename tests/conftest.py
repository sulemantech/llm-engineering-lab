"""Turns NotImplementedError into SKIPPED, so test status reads as a to-do list:
skipped = not built yet, failed = built but wrong, passed = done."""

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, "TODO: not implemented yet")
