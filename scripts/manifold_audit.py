"""Opt-in pytest plugin: required manifold tests must execute, without skips."""

import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--manifold-audit", help="Write required-test execution audit")


def pytest_configure(config):
    config._manifold_audit = {"collected": [], "reports": [], "deselected": []}


def pytest_collection_finish(session):
    session.config._manifold_audit["collected"] = [
        item.nodeid for item in session.items
    ]


def pytest_deselected(items):
    if items:
        items[0].config._manifold_audit["deselected"].extend(
            item.nodeid for item in items
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    item.config._manifold_audit["reports"].append({
        "nodeid": report.nodeid,
        "phase": report.when,
        "outcome": report.outcome,
        "duration_seconds": report.duration,
        "wasxfail": getattr(report, "wasxfail", None),
        "detail": str(report.longrepr) if report.longrepr else None,
    })


def pytest_sessionfinish(session, exitstatus):
    audit = session.config._manifold_audit
    passed = {r["nodeid"] for r in audit["reports"]
              if r["phase"] == "call" and r["outcome"] == "passed"
              and r["wasxfail"] is None}
    audit["not_passed"] = sorted(set(audit["collected"]) - passed)
    audit["pytest_exit_code"] = int(exitstatus)
    if (not audit["collected"] or audit["not_passed"]
            or audit["deselected"]
            or any(r["outcome"] == "skipped" or r["wasxfail"] is not None
                   for r in audit["reports"])):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    audit["exit_code"] = int(session.exitstatus)
    destination = session.config.getoption("--manifold-audit")
    if destination:
        Path(destination).write_text(json.dumps(audit, indent=2) + "\n")

