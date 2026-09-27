"""Required-lane failures must remain visible even when ordinary pytest skips."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("source,options,success", [
    ("def test_ok(): pass", [], True),
    ("import pytest\ndef test_skip(): pytest.skip('missing backend')", [], False),
    ("import pytest\npytest.importorskip('missing_manifold_test_backend')", [], False),
    ("import pytest\n@pytest.mark.xfail\ndef test_xfail(): assert False", [], False),
    ("def test_bad(): assert False", [], False),
    ("# empty test module", [], False),
    ("def test_a(): pass\ndef test_b(): pass", ["-k", "test_a"], False),
    ("import pytest\n@pytest.fixture\ndef fixture():\n yield\n"
     " pytest.skip('teardown')\ndef test_ok(fixture): pass", [], False),
])
def test_required_lane_audits_execution(tmp_path, source, options, success):
    (tmp_path / "test_example.py").write_text(source + "\n")
    audit = tmp_path / "audit.json"
    env = os.environ.copy()
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1] / "scripts"),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_example.py", "-q",
         "-p", "manifold_audit", f"--manifold-audit={audit}", *options],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    report = json.loads(audit.read_text())
    assert (report["exit_code"] == 0) == success



@pytest.mark.parametrize('returncode,audit,expected,success', [
    (0, {'exit_code': 0, 'collected': ['a', 'b']}, 2, True),
    (1, {'exit_code': 0, 'collected': ['a']}, 1, False),
    (0, {}, 1, False),
    (0, {'exit_code': 1, 'collected': ['a']}, 1, False),
    (0, {'exit_code': 0, 'collected': ['a']}, 2, False),
    (0, {'exit_code': 0, 'collected': []}, 0, False),
])
def test_inventory_count_is_required(returncode, audit, expected, success):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'scripts/run_manifold_baseline.py'
    spec = importlib.util.spec_from_file_location('baseline_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.audit_passed(returncode, audit, expected) == success


@pytest.mark.parametrize('step,qos,allowed', [
    (None, 'gpu_interactive', False),
    ('1', 'gpu_interactive', True),
    ('1', 'debug', True),
    ('1', 'regular', False),
])
def test_perlmutter_requires_interactive_or_debug_step(monkeypatch, step, qos, allowed):
    import argparse
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'scripts/run_manifold_baseline.py'
    spec = importlib.util.spec_from_file_location('baseline_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('NERSC_HOST', 'perlmutter')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    if step is None:
        monkeypatch.delenv('SLURM_STEP_ID', raising=False)
    else:
        monkeypatch.setenv('SLURM_STEP_ID', step)
    monkeypatch.setattr(module.subprocess, 'check_output',
                        lambda *a, **k: f'JobId=123 QOS={qos} NumNodes=1 NumCPUs=128')
    parser = argparse.ArgumentParser()
    if allowed:
        assert module.require_compute_allocation(parser)['QOS'] == qos
    else:
        with pytest.raises(SystemExit):
            module.require_compute_allocation(parser)


@pytest.mark.parametrize('lanes,counts', [
    ({}, {}),
    ({'analytic': {}}, {}),
    ({'analytic': {'pyna': []}}, {}),
    ({'analytic': {'pyna': ['test.py']}}, {}),
    ({'analytic': {'pyna': ['test.py']}}, {'pyna/test.py': 0}),
])
def test_empty_or_unaccounted_inventory_rejects(lanes, counts):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'scripts/run_manifold_baseline.py'
    spec = importlib.util.spec_from_file_location('baseline_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError):
        module.select_lanes({'lanes': lanes, 'expected_counts': counts}, 'all')
