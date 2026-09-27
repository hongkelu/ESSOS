"""Continuation acceptance and call-history independence regression checks."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('vmex')
from vmex.core.solver import SpectralState
from vmex import VmecInput


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[1]/'scripts/vmex_radial_continuation.py'
    spec = importlib.util.spec_from_file_location('continuation_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_ladder(module, monkeypatch, tmp_path, *, bad_edge=False):
    calls = []
    def interpolate(state, *, ns_fine, modes):
        return SpectralState(**{name:np.full((ns_fine, 1), getattr(state, name)[0, 0])
            for name in ('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin')})
    def stage(inp, *, external_field, resolution, initial_state, **kwargs):
        calls.append((float(external_field[0]), resolution.ns, initial_state))
        state = SpectralState(**{name:np.full((resolution.ns, 1), external_field[0])
            for name in ('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin')})
        result = SimpleNamespace(state=state, fsqr=1e-13, fsqz=1e-13, fsql=1e-13,
            fedge=1e-4 if bad_edge else 1e-13, converged=True, iterations=10)
        return SimpleNamespace(result=result, vacuum=object(), rcon0=np.zeros(1), zcon0=np.zeros(1))
    monkeypatch.setattr(module, 'interpolate_state', interpolate)
    monkeypatch.setattr(module.fb, '_solve_free_boundary_stage', stage)
    monkeypatch.setattr(module.fb, 'free_boundary_resolution',
        lambda inp, field, ns:SimpleNamespace(ns=ns))
    inp = VmecInput.from_file(Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization/li383.vmec')
    return module.RadialContinuation(inp, lambda c:c, [16,24,32], 1e-12, 100, tmp_path), calls


def test_current_history_does_not_change_seed(module, monkeypatch, tmp_path):
    ladder, calls = setup_ladder(module, monkeypatch, tmp_path)
    base = ladder.solve([1.])
    ladder.solve([1.001])
    assert ladder.solve([1.]) is base
    assert len(calls) == 6
    ladder.solve([1.], fresh=True)
    assert len(calls) == 9
    for offset in (0,3,6):
        assert calls[offset][2] is None
        assert calls[offset+1][2].R_cos.shape[0] == 24
        assert calls[offset+2][2].R_cos.shape[0] == 32
    records = json.loads((tmp_path/'continuation.json').read_text())
    assert len(records) == 3
    assert all(row['accepted'] for record in records for row in record['stages'])
    assert all((tmp_path/row['checkpoint']).is_file() for record in records for row in record['stages'])


def test_unconverged_edge_stops_before_finer_stage(module, monkeypatch, tmp_path):
    ladder, calls = setup_ladder(module, monkeypatch, tmp_path, bad_edge=True)
    with pytest.raises(ValueError, match='rejected NS=16'):
        ladder.solve([1.])
    assert len(calls) == 1
    records = json.loads((tmp_path/'continuation.json').read_text())
    assert records[0]['stages'][0]['accepted'] is False
    assert not list(tmp_path.rglob('state-*.npz'))
