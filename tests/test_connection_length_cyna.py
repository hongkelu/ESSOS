"""Native Cyna wall-hit tracing against the closed-form circular-tokamak plate solution.

The same exact reference checks the differentiable ESSOS tracer in
``test_connection_length.py``. Here the field is sampled onto Cyna's cylindrical grid,
so agreement is limited by trilinear interpolation and must improve under refinement.
"""
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pyna._cyna")
from pyna.toroidal.flt import trace_wall_hits_twall_field  # noqa: E402
from pyna.toroidal.geometry import ToroidalWall  # noqa: E402

from essos.fields import CircularTokamakField  # noqa: E402
from essos.manifold import essos_field_to_pyna_cylindrical_grid  # noqa: E402
from tests.test_connection_length import (  # noqa: E402
    B_POLOIDAL, B_TOROIDAL, MAJOR_RADIUS, PLATE_DEPTH, circular_tokamak_plate_solution)


def _box_wall(depth, half_width=1.0, top=1.0, nphi=4):
    phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
    R = MAJOR_RADIUS + np.array([-half_width, half_width, half_width, -half_width])
    Z = np.array([-depth, -depth, top, top])
    return ToroidalWall(phi, np.tile(R, (nphi, 1)), np.tile(Z, (nphi, 1)))


def _relative_errors(n_grid, step):
    field = CircularTokamakField(MAJOR_RADIUS, B_TOROIDAL, B_POLOIDAL)
    grid = essos_field_to_pyna_cylindrical_grid(
        field, np.linspace(MAJOR_RADIUS - 1.2, MAJOR_RADIUS + 1.2, n_grid),
        np.linspace(-1.2, 1.2, n_grid), np.linspace(0.0, 2 * np.pi, 4, endpoint=False))
    r = np.array([0.5, 0.7, 0.9])
    theta = np.array([0.0, 1.2, 2.9])
    R, Z = MAJOR_RADIUS + r * np.cos(theta), r * np.sin(theta)
    output = trace_wall_hits_twall_field(grid, R, Z, 0.0, 10, step, _box_wall(PLATE_DEPTH))
    assert np.all(output["term_plus"] == 1) and np.all(output["term_minus"] == 1)
    seeds = jnp.stack([jnp.asarray(R), jnp.zeros(3), jnp.asarray(Z)], axis=1)
    lengths, points = map(np.asarray, circular_tokamak_plate_solution(field, PLATE_DEPTH, seeds))
    native = np.stack([output["Lc_plus"], output["Lc_minus"]], axis=1)
    length_error = np.max(np.abs(native / lengths - 1.0))
    hits = np.stack([output["hit_plus"], output["hit_minus"]], axis=1)
    exact_R = np.hypot(points[..., 0], points[..., 1])
    position_error = np.max(np.hypot(hits[..., 0] - exact_R, hits[..., 1] - points[..., 2]))
    return length_error, position_error


def test_native_wall_hits_converge_to_circular_tokamak_plate_solution():
    coarse = _relative_errors(n_grid=33, step=0.01)
    fine = _relative_errors(n_grid=129, step=0.0025)
    assert fine[0] < 1e-4 and fine[1] < 1e-4
    assert fine[0] < coarse[0] / 4 and fine[1] < coarse[1] / 4
