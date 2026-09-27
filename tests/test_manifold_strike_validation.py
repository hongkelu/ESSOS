"""End-to-end production refresh for one exact ESSOS strike label."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("joblib")
pytest.importorskip("pyna.topo.manifold_strike")

from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.manifold_strike import local_wall_plane_from_strike
from pyna.topo.manifold_strike_contracts import (
    ManifoldStrikeLabel,
    ManifoldStrikeMatch,
)
from pyna.toroidal.geometry import ToroidalWall

from essos.topology_objectives import ManifoldStrikeStage2Target
from essos.topology_validation import validate_manifold_strike_candidate


RATE = 0.4
CENTER_R = 1.0
CENTER_Z = 0.0
PHI_START = 0.1
MAP_SPAN = 0.5
SEED_DISTANCE = 0.01
WALL_OUTER_R = 1.2
SEED_ORDER = 7


@jax.tree_util.register_pytree_node_class
class _ShiftedHyperbolicField:
    def __init__(self, parameters):
        self.parameters = jnp.asarray(parameters)

    def B(self, xyz):
        rate, center_r, center_z = self.parameters
        x, y, z = xyz
        radius = jnp.sqrt(x * x + y * y)
        cos_phi = x / radius
        sin_phi = y / radius
        b_r = rate * (radius - center_r)
        b_z = -rate * (z - center_z)
        b_phi = radius
        return jnp.array(
            [
                b_r * cos_phi - b_phi * sin_phi,
                b_r * sin_phi + b_phi * cos_phi,
                b_z,
            ]
        )

    def tree_flatten(self):
        return (self.parameters,), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        return cls(children[0])


def _wall(n_phi=8, n_pol=64):
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pol, endpoint=False)
    R = np.broadcast_to(1.0 + 0.2 * np.cos(theta), (n_phi, n_pol)).copy()
    Z = np.broadcast_to(0.2 * np.sin(theta), (n_phi, n_pol)).copy()
    return ToroidalWall(phi, R, Z)


def _branch(parameters):
    rate, center_r, center_z = map(float, parameters)
    seed = np.array([center_r + SEED_DISTANCE, center_z])
    return ManifoldBranchReference(
        origin_label="1:P0",
        stability="unstable",
        seed_side=1,
        section_phi=PHI_START,
        map_span=MAP_SPAN,
        anchor_RZ_m=np.array([center_r, center_z]),
        direction_RZ=np.array([1.0, 0.0]),
        expansion=np.exp(rate * MAP_SPAN),
        orientation_reversing=False,
        seed_orders=np.array([SEED_ORDER]),
        seed_distances_m=np.array([SEED_DISTANCE]),
        points_RZ_m=seed.reshape(1, 1, 2),
        available=np.ones((1, 1), dtype=bool),
    )


def _hit_phi(parameters, hit_radius=WALL_OUTER_R):
    rate, center_r, _center_z = map(float, parameters)
    return PHI_START + np.log(
        (hit_radius - center_r) / SEED_DISTANCE
    ) / rate


def _target_state():
    parameters = np.array([RATE, CENTER_R, CENTER_Z])
    branch = _branch(parameters)
    hit_phi = _hit_phi(parameters)
    label = ManifoldStrikeLabel(branch.label, "+", SEED_ORDER)
    match = ManifoldStrikeMatch(
        label=label,
        point_RZPhi_m_rad=np.array([WALL_OUTER_R, CENTER_Z, hit_phi]),
        target_distance_m=0.0,
        connection_length_m=WALL_OUTER_R * (hit_phi - PHI_START),
        bundle_seed_index=0,
        distance_mode="rz",
    )
    plane = local_wall_plane_from_strike(
        match,
        _wall(),
        maximum_projection_distance_m=1.0e-12,
    )
    return ManifoldStrikeStage2Target(
        branch_reference=branch,
        strike_match=match,
        wall_plane=plane,
        target_position_m=np.array([WALL_OUTER_R, CENTER_Z]),
        position_scales_m=np.array([0.01, 0.01]),
        maximum_phi_shift=0.25,
        n_steps_per_span=64,
        xline_newton_iterations=4,
        wall_n_steps=512,
        wall_newton_iterations=6,
    )


def _fake_trace(*, radial_offset=0.0, resolved=True):
    def trace(
        field,
        R,
        Z,
        phi_start,
        max_turns,
        DPhi,
        wall_phi,
        wall_R,
        wall_Z,
        *,
        extend_phi,
        direction,
    ):
        del max_turns, DPhi, wall_phi, wall_R, wall_Z, extend_phi
        assert direction == "+"
        parameters = np.asarray(field.parameters, dtype=float)
        rate, center_r, _center_z = parameters
        hit_radius = WALL_OUTER_R + radial_offset
        phi = phi_start + np.log((hit_radius - center_r) / (R - center_r)) / rate
        return {
            "Lc_plus": hit_radius * (phi - phi_start),
            "hit_plus": np.column_stack(
                (np.full(R.size, hit_radius), Z, phi)
            ),
            "term_plus": np.full(R.size, 1 if resolved else 2),
        }

    return trace


def _validate(candidate, candidate_branch, **overrides):
    options = {
        "maximum_hit_displacement_m": 2.0e-3,
        "maximum_projection_distance_m": 2.0e-3,
        "jax_cyna_tolerance_m": 2.0e-8,
        "max_turns": 4,
        "production_DPhi": 0.01,
        "production_trace_function": _fake_trace(),
    }
    options.update(overrides)
    return validate_manifold_strike_candidate(
        candidate,
        candidate,
        candidate_branch,
        _target_state(),
        _wall(),
        **options,
    )


def test_strike_candidate_refreshes_every_gate_and_immutable_target():
    candidate_parameters = np.array([0.405, 1.0001, 0.0])
    candidate = _ShiftedHyperbolicField(jnp.asarray(candidate_parameters))
    candidate_branch = _branch(candidate_parameters)
    initial_target = _target_state()
    report = _validate(candidate, candidate_branch)

    assert report.accepted
    assert report.strike_refresh.accepted
    assert report.correspondence.accepted
    assert report.accepted_target is not None
    assert report.accepted_target.label == initial_target.label
    assert report.accepted_target.branch_reference is candidate_branch
    assert report.accepted_target.strike_match is report.strike_refresh.candidate_match
    assert report.accepted_target.wall_plane is report.wall_plane
    np.testing.assert_allclose(
        report.accepted_target.target_position_m,
        initial_target.target_position_m,
    )
    assert report.diagnostics["jax_cyna_strike_deviation_m"] < 2.0e-8


def test_strike_candidate_rejects_an_unresolved_exact_seed_before_jax():
    parameters = np.array([RATE, CENTER_R, CENTER_Z])
    candidate = _ShiftedHyperbolicField(jnp.asarray(parameters))
    report = _validate(
        candidate,
        _branch(parameters),
        production_trace_function=_fake_trace(resolved=False),
    )

    assert not report.accepted
    assert report.rejection_reason == "strike_refresh:strike_unresolved"
    assert report.wall_plane is None
    assert report.jax_strike_state is None
    assert report.correspondence is None
    assert report.accepted_target is None


def test_strike_candidate_rejects_cyna_jax_position_disagreement():
    parameters = np.array([RATE, CENTER_R, CENTER_Z])
    candidate = _ShiftedHyperbolicField(jnp.asarray(parameters))
    report = _validate(
        candidate,
        _branch(parameters),
        maximum_hit_displacement_m=2.0e-3,
        production_trace_function=_fake_trace(radial_offset=-1.0e-3),
        jax_cyna_tolerance_m=1.0e-4,
    )

    assert not report.accepted
    assert report.strike_refresh.accepted
    assert report.wall_plane is not None
    assert report.jax_strike_state is not None
    assert not report.correspondence.accepted
    assert report.rejection_reason.endswith("strike_deviation_exceeds_tolerance")
    assert report.accepted_target is None
