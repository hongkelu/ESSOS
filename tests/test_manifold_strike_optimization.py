"""Tests for differentiable PyNA-labelled strike targets in ESSOS."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.topo.jax_strike")

from pyna.topo.manifold_correspondence import (
    ManifoldBranchReference,
    ManifoldSampleMatch,
)
from pyna.topo.manifold_strike_contracts import (
    LocalWallPlane,
    ManifoldStrikeLabel,
    ManifoldStrikeMatch,
)

from essos.manifold_strike_optimization import (
    ManifoldStrikeStage2Target,
    make_manifold_strike_stage2_loss,
    manifold_strike_stage2_target_loss,
    trace_manifold_strike_reference,
)
from essos.manifold_optimization import (
    ManifoldContinuationSchedule,
    ManifoldContinuationStage,
    ManifoldContinuationState,
    ManifoldStage2Target,
    compose_manifold_stage2_loss,
)
from essos.losses import custom_loss


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


PHI_SPAN = 0.5
PHI_HIT = 0.8
SEED_DISTANCE = 2.0e-4
PARAMETERS = np.array([0.25, 1.1, -0.04])
TARGET_RADIUS = 1.102
TARGET_Z = -0.035


def _branch_reference(parameters):
    rate, center_r, center_z = map(float, parameters)
    factors = np.exp(rate * PHI_SPAN * np.arange(2))
    points = np.empty((2, 1, 2))
    points[:, 0, 0] = center_r + SEED_DISTANCE * factors
    points[:, 0, 1] = center_z
    return ManifoldBranchReference(
        origin_label="orbit3:P0",
        stability="unstable",
        seed_side=1,
        section_phi=0.0,
        map_span=PHI_SPAN,
        anchor_RZ_m=np.array([center_r, center_z]),
        direction_RZ=np.array([1.0, 0.0]),
        expansion=np.exp(rate * PHI_SPAN),
        orientation_reversing=False,
        seed_orders=np.array([0]),
        seed_distances_m=np.array([SEED_DISTANCE]),
        points_RZ_m=points,
        available=np.ones(points.shape[:2], dtype=bool),
    )


def _target_geometry(distance_mode):
    if distance_mode == "rz":
        return (
            np.array([TARGET_RADIUS, TARGET_Z]),
            np.array([0.01, 0.02]),
        )
    return (
        np.array(
            [
                TARGET_RADIUS * np.cos(PHI_HIT),
                TARGET_RADIUS * np.sin(PHI_HIT),
                TARGET_Z,
            ]
        ),
        np.array([0.01, 0.01, 0.02]),
    )


def _target_state(parameters, distance_mode="rz"):
    rate, center_r, center_z = map(float, parameters)
    branch = _branch_reference(parameters)
    hit_radius = center_r + SEED_DISTANCE * np.exp(rate * PHI_HIT)
    hit_rzphi = np.array([hit_radius, center_z, PHI_HIT])
    hit_xyz = np.array(
        [
            hit_radius * np.cos(PHI_HIT),
            hit_radius * np.sin(PHI_HIT),
            center_z,
        ]
    )
    label = ManifoldStrikeLabel(branch.label, "+", 0)
    target, scales = _target_geometry(distance_mode)
    hit_coordinates = hit_rzphi[:2] if distance_mode == "rz" else hit_xyz
    match = ManifoldStrikeMatch(
        label=label,
        point_RZPhi_m_rad=hit_rzphi,
        target_distance_m=float(np.linalg.norm(hit_coordinates - target)),
        connection_length_m=hit_radius * PHI_HIT,
        bundle_seed_index=0,
        distance_mode=distance_mode,
    )
    wall_plane = LocalWallPlane(
        strike_label=label,
        point_xyz_m=hit_xyz,
        normal_xyz=np.array([-np.sin(PHI_HIT), np.cos(PHI_HIT), 0.0]),
        wall_phi_rad=PHI_HIT,
        wall_s=0.25,
        projection_distance_m=0.0,
    )
    return ManifoldStrikeStage2Target(
        branch_reference=branch,
        strike_match=match,
        wall_plane=wall_plane,
        target_position_m=target,
        position_scales_m=scales,
        maximum_phi_shift=0.1,
        n_steps_per_span=64,
        xline_newton_iterations=4,
        wall_n_steps=128,
        wall_newton_iterations=4,
    )


def _sample_target_state(parameters):
    branch = _branch_reference(parameters)
    label = branch.sample_label(1, 0)
    point = branch.point(label)
    match = ManifoldSampleMatch(
        label=label,
        point_RZ_m=point,
        distance_m=0.0,
        jax_seed_index=0,
    )
    return ManifoldStage2Target(
        branch_reference=branch,
        sample_match=match,
        target_RZ_m=point,
        rz_scales_m=np.array([0.01, 0.01]),
        n_steps_per_span=64,
        newton_iterations=4,
    )


def _analytic_loss(parameters, distance_mode):
    rate, center_r, center_z = parameters
    radius = center_r + SEED_DISTANCE * jnp.exp(rate * PHI_HIT)
    if distance_mode == "rz":
        point = jnp.array([radius, center_z])
    else:
        point = jnp.array(
            [
                radius * jnp.cos(PHI_HIT),
                radius * jnp.sin(PHI_HIT),
                center_z,
            ]
        )
    target, scales = _target_geometry(distance_mode)
    residual = (point - jnp.asarray(target)) / jnp.asarray(scales)
    return 0.5 * jnp.sum(residual * residual)


@pytest.mark.parametrize("distance_mode", ["rz", "xyz"])
def test_strike_target_has_analytic_live_field_gradient(distance_mode):
    parameters = jnp.asarray(PARAMETERS)
    target_state = _target_state(PARAMETERS, distance_mode)
    trace = trace_manifold_strike_reference(
        _ShiftedHyperbolicField(parameters),
        target_state,
    )

    assert bool(trace.periodic_point.converged)
    assert bool(trace.intersection.converged)
    np.testing.assert_allclose(trace.intersection.phi, PHI_HIT, atol=2.0e-13)

    objective = lambda dynamic_parameters: manifold_strike_stage2_target_loss(
        _ShiftedHyperbolicField(dynamic_parameters),
        target_state=target_state,
    )
    value, gradient = jax.value_and_grad(objective)(parameters)
    expected_value, expected_gradient = jax.value_and_grad(
        lambda dynamic_parameters: _analytic_loss(
            dynamic_parameters,
            distance_mode,
        )
    )(parameters)
    np.testing.assert_allclose(value, expected_value, rtol=3.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(
        gradient,
        expected_gradient,
        rtol=5.0e-10,
        atol=2.0e-11,
    )


def test_strike_target_builds_a_jitted_essos_custom_loss():
    parameters = jnp.asarray(PARAMETERS)
    target_state = _target_state(PARAMETERS)
    field = _ShiftedHyperbolicField(parameters)
    loss = make_manifold_strike_stage2_loss(target_state)
    loss.dependencies = {"field": field}

    value, gradient = loss.value_and_grad(loss.starting_dofs)
    expected_value, expected_gradient = jax.value_and_grad(
        lambda dynamic_parameters: manifold_strike_stage2_target_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            target_state=target_state,
        )
    )(parameters)
    np.testing.assert_allclose(value, expected_value, rtol=3.0e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=5.0e-10)


def test_continuation_composes_the_fixed_label_strike_value_and_gradient():
    parameters = jnp.asarray(PARAMETERS)
    field = _ShiftedHyperbolicField(parameters)
    strike_target = _target_state(PARAMETERS)
    strike_weight = 0.3
    continuation = ManifoldContinuationState(
        ManifoldContinuationSchedule(
            (
                ManifoldContinuationStage(
                    name="strike",
                    strike_weight=strike_weight,
                ),
            )
        ),
        _sample_target_state(PARAMETERS),
        strike_target_state=strike_target,
    )
    base = custom_loss(
        lambda dynamic_field: dynamic_field.parameters[0] ** 2,
        "field",
    )
    total = compose_manifold_stage2_loss(
        base,
        continuation,
        dependencies={"field": field},
    )

    expected = lambda dynamic_parameters: (
        dynamic_parameters[0] ** 2
        + strike_weight
        * manifold_strike_stage2_target_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            target_state=strike_target,
        )
    )
    value = total(total.starting_dofs)
    gradient = total.grad(total.starting_dofs)
    expected_value, expected_gradient = jax.value_and_grad(expected)(parameters)
    np.testing.assert_allclose(value, expected_value, rtol=3.0e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=5.0e-10)

    without_strike = ManifoldContinuationState(
        continuation.schedule,
        continuation.target_state,
    )
    with pytest.raises(ValueError, match="strike_target_state"):
        compose_manifold_stage2_loss(
            base,
            without_strike,
            dependencies={"field": field},
        )


def test_strike_target_validates_metric_shape_and_label_identity():
    target_state = _target_state(PARAMETERS)
    with pytest.raises(ValueError, match="length-2"):
        ManifoldStrikeStage2Target(
            **{
                **target_state.__dict__,
                "target_position_m": np.zeros(3),
            }
        )

    other_label = ManifoldStrikeLabel("other-branch", "+", 0)
    other_plane = LocalWallPlane(
        strike_label=other_label,
        point_xyz_m=target_state.wall_plane.point_xyz_m,
        normal_xyz=target_state.wall_plane.normal_xyz,
        wall_phi_rad=PHI_HIT,
        wall_s=0.25,
        projection_distance_m=0.0,
    )
    with pytest.raises(ValueError, match="different strike label"):
        ManifoldStrikeStage2Target(
            **{
                **target_state.__dict__,
                "wall_plane": other_plane,
            }
        )
