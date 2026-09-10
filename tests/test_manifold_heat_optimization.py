"""Tests for differentiable, power-conserving manifold heat loading."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.topo.jax_strike")

from pyna.topo.manifold_correspondence import (
    ManifoldBranchReference,
    ManifoldSampleMatch,
    compare_jax_manifold_branch,
    refresh_manifold_sample_match,
)
from pyna.topo.manifold_strike_contracts import (
    LocalWallPlane,
    ManifoldStrikeLabel,
    ManifoldStrikeMatch,
)

from essos.losses import custom_loss
from essos.manifold_leg_optimization import (
    make_manifold_leg_clearance_loss,
    manifold_leg_clearance_loss,
    manifold_leg_clearance_state,
)
from essos.manifold_heat_optimization import (
    ManifoldHeatStage2Target,
    make_manifold_heat_stage2_loss,
    manifold_heat_flux_state,
    manifold_heat_stage2_limit_loss,
    power_conserving_gaussian_heat_flux,
)
from essos.manifold_optimization import (
    ManifoldContinuationSchedule,
    ManifoldContinuationStage,
    ManifoldContinuationState,
    ManifoldStage2Target,
    accept_manifold_continuation_stage,
    compose_manifold_stage2_loss,
    trace_manifold_reference,
)


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


PARAMETERS = np.array([0.25, 1.1, -0.04])
PHI_SPAN = 0.5
PHI_HIT = 0.8
SEED_ORDERS = np.array([3, 8])
SEED_DISTANCES = np.array([2.0e-4, 4.0e-4])
STRIKE_POWERS_W = np.array([600.0, 400.0])


def _xyz(radius, height, phi=PHI_HIT):
    return np.array(
        [
            radius * np.cos(phi),
            radius * np.sin(phi),
            height,
        ]
    )


def _heat_target(parameters=PARAMETERS):
    rate, center_r, center_z = map(float, parameters)
    factors = np.exp(rate * PHI_SPAN * np.arange(2))
    points = np.empty((2, len(SEED_DISTANCES), 2))
    for generation, factor in enumerate(factors):
        points[generation, :, 0] = center_r + SEED_DISTANCES * factor
        points[generation, :, 1] = center_z
    branch = ManifoldBranchReference(
        origin_label="heat-orbit:P0",
        stability="unstable",
        seed_side=1,
        section_phi=0.0,
        map_span=PHI_SPAN,
        anchor_RZ_m=np.array([center_r, center_z]),
        direction_RZ=np.array([1.0, 0.0]),
        expansion=np.exp(rate * PHI_SPAN),
        orientation_reversing=False,
        seed_orders=SEED_ORDERS,
        seed_distances_m=SEED_DISTANCES,
        points_RZ_m=points,
        available=np.ones(points.shape[:2], dtype=bool),
    )
    matches = []
    planes = []
    for index, (order, distance) in enumerate(
        zip(SEED_ORDERS, SEED_DISTANCES, strict=True)
    ):
        hit_radius = center_r + distance * np.exp(rate * PHI_HIT)
        hit_xyz = _xyz(hit_radius, center_z)
        label = ManifoldStrikeLabel(branch.label, "+", int(order))
        matches.append(
            ManifoldStrikeMatch(
                label=label,
                point_RZPhi_m_rad=np.array([hit_radius, center_z, PHI_HIT]),
                target_distance_m=0.0,
                connection_length_m=hit_radius * PHI_HIT,
                bundle_seed_index=index,
                distance_mode="xyz",
            )
        )
        planes.append(
            LocalWallPlane(
                strike_label=label,
                point_xyz_m=hit_xyz,
                normal_xyz=np.array([-np.sin(PHI_HIT), np.cos(PHI_HIT), 0.0]),
                wall_phi_rad=PHI_HIT,
                wall_s=0.2 + 0.1 * index,
                projection_distance_m=0.0,
            )
        )

    cell_rz = np.array(
        [
            [center_r - 5.0e-4, center_z],
            [center_r + 1.5e-4, center_z],
            [center_r + 4.5e-4, center_z],
            [center_r + 9.0e-4, center_z + 4.0e-4],
            [center_r + 4.0e-4, center_z - 5.0e-4],
        ]
    )
    cell_centers = np.stack([_xyz(radius, height) for radius, height in cell_rz])
    return ManifoldHeatStage2Target(
        branch_reference=branch,
        strike_matches=tuple(matches),
        wall_planes=tuple(planes),
        strike_powers_W=STRIKE_POWERS_W,
        power_provenance="analytic absolute-power regression",
        wall_cell_centers_xyz_m=cell_centers,
        wall_cell_areas_m2=np.array([0.002, 0.001, 0.003, 0.0015, 0.0025]),
        deposition_width_m=5.0e-4,
        maximum_heat_flux_W_m2=2.0e4,
        heat_flux_scale_W_m2=1.0e5,
        maximum_phi_shift=0.1,
        n_steps_per_span=64,
        xline_newton_iterations=4,
        wall_n_steps=128,
        wall_newton_iterations=4,
    )


def _sample_target(heat_target):
    branch = heat_target.branch_reference
    label = branch.sample_label(1, 0)
    point = branch.point(label)
    return ManifoldStage2Target(
        branch_reference=branch,
        sample_match=ManifoldSampleMatch(
            label=label,
            point_RZ_m=point,
            distance_m=0.0,
            jax_seed_index=0,
        ),
        target_RZ_m=point,
        rz_scales_m=np.array([0.01, 0.01]),
        n_steps_per_span=64,
        newton_iterations=4,
    )


def _leg_wall_signed_distance(point_xyz):
    """Smooth local wall proxy, positive on the allowed pre-strike side."""

    x, y, z = point_xyz
    del z
    radius = jnp.sqrt(x * x + y * y)
    phi = jnp.arctan2(y, x)
    return (
        6.0e-3 * (PHI_HIT - phi)
        + 0.2 * (1.1003 - radius)
        + 2.0e-4
    )


def test_gaussian_deposition_conserves_power_even_for_remote_cells():
    strike_points = jnp.array([[0.0, 0.0, 0.0], [1.0, -0.5, 0.2]])
    powers = jnp.array([30.0, 70.0])
    centers = jnp.array(
        [
            [1000.0, 0.0, 0.0],
            [1000.2, 0.1, 0.0],
            [999.8, -0.1, 0.1],
        ]
    )
    areas = jnp.array([0.5, 2.0, 1.5])

    heat_flux = power_conserving_gaussian_heat_flux(
        strike_points,
        powers,
        centers,
        areas,
        deposition_width_m=1.0e-3,
    )

    assert np.all(np.isfinite(np.asarray(heat_flux)))
    np.testing.assert_allclose(jnp.sum(heat_flux * areas), 100.0, atol=1.0e-13)


def test_heat_state_traces_exact_labels_and_conserves_quantitative_power():
    target = _heat_target()
    state = jax.jit(
        lambda parameters: manifold_heat_flux_state(
            _ShiftedHyperbolicField(parameters),
            target_state=target,
        )
    )(jnp.asarray(PARAMETERS))

    assert bool(state.strike_trace.periodic_point.converged)
    assert np.all(np.asarray(state.strike_trace.intersections.converged))
    np.testing.assert_allclose(
        state.strike_trace.intersections.phi,
        PHI_HIT,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        state.strike_trace.intersections.point_xyz,
        [match.point_xyz_m for match in target.strike_matches],
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        state.deposited_power_W,
        target.total_power_W,
        atol=2.0e-13,
    )


def test_heat_limit_loss_has_live_field_gradient_and_custom_loss():
    target = _heat_target()
    parameters = jnp.asarray(PARAMETERS)

    def objective(dynamic_parameters):
        return manifold_heat_stage2_limit_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            target_state=target,
        )

    value, gradient = jax.value_and_grad(objective)(parameters)
    direction = jnp.array([0.2, -0.3, 0.4])
    delta = 2.0e-7
    finite_difference = (
        objective(parameters + delta * direction)
        - objective(parameters - delta * direction)
    ) / (2.0 * delta)
    assert float(value) > 0.0
    assert np.linalg.norm(np.asarray(gradient)) > 0.0
    np.testing.assert_allclose(
        jnp.dot(gradient, direction),
        finite_difference,
        rtol=3.0e-5,
        atol=3.0e-7,
    )

    loss = make_manifold_heat_stage2_loss(target)
    assert isinstance(loss, custom_loss)
    loss.dependencies = {"field": _ShiftedHyperbolicField(parameters)}
    custom_value, custom_gradient = loss.value_and_grad(loss.starting_dofs)
    np.testing.assert_allclose(custom_value, value, rtol=2.0e-12)
    np.testing.assert_allclose(custom_gradient, gradient, rtol=2.0e-10)


def test_leg_clearance_uses_fixed_pre_strike_samples_and_excludes_wall_hit():
    target = _heat_target()
    field = _ShiftedHyperbolicField(jnp.asarray(PARAMETERS))
    near_wall = manifold_leg_clearance_state(
        field,
        _leg_wall_signed_distance,
        target_state=target,
        minimum_clearance_m=1.0e-3,
        clearance_scale_m=1.0e-3,
        terminal_exclusion_fraction=0.01,
    )
    protected = manifold_leg_clearance_state(
        field,
        _leg_wall_signed_distance,
        target_state=target,
        minimum_clearance_m=1.0e-3,
        clearance_scale_m=1.0e-3,
        terminal_exclusion_fraction=0.2,
    )

    assert near_wall.trajectories.point_xyz.shape == (2, 129, 3)
    assert near_wall.signed_clearance_m.shape == (2, 127)
    assert protected.signed_clearance_m.shape == (2, 103)
    assert np.any(np.asarray(near_wall.normalized_violation) > 0.0)
    np.testing.assert_allclose(protected.normalized_violation, 0.0, atol=0.0)
    np.testing.assert_allclose(
        near_wall.trajectories.point_xyz[:, -1],
        [match.point_xyz_m for match in target.strike_matches],
        atol=2.0e-13,
    )
    assert not np.allclose(
        protected.trajectories.point_xyz[:, 102],
        protected.trajectories.point_xyz[:, -1],
    )


def test_leg_clearance_loss_has_live_field_gradient_and_custom_loss():
    target = _heat_target()
    parameters = jnp.asarray(PARAMETERS)

    def objective(dynamic_parameters):
        return manifold_leg_clearance_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            wall_signed_distance=_leg_wall_signed_distance,
            target_state=target,
            minimum_clearance_m=1.0e-3,
            clearance_scale_m=1.0e-3,
            terminal_exclusion_fraction=0.05,
            strike_weights=target.strike_powers_W,
        )

    value, gradient = jax.value_and_grad(objective)(parameters)
    direction = jnp.array([0.2, -0.3, 0.4])
    delta = 2.0e-7
    finite_difference = (
        objective(parameters + delta * direction)
        - objective(parameters - delta * direction)
    ) / (2.0 * delta)
    assert float(value) > 0.0
    assert np.linalg.norm(np.asarray(gradient)) > 0.0
    np.testing.assert_allclose(
        jnp.dot(gradient, direction),
        finite_difference,
        rtol=3.0e-5,
        atol=3.0e-7,
    )

    loss = make_manifold_leg_clearance_loss(
        target,
        _leg_wall_signed_distance,
        minimum_clearance_m=1.0e-3,
        clearance_scale_m=1.0e-3,
        terminal_exclusion_fraction=0.05,
        strike_weights=target.strike_powers_W,
    )
    assert isinstance(loss, custom_loss)
    assert loss.args_names == ("field",)
    loss.dependencies = {"field": _ShiftedHyperbolicField(parameters)}
    custom_value, custom_gradient = loss.value_and_grad(loss.starting_dofs)
    np.testing.assert_allclose(custom_value, value, rtol=2.0e-12)
    np.testing.assert_allclose(custom_gradient, gradient, rtol=2.0e-10)


def test_leg_clearance_rejects_invalid_static_configuration():
    target = _heat_target()
    field = _ShiftedHyperbolicField(jnp.asarray(PARAMETERS))
    common = dict(
        target_state=target,
        minimum_clearance_m=1.0e-3,
        clearance_scale_m=1.0e-3,
        terminal_exclusion_fraction=0.1,
    )

    with pytest.raises(ValueError, match="terminal_exclusion_fraction"):
        manifold_leg_clearance_state(
            field,
            _leg_wall_signed_distance,
            **{**common, "terminal_exclusion_fraction": 0.0},
        )
    with pytest.raises(ValueError, match="clearance_scale_m"):
        manifold_leg_clearance_state(
            field,
            _leg_wall_signed_distance,
            **{**common, "clearance_scale_m": 0.0},
        )
    with pytest.raises(ValueError, match="return a scalar"):
        manifold_leg_clearance_state(
            field,
            lambda point_xyz: point_xyz[:2],
            **common,
        )
    with pytest.raises(ValueError, match="one value per strike"):
        manifold_leg_clearance_loss(
            field,
            wall_signed_distance=_leg_wall_signed_distance,
            strike_weights=np.ones(3),
            **common,
        )


def test_continuation_composes_heat_weight_and_omits_zero_weight_trace():
    heat_target = _heat_target()
    sample_target = _sample_target(heat_target)
    parameters = jnp.asarray(PARAMETERS)
    field = _ShiftedHyperbolicField(parameters)
    heat_weight = 0.35
    active = ManifoldContinuationState(
        ManifoldContinuationSchedule(
            (ManifoldContinuationStage(name="heat", heat_weight=heat_weight),)
        ),
        sample_target,
        heat_target_state=heat_target,
    )
    base = custom_loss(
        lambda dynamic_field: dynamic_field.parameters[0] ** 2,
        "field",
    )
    total = compose_manifold_stage2_loss(
        base,
        active,
        dependencies={"field": field},
    )
    expected = lambda dynamic_parameters: (
        dynamic_parameters[0] ** 2
        + heat_weight
        * manifold_heat_stage2_limit_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            target_state=heat_target,
        )
    )
    value = total(total.starting_dofs)
    gradient = total.grad(total.starting_dofs)
    expected_value, expected_gradient = jax.value_and_grad(expected)(parameters)
    np.testing.assert_allclose(value, expected_value, rtol=2.0e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2.0e-10)

    without_target = replace(active, heat_target_state=None)
    with pytest.raises(ValueError, match="heat_target_state"):
        compose_manifold_stage2_loss(
            base,
            without_target,
            dependencies={"field": field},
        )
    inactive = ManifoldContinuationState(
        ManifoldContinuationSchedule((ManifoldContinuationStage(name="base"),)),
        sample_target,
    )
    base_only = compose_manifold_stage2_loss(
        base,
        inactive,
        dependencies={"field": field},
    )
    np.testing.assert_allclose(base_only(base_only.starting_dofs), parameters[0] ** 2)
    with pytest.raises(ValueError, match="non-negative"):
        ManifoldContinuationStage(name="bad-heat", heat_weight=-1.0)


def test_continuation_requires_accepted_heat_refresh_and_fixed_physics():
    heat_target = _heat_target()
    sample_target = _sample_target(heat_target)
    state = ManifoldContinuationState(
        ManifoldContinuationSchedule(
            (
                ManifoldContinuationStage(name="base"),
                ManifoldContinuationStage(name="heat", heat_weight=0.1),
            )
        ),
        sample_target,
        heat_target_state=heat_target,
    )
    field = _ShiftedHyperbolicField(jnp.asarray(PARAMETERS))
    candidate_branch = heat_target.branch_reference
    candidate_trace = trace_manifold_reference(
        field,
        candidate_branch,
        n_steps_per_span=64,
        newton_iterations=4,
    )
    correspondence = compare_jax_manifold_branch(
        candidate_branch,
        candidate_trace.generations,
        absolute_tolerance_m=3.0e-12,
    )
    sample_refresh = refresh_manifold_sample_match(
        sample_target.sample_match,
        candidate_branch,
        sample_target.target_RZ_m,
        maximum_sample_displacement_m=1.0e-6,
    )

    with pytest.raises(ValueError, match="heat target requires accepted refresh"):
        accept_manifold_continuation_stage(
            state,
            candidate_branch,
            sample_refresh,
            correspondence,
        )
    accepted = accept_manifold_continuation_stage(
        state,
        candidate_branch,
        sample_refresh,
        correspondence,
        accepted_heat_target=heat_target,
    )
    assert accepted.stage.name == "heat"
    assert accepted.heat_target_state is heat_target

    changed_power = replace(
        heat_target,
        strike_powers_W=np.array([500.0, 500.0]),
    )
    with pytest.raises(ValueError, match="strike_powers_W"):
        accept_manifold_continuation_stage(
            state,
            candidate_branch,
            sample_refresh,
            correspondence,
            accepted_heat_target=changed_power,
        )


def test_heat_target_requires_aligned_labels_and_power_provenance():
    target = _heat_target()
    with pytest.raises(ValueError, match="power_provenance"):
        ManifoldHeatStage2Target(
            **{
                **target.__dict__,
                "power_provenance": "",
            }
        )
    with pytest.raises(ValueError, match="one value per strike"):
        ManifoldHeatStage2Target(
            **{
                **target.__dict__,
                "strike_powers_W": np.ones(3),
            }
        )
    with pytest.raises(ValueError, match="unique"):
        ManifoldHeatStage2Target(
            **{
                **target.__dict__,
                "strike_matches": (
                    target.strike_matches[0],
                    target.strike_matches[0],
                ),
                "wall_planes": (
                    target.wall_planes[0],
                    target.wall_planes[0],
                ),
            }
        )
    with pytest.raises(ValueError, match="wall_cell_areas_m2"):
        ManifoldHeatStage2Target(
            **{
                **target.__dict__,
                "wall_cell_areas_m2": np.array([1.0, -1.0, 1.0, 1.0, 1.0]),
            }
        )
