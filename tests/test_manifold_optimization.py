"""Tests for immutable manifold-aware ESSOS Stage-2 optimizer state."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.topo.jax_manifold")

from pyna.topo.manifold_correspondence import (
    ManifoldBranchReference,
    ManifoldSampleMatch,
    compare_jax_manifold_branch,
    refresh_manifold_sample_match,
)

from essos.manifold_optimization import (
    ManifoldContinuationSchedule,
    ManifoldContinuationStage,
    ManifoldContinuationState,
    ManifoldStage2Target,
    accept_manifold_continuation_stage,
    compose_manifold_stage2_loss,
    make_manifold_stage2_loss,
    manifold_stage2_target_loss,
    refresh_manifold_stage2_target,
    trace_manifold_reference,
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
N_GENERATIONS = 3
SEED_DISTANCE = 2.0e-4
TARGET_RZ = np.array([1.101, -0.04])
RZ_SCALES = np.array([0.01, 0.01])


def _branch_reference(parameters):
    rate, center_r, center_z = map(float, parameters)
    factors = np.exp(rate * PHI_SPAN * np.arange(N_GENERATIONS + 1))
    points = np.empty((N_GENERATIONS + 1, 1, 2))
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


def _sample_match(branch):
    label = branch.sample_label(N_GENERATIONS, 0)
    point = branch.point(label)
    return ManifoldSampleMatch(
        label=label,
        point_RZ_m=point,
        distance_m=float(np.linalg.norm(point - TARGET_RZ)),
        jax_seed_index=0,
    )


def _target_state(parameters):
    branch = _branch_reference(parameters)
    return ManifoldStage2Target(
        branch_reference=branch,
        sample_match=_sample_match(branch),
        target_RZ_m=TARGET_RZ,
        rz_scales_m=RZ_SCALES,
        n_steps_per_span=128,
        newton_iterations=4,
    )


def test_target_state_traces_reference_and_has_analytic_gradient():
    parameters = jnp.array([0.25, 1.1, -0.04])
    target_state = _target_state(parameters)
    trace = trace_manifold_reference(
        _ShiftedHyperbolicField(parameters),
        target_state.branch_reference,
        n_steps_per_span=128,
        newton_iterations=4,
    )
    parity = compare_jax_manifold_branch(
        target_state.branch_reference,
        trace.generations,
        absolute_tolerance_m=3.0e-12,
    )
    assert parity.accepted

    def objective(dynamic_parameters):
        return manifold_stage2_target_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            target_state=target_state,
        )

    rate = float(parameters[0])
    exponential = np.exp(rate * PHI_SPAN * N_GENERATIONS)
    endpoint_r = float(parameters[1]) + SEED_DISTANCE * exponential
    residual_r = (endpoint_r - TARGET_RZ[0]) / RZ_SCALES[0]
    endpoint_rate_derivative = (
        SEED_DISTANCE * PHI_SPAN * N_GENERATIONS * exponential
    )
    expected = np.array(
        [
            residual_r * endpoint_rate_derivative / RZ_SCALES[0],
            residual_r / RZ_SCALES[0],
            0.0,
        ]
    )
    value, gradient = jax.value_and_grad(objective)(parameters)
    np.testing.assert_allclose(value, 0.5 * residual_r**2, rtol=2.0e-11)
    np.testing.assert_allclose(gradient, expected, rtol=3.0e-11, atol=2.0e-12)


def test_target_state_builds_a_jitted_essos_custom_loss():
    parameters = jnp.array([0.25, 1.1, -0.04])
    target_state = _target_state(parameters)
    field = _ShiftedHyperbolicField(parameters)
    loss = make_manifold_stage2_loss(target_state)
    loss.dependencies = {"field": field}

    value, gradient = loss.value_and_grad(loss.starting_dofs)
    direct_value, direct_gradient = jax.value_and_grad(
        lambda dynamic_parameters: manifold_stage2_target_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            target_state=target_state,
        )
    )(parameters)
    np.testing.assert_allclose(value, direct_value, rtol=2.0e-13)
    np.testing.assert_allclose(gradient, direct_gradient, rtol=2.0e-11)


def test_target_refresh_requires_both_pyna_acceptance_gates():
    initial_parameters = np.array([0.25, 1.1, -0.04])
    candidate_parameters = np.array([0.26, 1.1002, -0.04])
    target_state = _target_state(initial_parameters)
    candidate_branch = _branch_reference(candidate_parameters)
    candidate_trace = trace_manifold_reference(
        _ShiftedHyperbolicField(jnp.asarray(candidate_parameters)),
        candidate_branch,
        n_steps_per_span=128,
        newton_iterations=4,
    )
    correspondence = compare_jax_manifold_branch(
        candidate_branch,
        candidate_trace.generations,
        absolute_tolerance_m=3.0e-12,
    )
    sample_refresh = refresh_manifold_sample_match(
        target_state.sample_match,
        candidate_branch,
        TARGET_RZ,
        maximum_sample_displacement_m=2.0e-3,
    )
    refreshed = refresh_manifold_stage2_target(
        target_state,
        candidate_branch,
        sample_refresh,
        correspondence,
    )

    assert refreshed.label == target_state.label
    assert refreshed.branch_reference is candidate_branch
    assert refreshed.sample_match is sample_refresh.candidate_match
    np.testing.assert_allclose(refreshed.target_RZ_m, target_state.target_RZ_m)

    with pytest.raises(ValueError, match="correspondence"):
        refresh_manifold_stage2_target(
            target_state,
            candidate_branch,
            sample_refresh,
            replace(correspondence, accepted=False),
        )

    rejected_sample = refresh_manifold_sample_match(
        target_state.sample_match,
        candidate_branch,
        TARGET_RZ,
        maximum_sample_displacement_m=1.0e-8,
    )
    with pytest.raises(ValueError, match="sample refresh"):
        refresh_manifold_stage2_target(
            target_state,
            candidate_branch,
            rejected_sample,
            correspondence,
        )


def test_target_state_rejects_inconsistent_seed_index_and_scales():
    branch = _branch_reference(np.array([0.25, 1.1, -0.04]))
    match = _sample_match(branch)
    with pytest.raises(ValueError, match="seed index"):
        ManifoldStage2Target(
            branch_reference=branch,
            sample_match=replace(match, jax_seed_index=1),
            target_RZ_m=TARGET_RZ,
            rz_scales_m=RZ_SCALES,
        )
    with pytest.raises(ValueError, match="must be positive"):
        ManifoldStage2Target(
            branch_reference=branch,
            sample_match=match,
            target_RZ_m=TARGET_RZ,
            rz_scales_m=np.array([0.0, 0.01]),
        )


def _continuation_schedule():
    return ManifoldContinuationSchedule(
        (
            ManifoldContinuationStage(name="base"),
            ManifoldContinuationStage(
                name="topology",
                return_map_weight=0.3,
                xline_weight=0.2,
                xline_clearance_weight=0.1,
                manifold_weight=0.4,
            ),
        )
    )


def test_continuation_schedule_is_immutable_and_validated():
    schedule = _continuation_schedule()
    state = ManifoldContinuationState(schedule, _target_state([0.25, 1.1, -0.04]))

    assert isinstance(schedule.stages, tuple)
    assert state.stage.name == "base"
    assert not state.stage.topology_active
    assert schedule[1].topology_active
    assert not state.final_stage

    with pytest.raises(ValueError, match="non-negative"):
        ManifoldContinuationStage(name="bad", manifold_weight=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        ManifoldContinuationStage(name="bad-clearance", xline_clearance_weight=-1.0)
    with pytest.raises(ValueError, match="unique"):
        ManifoldContinuationSchedule(
            (
                ManifoldContinuationStage(name="same"),
                ManifoldContinuationStage(name="same"),
            )
        )
    with pytest.raises(IndexError, match="stage_index"):
        ManifoldContinuationState(schedule, state.target_state, stage_index=2)


def test_continuation_composes_weighted_value_and_gradient():
    parameters = jnp.array([0.25, 1.1, -0.04])
    field = _ShiftedHyperbolicField(parameters)
    target_state = _target_state(parameters)
    state = ManifoldContinuationState(
        _continuation_schedule(),
        target_state,
        stage_index=1,
    )
    base_target = jnp.array([0.2, 1.0, 0.0])

    def base_objective(dynamic_field):
        return 0.5 * jnp.sum((dynamic_field.parameters - base_target) ** 2)

    def return_map_objective(dynamic_field):
        return dynamic_field.parameters[0] ** 2

    def xline_r_objective(dynamic_field):
        return dynamic_field.parameters[1] ** 2

    def xline_z_objective(dynamic_field):
        return 2.0 * dynamic_field.parameters[2] ** 2

    def xline_clearance_objective(dynamic_field):
        return 3.0 * dynamic_field.parameters[1] ** 2

    base = custom_loss(base_objective, "field")
    return_map = custom_loss(return_map_objective, "field")
    xline = custom_loss(xline_r_objective, "field") + custom_loss(
        xline_z_objective,
        "field",
    )
    xline_clearance = custom_loss(xline_clearance_objective, "field")
    total = compose_manifold_stage2_loss(
        base,
        state,
        return_map_loss=return_map,
        xline_loss=xline,
        xline_clearance_loss=xline_clearance,
        dependencies={"field": field},
    )

    def expected(dynamic_parameters):
        dynamic_field = _ShiftedHyperbolicField(dynamic_parameters)
        return (
            base_objective(dynamic_field)
            + 0.3 * return_map_objective(dynamic_field)
            + 0.2
            * (xline_r_objective(dynamic_field) + xline_z_objective(dynamic_field))
            + 0.1 * xline_clearance_objective(dynamic_field)
            + 0.4
            * manifold_stage2_target_loss(
                dynamic_field,
                target_state=target_state,
            )
        )

    value = total(total.starting_dofs)
    gradient = total.grad(total.starting_dofs)
    expected_value, expected_gradient = jax.value_and_grad(expected)(parameters)
    np.testing.assert_allclose(value, expected_value, rtol=2.0e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3.0e-11)


def test_continuation_omits_inactive_terms_and_requires_active_ones():
    parameters = jnp.array([0.25, 1.1, -0.04])
    field = _ShiftedHyperbolicField(parameters)
    target_state = _target_state(parameters)
    base = custom_loss(lambda dynamic_field: dynamic_field.parameters[0] ** 2, "field")
    inactive_state = ManifoldContinuationState(_continuation_schedule(), target_state)
    total = compose_manifold_stage2_loss(
        base,
        inactive_state,
        dependencies={"field": field},
    )
    np.testing.assert_allclose(total(total.starting_dofs), parameters[0] ** 2)

    active_state = replace(inactive_state, stage_index=1)
    with pytest.raises(ValueError, match="return_map_loss"):
        compose_manifold_stage2_loss(
            base,
            active_state,
            dependencies={"field": field},
        )

    clearance_only_state = ManifoldContinuationState(
        ManifoldContinuationSchedule(
            (
                ManifoldContinuationStage(
                    name="clearance",
                    xline_clearance_weight=1.0,
                ),
            )
        ),
        target_state,
    )
    with pytest.raises(ValueError, match="xline_clearance_loss"):
        compose_manifold_stage2_loss(
            base,
            clearance_only_state,
            dependencies={"field": field},
        )


def test_continuation_advances_only_after_both_pyna_gates_accept():
    initial_parameters = np.array([0.25, 1.1, -0.04])
    candidate_parameters = np.array([0.26, 1.1002, -0.04])
    state = ManifoldContinuationState(
        _continuation_schedule(),
        _target_state(initial_parameters),
    )
    candidate_branch = _branch_reference(candidate_parameters)
    candidate_trace = trace_manifold_reference(
        _ShiftedHyperbolicField(jnp.asarray(candidate_parameters)),
        candidate_branch,
        n_steps_per_span=128,
        newton_iterations=4,
    )
    correspondence = compare_jax_manifold_branch(
        candidate_branch,
        candidate_trace.generations,
        absolute_tolerance_m=3.0e-12,
    )
    sample_refresh = refresh_manifold_sample_match(
        state.target_state.sample_match,
        candidate_branch,
        TARGET_RZ,
        maximum_sample_displacement_m=2.0e-3,
    )

    accepted = accept_manifold_continuation_stage(
        state,
        candidate_branch,
        sample_refresh,
        correspondence,
    )
    assert accepted.stage_index == 1
    assert accepted.stage.name == "topology"
    assert accepted.target_state.label == state.target_state.label
    assert accepted.final_stage

    with pytest.raises(ValueError, match="correspondence"):
        accept_manifold_continuation_stage(
            state,
            candidate_branch,
            sample_refresh,
            replace(correspondence, accepted=False),
        )
    assert state.stage_index == 0
