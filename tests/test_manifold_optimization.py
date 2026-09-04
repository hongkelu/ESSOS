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
    ManifoldStage2Target,
    make_manifold_stage2_loss,
    manifold_stage2_target_loss,
    refresh_manifold_stage2_target,
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
