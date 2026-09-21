"""Tests for PyNA-owned invariant manifolds in ESSOS Stage-2 losses."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.topo.jax_manifold")

from essos.losses import custom_loss
from essos.manifold import (
    manifold_sample_location_loss,
    trace_manifold_branch,
)


@jax.tree_util.register_pytree_node_class
class _ShiftedHyperbolicField:
    """ESSOS-style field with an exactly known hyperbolic section point."""

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


INITIAL_GUESS = jnp.array([1.0, 0.0])
PHI_SPAN = 0.5
N_GENERATIONS = 3
SEED_DISTANCE = 2.0e-4
TARGET = jnp.array([1.101, -0.04])
SCALES = jnp.array([0.01, 0.01])


def _location_objective(parameters):
    return manifold_sample_location_loss(
        _ShiftedHyperbolicField(parameters),
        INITIAL_GUESS,
        jnp.array([1.0, 0.0]),
        jnp.array([SEED_DISTANCE]),
        TARGET,
        stability="unstable",
        side=1,
        generation_index=N_GENERATIONS,
        seed_index=0,
        phi_span=PHI_SPAN,
        n_generations=N_GENERATIONS,
        n_steps_per_span=128,
        newton_iterations=4,
        rz_scales=SCALES,
    )


def test_stable_branch_uses_a_backward_map_anchor():
    parameters = jnp.array([0.3, 1.1, -0.04])
    trace = trace_manifold_branch(
        _ShiftedHyperbolicField(parameters),
        INITIAL_GUESS,
        jnp.array([0.0, 1.0]),
        jnp.array([1.0e-4, 2.0e-4]),
        stability="stable",
        side=1,
        phi_span=0.6,
        n_generations=2,
        n_steps_per_span=128,
        newton_iterations=4,
    )

    np.testing.assert_allclose(trace.anchor, parameters[1:], atol=2.0e-14)
    expected_growth = np.exp(float(parameters[0]) * 0.6 * 2)
    np.testing.assert_allclose(
        trace.generations[-1, :, 1] - parameters[2],
        expected_growth * np.array([1.0e-4, 2.0e-4]),
        rtol=3.0e-12,
        atol=2.0e-14,
    )


def test_fixed_correspondence_location_loss_has_exact_gradient():
    parameters = jnp.array([0.25, 1.1, -0.04])
    rate = float(parameters[0])
    exponential = np.exp(rate * PHI_SPAN * N_GENERATIONS)
    endpoint_r = float(parameters[1]) + SEED_DISTANCE * exponential
    residual_r = (endpoint_r - float(TARGET[0])) / float(SCALES[0])
    derivative_r = (
        SEED_DISTANCE
        * PHI_SPAN
        * N_GENERATIONS
        * exponential
        / float(SCALES[0])
    )
    expected_value = 0.5 * residual_r**2
    expected_rate_gradient = residual_r * derivative_r

    value, gradient = jax.value_and_grad(_location_objective)(parameters)
    np.testing.assert_allclose(value, expected_value, rtol=1.0e-11)
    np.testing.assert_allclose(
        gradient[0],
        expected_rate_gradient,
        rtol=2.0e-11,
    )


def test_manifold_location_loss_composes_with_essos_custom_loss():
    parameters = jnp.array([0.25, 1.1, -0.04])
    field = _ShiftedHyperbolicField(parameters)
    loss = custom_loss(
        manifold_sample_location_loss,
        "field",
        initial_guess=INITIAL_GUESS,
        branch_direction=jnp.array([1.0, 0.0]),
        seed_distances=jnp.array([SEED_DISTANCE]),
        target_rz=TARGET,
        stability="unstable",
        side=1,
        generation_index=N_GENERATIONS,
        seed_index=0,
        phi_span=PHI_SPAN,
        n_generations=N_GENERATIONS,
        n_steps_per_span=128,
        newton_iterations=4,
        rz_scales=SCALES,
    )
    loss.dependencies = {"field": field}

    value, gradient = loss.value_and_grad(loss.starting_dofs)
    expected_value, expected_gradient = jax.value_and_grad(_location_objective)(
        parameters
    )
    np.testing.assert_allclose(value, expected_value, rtol=2.0e-13)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2.0e-11)


def test_one_gradient_step_moves_the_labelled_sample_toward_target():
    parameters = jnp.array([0.25, 1.1, -0.04])
    initial_loss, gradient = jax.value_and_grad(_location_objective)(parameters)
    updated_parameters = parameters.at[0].add(-2.0 * gradient[0])
    assert float(_location_objective(updated_parameters)) < float(initial_loss)


def test_manifold_location_loss_validates_labels_and_target_shape():
    field = _ShiftedHyperbolicField(jnp.array([0.25, 1.1, -0.04]))
    with pytest.raises(ValueError, match="target_rz"):
        manifold_sample_location_loss(
            field,
            INITIAL_GUESS,
            jnp.array([1.0, 0.0]),
            jnp.array([SEED_DISTANCE]),
            jnp.ones(3),
            stability="unstable",
            side=1,
            generation_index=1,
            seed_index=0,
            phi_span=PHI_SPAN,
        )
    with pytest.raises(ValueError, match="stability"):
        trace_manifold_branch(
            field,
            INITIAL_GUESS,
            jnp.array([1.0, 0.0]),
            jnp.array([SEED_DISTANCE]),
            stability="center",
            side=1,
            phi_span=PHI_SPAN,
        )
    with pytest.raises(IndexError, match="generation_index"):
        manifold_sample_location_loss(
            field,
            INITIAL_GUESS,
            jnp.array([1.0, 0.0]),
            jnp.array([SEED_DISTANCE]),
            TARGET,
            stability="unstable",
            side=1,
            generation_index=3,
            seed_index=0,
            phi_span=PHI_SPAN,
            n_generations=1,
            n_steps_per_span=32,
            newton_iterations=4,
        )
