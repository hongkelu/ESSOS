"""Tests for PyNA-owned periodic X-lines in ESSOS Stage-2 losses."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.topo.jax_periodic")

from essos.losses import custom_loss
from essos.topology import (
    periodic_xline_clearance_loss,
    periodic_xline_clearance_residuals,
    periodic_xline_hyperbolicity_loss,
    periodic_xline_location_loss,
    periodic_xline_state,
    periodic_xline_trajectory,
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
        flow_r = rate * (radius - center_r)
        flow_z = -rate * (z - center_z)
        b_phi = radius
        return jnp.array(
            [
                flow_r * cos_phi - b_phi * sin_phi,
                flow_r * sin_phi + b_phi * cos_phi,
                flow_z,
            ]
        )

    def tree_flatten(self):
        return (self.parameters,), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        return cls(children[0])


INITIAL_GUESS = jnp.array([0.94, -0.06])
PHI_START = 0.19
PHI_SPAN = 0.7


def _state(parameters):
    return periodic_xline_state(
        _ShiftedHyperbolicField(parameters),
        INITIAL_GUESS,
        phi_start=PHI_START,
        phi_span=PHI_SPAN,
        n_steps_per_span=192,
        newton_iterations=4,
    )


def test_periodic_xline_state_is_owned_by_pyna_and_jittable():
    parameters = jnp.array([0.28, 1.03, 0.025])
    state = jax.jit(_state)(parameters)
    expected_monodromy = np.diag(
        [
            np.exp(parameters[0] * PHI_SPAN),
            np.exp(-parameters[0] * PHI_SPAN),
        ]
    )

    assert bool(state.converged)
    np.testing.assert_allclose(state.position, parameters[1:], atol=3e-14)
    np.testing.assert_allclose(
        state.monodromy,
        expected_monodromy,
        rtol=2e-12,
        atol=2e-12,
    )


def test_xline_location_loss_and_reverse_gradient_are_analytic():
    parameters = jnp.array([0.28, 1.03, 0.025])
    target = jnp.array([1.0, 0.0])
    scales = jnp.array([0.1, 0.05])

    def objective(dynamic_parameters):
        return periodic_xline_location_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            INITIAL_GUESS,
            target,
            rz_scales=scales,
            phi_start=PHI_START,
            phi_span=PHI_SPAN,
            n_steps_per_span=128,
            newton_iterations=4,
        )

    displacement = np.asarray(parameters[1:] - target)
    expected_value = np.mean(np.square(displacement / np.asarray(scales)))
    expected_gradient = np.array(
        [
            0.0,
            displacement[0] / float(scales[0] ** 2),
            displacement[1] / float(scales[1] ** 2),
        ]
    )
    value, gradient = jax.value_and_grad(objective)(parameters)
    np.testing.assert_allclose(value, expected_value, rtol=2e-13)
    np.testing.assert_allclose(gradient, expected_gradient, atol=8e-13)


def test_xline_hyperbolicity_window_and_gradient_are_analytic():
    parameters = jnp.array([0.25, 1.03, 0.025])
    minimum_margin = 0.08

    def objective(dynamic_parameters):
        return periodic_xline_hyperbolicity_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            INITIAL_GUESS,
            phi_start=PHI_START,
            phi_span=PHI_SPAN,
            n_steps_per_span=192,
            newton_iterations=4,
            minimum_margin=minimum_margin,
            branch_sign=1.0,
        )

    rate = float(parameters[0])
    margin = 2.0 * np.cosh(rate * PHI_SPAN) - 2.0
    violation = minimum_margin - margin
    expected_value = violation**2
    expected_rate_gradient = (
        -2.0 * violation * 2.0 * PHI_SPAN * np.sinh(rate * PHI_SPAN)
    )
    value, gradient = jax.value_and_grad(objective)(parameters)
    np.testing.assert_allclose(value, expected_value, rtol=2e-10)
    np.testing.assert_allclose(
        gradient,
        [expected_rate_gradient, 0.0, 0.0],
        rtol=3e-9,
        atol=2e-12,
    )

    no_penalty = periodic_xline_hyperbolicity_loss(
        _ShiftedHyperbolicField(parameters),
        INITIAL_GUESS,
        phi_start=PHI_START,
        phi_span=PHI_SPAN,
        n_steps_per_span=128,
        newton_iterations=4,
        minimum_margin=0.01,
        maximum_margin=0.2,
    )
    np.testing.assert_allclose(no_penalty, 0.0, atol=1e-15)


def _circular_wall_clearance(point_xyz):
    radius = jnp.sqrt(point_xyz[0] ** 2 + point_xyz[1] ** 2)
    distance_from_centerline = jnp.sqrt(
        (radius - 1.0) ** 2 + point_xyz[2] ** 2
    )
    return 0.15 - distance_from_centerline


def test_xline_trajectory_and_clearance_gradient_are_analytic():
    parameters = jnp.array([0.28, 1.03, 0.025])
    minimum_clearance = 0.12
    clearance_scale = 0.01

    trajectory = periodic_xline_trajectory(
        _ShiftedHyperbolicField(parameters),
        INITIAL_GUESS,
        phi_start=PHI_START,
        phi_span=PHI_SPAN,
        map_power=2,
        n_steps_per_span=48,
        newton_iterations=4,
    )
    assert trajectory.point_xyz.shape == (97, 3)
    np.testing.assert_allclose(trajectory.closure_error, 0.0, atol=3e-14)

    def objective(dynamic_parameters):
        return periodic_xline_clearance_loss(
            _ShiftedHyperbolicField(dynamic_parameters),
            INITIAL_GUESS,
            _circular_wall_clearance,
            minimum_clearance_m=minimum_clearance,
            clearance_scale_m=clearance_scale,
            phi_start=PHI_START,
            phi_span=PHI_SPAN,
            n_steps_per_span=96,
            newton_iterations=4,
            sample_stride=8,
        )

    radial_offset = float(parameters[1] - 1.0)
    vertical_offset = float(parameters[2])
    centerline_distance = np.hypot(radial_offset, vertical_offset)
    clearance = 0.15 - centerline_distance
    violation = minimum_clearance - clearance
    expected_value = 0.5 * (violation / clearance_scale) ** 2
    expected_gradient = np.asarray(
        [
            0.0,
            violation * radial_offset
            / (clearance_scale**2 * centerline_distance),
            violation * vertical_offset
            / (clearance_scale**2 * centerline_distance),
        ]
    )
    value, gradient = jax.value_and_grad(objective)(parameters)
    np.testing.assert_allclose(value, expected_value, rtol=2e-12)
    np.testing.assert_allclose(
        gradient,
        expected_gradient,
        rtol=2e-11,
        atol=1e-12,
    )

    residuals = periodic_xline_clearance_residuals(
        _ShiftedHyperbolicField(parameters),
        INITIAL_GUESS,
        _circular_wall_clearance,
        minimum_clearance_m=minimum_clearance,
        clearance_scale_m=clearance_scale,
        phi_start=PHI_START,
        phi_span=PHI_SPAN,
        n_steps_per_span=96,
        newton_iterations=4,
        sample_stride=8,
    )
    assert residuals.shape == (12,)
    np.testing.assert_allclose(residuals, violation / clearance_scale)


def test_xline_location_loss_composes_with_essos_custom_loss():
    field = _ShiftedHyperbolicField(jnp.array([0.28, 1.03, 0.025]))
    loss = custom_loss(
        periodic_xline_location_loss,
        "field",
        initial_guess=INITIAL_GUESS,
        target_rz=jnp.array([1.0, 0.0]),
        rz_scales=jnp.array([0.1, 0.05]),
        phi_start=PHI_START,
        phi_span=PHI_SPAN,
        n_steps_per_span=128,
        newton_iterations=4,
    )
    loss.dependencies = {"field": field}

    value, gradient = loss.value_and_grad(loss.starting_dofs)
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, [0.0, 3.0, 10.0], atol=8e-13)


def test_xline_losses_validate_static_configuration():
    field = _ShiftedHyperbolicField(jnp.array([0.25, 1.0, 0.0]))
    with pytest.raises(ValueError, match="target_rz"):
        periodic_xline_location_loss(
            field,
            INITIAL_GUESS,
            jnp.ones(3),
            phi_span=PHI_SPAN,
        )
    with pytest.raises(ValueError, match="rz_scales"):
        periodic_xline_location_loss(
            field,
            INITIAL_GUESS,
            jnp.ones(2),
            rz_scales=jnp.ones(3),
            phi_span=PHI_SPAN,
        )
    with pytest.raises(ValueError, match="branch_sign"):
        periodic_xline_hyperbolicity_loss(
            field,
            INITIAL_GUESS,
            phi_span=PHI_SPAN,
            branch_sign=0.0,
        )
    with pytest.raises(ValueError, match="maximum_margin"):
        periodic_xline_hyperbolicity_loss(
            field,
            INITIAL_GUESS,
            phi_span=PHI_SPAN,
            minimum_margin=0.2,
            maximum_margin=0.1,
        )
    with pytest.raises(TypeError, match="wall_signed_distance"):
        periodic_xline_clearance_loss(
            field,
            INITIAL_GUESS,
            None,
            minimum_clearance_m=0.01,
            clearance_scale_m=0.01,
            phi_span=PHI_SPAN,
        )
    with pytest.raises(ValueError, match="minimum_clearance_m"):
        periodic_xline_clearance_loss(
            field,
            INITIAL_GUESS,
            _circular_wall_clearance,
            minimum_clearance_m=-0.01,
            clearance_scale_m=0.01,
            phi_span=PHI_SPAN,
        )
    with pytest.raises(ValueError, match="clearance_scale_m"):
        periodic_xline_clearance_loss(
            field,
            INITIAL_GUESS,
            _circular_wall_clearance,
            minimum_clearance_m=0.01,
            clearance_scale_m=0.0,
            phi_span=PHI_SPAN,
        )
    with pytest.raises(ValueError, match="sample_stride"):
        periodic_xline_clearance_loss(
            field,
            INITIAL_GUESS,
            _circular_wall_clearance,
            minimum_clearance_m=0.01,
            clearance_scale_m=0.01,
            phi_span=PHI_SPAN,
            sample_stride=0,
        )
