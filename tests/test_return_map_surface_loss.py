"""Tests for the first manifold-aware ESSOS Stage-2 objective."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.toroidal.flt.jax_poincare")

from essos.losses import custom_loss
from essos.topology import (
    return_map_surface_loss,
    return_map_surface_residuals,
)


@jax.tree_util.register_pytree_node_class
class _ExpandingRotatingField:
    """Analytic ESSOS field whose section flow expands and rotates circles."""

    def __init__(self, growth, rotation=0.4):
        self.growth = growth
        self.rotation = rotation

    def B(self, xyz):
        x, y, z = xyz
        radius = jnp.sqrt(x * x + y * y)
        cos_phi = x / radius
        sin_phi = y / radius
        radial_offset = radius - 1.0
        flow_r = self.growth * radial_offset - self.rotation * z
        flow_z = self.rotation * radial_offset + self.growth * z
        b_phi = radius
        return jnp.array(
            [
                flow_r * cos_phi - b_phi * sin_phi,
                flow_r * sin_phi + b_phi * cos_phi,
                flow_z,
            ]
        )

    def tree_flatten(self):
        return (self.growth,), self.rotation

    @classmethod
    def tree_unflatten(cls, rotation, children):
        return cls(children[0], rotation=rotation)


MINOR_RADIUS = 0.2
ANGLES = jnp.linspace(0.0, 2.0 * jnp.pi, 12, endpoint=False)
SEEDS = jnp.column_stack(
    (
        1.0 + MINOR_RADIUS * jnp.cos(ANGLES),
        MINOR_RADIUS * jnp.sin(ANGLES),
    )
)


def _circle_signed_distance(rz, phi):
    del phi
    return jnp.hypot(rz[0] - 1.0, rz[1]) - MINOR_RADIUS


def _objective(growth, *, weights=None):
    return return_map_surface_loss(
        _ExpandingRotatingField(growth),
        SEEDS,
        _circle_signed_distance,
        phi_start=0.13,
        phi_span=0.6,
        n_steps=128,
        weights=weights,
    )


def test_surface_residuals_vanish_for_invariant_circle():
    residuals = return_map_surface_residuals(
        _ExpandingRotatingField(0.0),
        SEEDS,
        _circle_signed_distance,
        phi_start=0.13,
        phi_span=0.6,
        n_steps=128,
    )
    np.testing.assert_allclose(residuals, 0.0, atol=2e-14)


def test_surface_loss_and_gradient_match_analytic_expansion():
    growth = 0.12
    phi_span = 0.6
    exponential = np.exp(growth * phi_span)
    expected_loss = MINOR_RADIUS**2 * (exponential - 1.0) ** 2
    expected_gradient = (
        2.0
        * MINOR_RADIUS**2
        * (exponential - 1.0)
        * exponential
        * phi_span
    )

    np.testing.assert_allclose(_objective(growth), expected_loss, rtol=2e-11)
    np.testing.assert_allclose(
        jax.grad(_objective)(growth),
        expected_gradient,
        rtol=3e-11,
    )


def test_normalized_weights_preserve_uniform_weight_loss():
    growth = 0.08
    unweighted = _objective(growth)
    weighted = _objective(growth, weights=3.0 * jnp.ones(len(SEEDS)))
    np.testing.assert_allclose(weighted, unweighted, rtol=1e-14)


def test_surface_loss_composes_with_essos_custom_loss():
    field = _ExpandingRotatingField(0.1)
    loss = custom_loss(
        return_map_surface_loss,
        "field",
        rz_seeds=SEEDS,
        target_signed_distance=_circle_signed_distance,
        phi_start=0.13,
        phi_span=0.6,
        n_steps=128,
    )
    loss.dependencies = {"field": field}

    value, gradient = loss.value_and_grad(loss.starting_dofs)
    expected_value = _objective(field.growth)
    expected_gradient = jax.grad(_objective)(field.growth)
    np.testing.assert_allclose(value, expected_value, rtol=1e-14)
    np.testing.assert_allclose(gradient, [expected_gradient], rtol=1e-13)


def test_surface_loss_validates_seed_and_weight_shapes():
    field = _ExpandingRotatingField(0.1)
    with pytest.raises(ValueError, match="rz_seeds"):
        return_map_surface_loss(
            field,
            jnp.ones(2),
            _circle_signed_distance,
            phi_span=0.2,
        )
    with pytest.raises(ValueError, match="weights"):
        return_map_surface_loss(
            field,
            SEEDS,
            _circle_signed_distance,
            phi_span=0.2,
            weights=jnp.ones(2),
        )
