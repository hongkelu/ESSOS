"""Tests for the one-way ESSOS-to-PyNA topology adapter."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.toroidal.flt.jax_poincare")

from essos.coils import Coils, Curves
from essos.fields import BiotSavart
from essos.manifold import (
    biot_savart_field_callable,
    fixed_phi_poincare_map,
    fixed_phi_poincare_map_from_coils,
)


@jax.tree_util.register_pytree_node_class
class _HelicalField:
    """Minimal ESSOS-style field with B_phi=R and constant B_z."""

    def __init__(self, pitch):
        self.pitch = pitch

    def B(self, xyz):
        x, y, _ = xyz
        return jnp.array([-y, x, self.pitch])

    def tree_flatten(self):
        return (self.pitch,), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        return cls(children[0])


def _circular_coils(current, radius=1.0):
    dofs = jnp.zeros((1, 3, 3))
    dofs = dofs.at[0, 0, 2].set(radius)  # x = radius * cos(theta)
    dofs = dofs.at[0, 1, 1].set(radius)  # y = radius * sin(theta)
    curves = Curves(dofs, n_segments=64, nfp=1, stellsym=False)
    return Coils(curves, jnp.atleast_1d(current), currents_scale=1.0e5)


def test_essos_field_object_drives_jitted_pyna_map_and_gradient():
    rz0 = jnp.array([1.3, -0.1])
    phi_span = 0.7

    def final_z(pitch):
        field = _HelicalField(pitch)
        return fixed_phi_poincare_map(
            field,
            rz0,
            phi_span=phi_span,
            n_steps=32,
        )[1]

    value = jax.jit(final_z)(0.2)
    np.testing.assert_allclose(value, 0.04, atol=1e-13)
    np.testing.assert_allclose(jax.grad(final_z)(0.2), phi_span, atol=1e-13)


def test_biot_savart_callable_reuses_essos_field_implementation():
    coils = _circular_coils(1.0e5)
    xyz = jnp.array([0.0, 0.0, 0.2])
    expected = BiotSavart(coils).B(xyz)
    actual = biot_savart_field_callable(xyz, coils)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_biot_savart_callable_is_differentiable_in_current():
    xyz = jnp.array([0.0, 0.0, 0.2])

    def axial_field(current):
        return biot_savart_field_callable(xyz, _circular_coils(current))[2]

    derivative = jax.grad(axial_field)(1.0e5)
    assert np.isfinite(derivative)
    assert abs(float(derivative)) > 0.0


def test_biot_savart_callable_is_differentiable_in_curve_dof():
    xyz = jnp.array([0.0, 0.0, 0.2])

    def axial_field(radius):
        return biot_savart_field_callable(
            xyz,
            _circular_coils(1.0e5, radius=radius),
        )[2]

    derivative = jax.grad(axial_field)(1.0)
    assert np.isfinite(derivative)
    assert abs(float(derivative)) > 0.0


def test_return_map_differentiates_through_essos_coil_current():
    major_radius = 2.0
    tf_radius = 0.8
    n_tf = 8
    dofs = jnp.zeros((n_tf + 1, 3, 3))
    for index in range(n_tf):
        phi = 2.0 * jnp.pi * index / n_tf
        radial = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
        dofs = dofs.at[index, :, 0].set(major_radius * radial)
        dofs = dofs.at[index, :, 2].set(tf_radius * radial)
        dofs = dofs.at[index, 2, 1].set(tf_radius)

    # Add one horizontal PF coil to produce a nontrivial poloidal map.
    dofs = dofs.at[-1, 0, 2].set(3.0)
    dofs = dofs.at[-1, 1, 1].set(3.0)
    dofs = dofs.at[-1, 2, 0].set(0.6)
    curves = Curves(dofs, n_segments=32, nfp=1, stellsym=False)
    tf_currents = -8.0e5 * jnp.ones(n_tf)
    rz0 = jnp.array([major_radius, 0.05])

    def endpoint(pf_current):
        currents = jnp.concatenate((tf_currents, jnp.atleast_1d(pf_current)))
        coils = Coils(curves, currents, currents_scale=8.0e5)
        return fixed_phi_poincare_map_from_coils(
            coils,
            rz0,
            phi_span=2.0 * jnp.pi / n_tf,
            n_steps=32,
            bphi_floor=1.0e-10,
        )

    pf_current = 1.0e5
    derivative = jax.jacfwd(endpoint)(pf_current)
    delta_current = 10.0
    finite_difference = (
        endpoint(pf_current + delta_current)
        - endpoint(pf_current - delta_current)
    ) / (2.0 * delta_current)

    assert bool(jnp.isfinite(endpoint(pf_current)).all())
    assert float(jnp.linalg.norm(derivative)) > 0.0
    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=2e-7,
        atol=1e-12,
    )
