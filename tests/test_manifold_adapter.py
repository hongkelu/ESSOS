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
from essos.topology import (
    biot_savart_field_callable,
    essos_field_to_pyna_cylindrical_grid,
    fixed_phi_poincare_map,
    fixed_phi_poincare_map_from_coils,
)
from essos.topology_objectives import trace_manifold_reference


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


@jax.tree_util.register_pytree_node_class
class _HyperbolicField:
    """Axisymmetric field with an exact hyperbolic section point."""

    def __init__(self, rate):
        self.rate = jnp.asarray(rate)

    def B(self, xyz):
        x, y, z = xyz
        radius = jnp.sqrt(x * x + y * y)
        cos_phi = x / radius
        sin_phi = y / radius
        b_r = self.rate * (radius - 1.0)
        b_z = -self.rate * z
        b_phi = radius
        return jnp.array(
            [
                b_r * cos_phi - b_phi * sin_phi,
                b_r * sin_phi + b_phi * cos_phi,
                b_z,
            ]
        )

    def tree_flatten(self):
        return (self.rate,), None

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


def test_essos_field_samples_to_pyna_cylindrical_components_in_batches():
    radius = np.linspace(0.8, 1.4, 4)
    height = np.linspace(-0.2, 0.2, 3)
    phi = np.linspace(0.0, np.pi, 5, endpoint=False)
    pitch = 0.17
    sampled = essos_field_to_pyna_cylindrical_grid(
        _HelicalField(pitch),
        radius,
        height,
        phi,
        nfp=2,
        batch_size=17,
        axisymmetric=True,
    )

    expected_radius = np.broadcast_to(
        radius[:, None, None],
        (radius.size, height.size, phi.size),
    )
    np.testing.assert_allclose(sampled.BR, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(sampled.BZ, pitch, atol=2.0e-15)
    np.testing.assert_allclose(sampled.BPhi, expected_radius, atol=2.0e-15)
    assert sampled.nfp == 2
    assert sampled.is_axisymmetric


def _require_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind is unavailable")


@pytest.mark.parametrize("stability", ["unstable", "stable"])
def test_sampled_essos_field_closes_the_jax_cyna_manifold_loop(stability):
    _require_cyna()
    from pyna.topo.manifold_correspondence import (
        compare_jax_manifold_branch,
        manifold_branch_reference_from_trace,
    )
    from pyna.topo.toroidal import FixedPoint
    from pyna.toroidal.flt import trace_fixed_point_manifolds_field

    rate = 0.35
    map_span = 0.6
    n_steps_per_span = 60
    n_generations = 3
    field = _HyperbolicField(rate)
    production_field = essos_field_to_pyna_cylindrical_grid(
        field,
        np.linspace(0.7, 1.5, 49),
        np.linspace(-0.4, 0.4, 49),
        np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
        batch_size=2048,
        axisymmetric=True,
    )
    expansion = np.exp(rate * map_span)
    fixed_point = FixedPoint(
        phi=0.0,
        R=1.0,
        Z=0.0,
        kind="X",
        DPm=np.diag([expansion, 1.0 / expansion]),
    )
    fixed_point.metadata.update(
        {
            "orbit_id": 4,
            "map_order_index": 0,
            "monodromy_map_span": map_span,
        }
    )
    payload = trace_fixed_point_manifolds_field(
        production_field,
        [fixed_point],
        phi_section=0.0,
        map_span=map_span,
        N_turns=n_generations,
        DPhi=map_span / n_steps_per_span,
        seed_distances=np.array([1.0e-3, 2.0e-3]),
        refine_stable_inverse_anchor=False,
    )[0]
    branch = manifold_branch_reference_from_trace(
        payload,
        stability=stability,
        seed_side=1,
        n_generations=n_generations,
    )
    jax_trace = trace_manifold_reference(
        field,
        branch,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=4,
    )
    report = compare_jax_manifold_branch(
        branch,
        jax_trace.generations,
        absolute_tolerance_m=4.0e-12,
    )

    assert branch.complete
    assert report.accepted
    assert report.n_compared == (n_generations + 1) * 2


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


def test_biot_savart_field_samples_to_the_production_grid():
    field = BiotSavart(_circular_coils(1.0e5))
    radius = np.linspace(0.2, 0.6, 3)
    height = np.linspace(-0.2, 0.2, 3)
    phi = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    sampled = essos_field_to_pyna_cylindrical_grid(
        field,
        radius,
        height,
        phi,
        batch_size=10,
        axisymmetric=True,
    )

    i_r, i_z, i_phi = 1, 2, 1
    angle = phi[i_phi]
    xyz = jnp.array(
        [
            radius[i_r] * np.cos(angle),
            radius[i_r] * np.sin(angle),
            height[i_z],
        ]
    )
    bx, by, bz = np.asarray(field.B(xyz))
    expected = np.array(
        [
            bx * np.cos(angle) + by * np.sin(angle),
            bz,
            -bx * np.sin(angle) + by * np.cos(angle),
        ]
    )
    actual = np.array(
        [
            sampled.BR[i_r, i_z, i_phi],
            sampled.BZ[i_r, i_z, i_phi],
            sampled.BPhi[i_r, i_z, i_phi],
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-15)


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
