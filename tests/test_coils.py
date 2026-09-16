import json
import pytest
import jax
from essos.coils import Coils, Curves
from essos.surfaces import surfacerzfourier_from_boundary
import jax.numpy as jnp
import random

import numpy as np

from essos.coils import Coils, Coils_from_json, Curves


def test_curves_initialization():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    assert curves.dofs.shape == (2, 3, 5)
    assert curves.n_segments == 100
    assert curves.nfp == 1
    assert curves.stellsym
    assert curves.order == 2
    assert curves.curves.shape == (4, 3, 5)
    assert curves.gamma.shape == (4, 100, 3)
    assert curves.gamma_dash.shape == (4, 100, 3)

def test_curve_and_coil_dof_names_follow_flattened_dofs():
    curves = Curves(jnp.zeros((2, 3, 5)), stellsym=False)
    assert curves.dof_names[:7] == (
        "coil[0].x0", "coil[0].xs(1)", "coil[0].xc(1)",
        "coil[0].xs(2)", "coil[0].xc(2)", "coil[0].y0", "coil[0].ys(1)")
    assert len(curves.dof_names) == curves.dofs.size
    coils = Coils(curves, jnp.array([1.0, 2.0]))
    assert coils.dof_names[-2:] == ("coil[0].current", "coil[1].current")
    assert len(coils.dof_names) == coils.dofs.size
    updated = coils.with_dofs(coils.dofs + 1.0)
    assert jnp.allclose(updated.dofs, coils.dofs + 1.0)
    assert not jnp.allclose(coils.dofs, updated.dofs)
    gradient = jax.grad(lambda dofs: jnp.sum(coils.with_dofs(dofs).gamma))(coils.dofs)
    assert gradient.shape == coils.dofs.shape and jnp.all(jnp.isfinite(gradient))
    updated_curves = curves.with_dofs(curves.dofs + 2.0)
    assert jnp.allclose(updated_curves.dofs, curves.dofs + 2.0)
    assert not jnp.allclose(curves.dofs, updated_curves.dofs)

def test_surface_from_vmec_boundary_preserves_modes_and_is_differentiable():
    rbc = jnp.arange(15.0).reshape(5, 3); zbs = -rbc
    surface = surfacerzfourier_from_boundary(
        rbc, zbs, nfp=2, nphi=8, ntheta=10)
    expected_r = jnp.concatenate((rbc[2:, 0], rbc[:, 1:].T.ravel()))
    expected_z = jnp.concatenate((zbs[2:, 0], zbs[:, 1:].T.ravel()))
    assert surface.mpol == 2 and surface.ntor == 2
    assert jnp.array_equal(surface.rc, expected_r)
    assert jnp.array_equal(surface.zs, expected_z)
    gradient = jax.grad(lambda values: jnp.sum(
        surfacerzfourier_from_boundary(values, zbs, 2, nphi=8, ntheta=10).gamma))(rbc)
    assert jnp.all(jnp.isfinite(gradient))

    with pytest.raises(ValueError, match="equal shape"):
        surfacerzfourier_from_boundary(jnp.zeros((4, 3)), jnp.zeros((4, 3)), 2)

def _small_surface():
    rbc = jnp.zeros((5, 3)); zbs = jnp.zeros((5, 3))
    rbc = rbc.at[2, 0].set(1.0).at[2, 1].set(0.2)
    zbs = zbs.at[2, 1].set(0.2)
    return surfacerzfourier_from_boundary(rbc, zbs, 2, nphi=8, ntheta=10)

@pytest.mark.parametrize("name, caches", (
    ("theta2d", ("_theta2d", "_phi2d")),
    ("phi2d", ("_theta2d", "_phi2d")),
    ("angles", ("_angles",)),
    ("gamma", ("_gamma", "_gammadash_theta", "_gammadash_phi")),
    ("gammadash_theta", ("_gamma", "_gammadash_theta", "_gammadash_phi")),
    ("gammadash_phi", ("_gamma", "_gammadash_theta", "_gammadash_phi")),
    ("normal", ("_normal", "_unitnormal", "_area_element")),
    ("unitnormal", ("_normal", "_unitnormal", "_area_element")),
    ("area_element", ("_normal", "_unitnormal", "_area_element")),
))
def test_surface_cache_is_concrete_and_does_not_retain_tracers(name, caches):
    surface = _small_surface()
    value = jax.jit(lambda scale: scale * jnp.sum(getattr(surface, name)))(1.0)
    assert jnp.isfinite(value)
    # Access after the transform must recompute concrete values, not retrieve a
    # DynamicJaxprTracer that escaped from the compiled objective.
    assert jnp.all(jnp.isfinite(getattr(surface, name)))
    cached = [getattr(surface, cache) for cache in caches]
    assert all(item is not None and not isinstance(item, jax.core.Tracer) for item in cached)

def test_curves_initialization_with_params():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs, n_segments=50, nfp=2, stellsym=False)
    assert curves.dofs.shape == (2, 3, 5)
    assert curves.n_segments == 50
    assert curves.nfp == 2
    assert not curves.stellsym
    assert curves.order == 2
    assert curves.curves.shape == (4, 3, 5)
    assert curves.gamma.shape == (4, 50, 3)
    assert curves.gamma_dash.shape == (4, 50, 3)


def test_curves_computed_attributes():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    assert curves.gamma.shape == (4, 100, 3)
    assert curves.gamma_dash.shape == (4, 100, 3)
    assert curves.length.shape == (4,)


def test_curves_property_setters():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    new_dofs = jnp.ones((2, 3, 5))
    curves.dofs = new_dofs
    assert jnp.allclose(curves.dofs, new_dofs)
    curves.n_segments = 50
    assert curves.n_segments == 50
    curves.nfp = 2
    assert curves.nfp == 2
    curves.stellsym = False
    assert not curves.stellsym


def test_curves_pytree_preserves_scaling_metadata():
    dofs = jnp.ones((2, 3, 5))
    curves = Curves(dofs, scaling_type=2, scaling_factor=0.3, scale_fixed=7.0)
    curves_copy = jax.tree_util.tree_map(lambda x: x, curves)

    assert curves_copy.scaling_type == curves.scaling_type
    assert curves_copy.scaling_factor == curves.scaling_factor
    assert curves_copy.scale_fixed == curves.scale_fixed

def test_curves_str_repr():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    assert isinstance(str(curves), str)
    assert isinstance(repr(curves), str)


def test_curves_save_curves(tmp_path):
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    filename = tmp_path / "curves.txt"
    curves.save_curves(filename)
    with open(filename, "r") as file:
        content = file.read()
    assert "nfp stellsym order" in content


def test_curves_plot():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    curves.plot(show=False)


def test_curves_len():
    dofs = jnp.zeros((2, 3, 5))
    nfp = random.randint(1, 10)
    curves = Curves(dofs, nfp=nfp)
    assert len(curves) == 2 * 2 * nfp


def test_curves_getitem():
    dofs = jnp.ones((2, 3, 5))
    nfp = random.randint(4, 10)
    curves = Curves(dofs, nfp=nfp)
    assert curves[0].curves.shape == (1, 3, 5)
    assert curves[1].curves.shape == (1, 3, 5)
    assert curves[2].curves.shape == (1, 3, 5)
    assert curves[3].curves.shape == (1, 3, 5)
    assert curves[1:3].curves.shape == (2, 3, 5)
    assert curves[jnp.array([0, 3])].curves.shape == (2, 3, 5)


def test_curves_add():
    dofs = jnp.zeros((2, 3, 5))
    curves1 = Curves(dofs, stellsym=False)
    curves2 = Curves(dofs, stellsym=False)
    curves3 = curves1 + curves2
    assert curves3.dofs.shape == (4, 3, 5)


def test_curves_contains():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs, stellsym=False)
    assert curves[0] in curves
    assert curves[1] in curves


def test_curves_eq():
    dofs = jnp.zeros((2, 3, 5))
    curves1 = Curves(dofs, stellsym=False)
    curves2 = Curves(dofs, stellsym=False)
    assert curves1 == curves2


def test_curves_ne():
    dofs = jnp.zeros((2, 3, 5))
    curves1 = Curves(dofs, stellsym=False)
    curves2 = Curves(dofs, stellsym=True)
    assert curves1 != curves2


def test_curves_iter():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs, stellsym=False)
    for curve in curves:
        assert curve.curves.shape == (1, 3, 5)


def test_coils_pytree_roundtrip_preserves_physical_current_scale():
    curves = Curves(jnp.ones((2, 3, 5)), n_segments=12, stellsym=False)
    coils = Coils(curves, jnp.asarray([1.0e5, 2.0e5]))
    rebuilt = jax.jit(lambda item: item)(coils)

    np.testing.assert_allclose(
        rebuilt.base_currents, coils.base_currents, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(rebuilt.currents, coils.currents, rtol=0.0, atol=0.0)


def test_coils_pytree_unflatten_accepts_an_all_zero_current_tangent():
    curves = Curves(jnp.ones((2, 3, 5)), n_segments=12, stellsym=False)
    coils = Coils(curves, jnp.asarray([1.0e5, 2.0e5]))
    leaves, definition = jax.tree_util.tree_flatten(coils)
    rebuilt = jax.tree_util.tree_unflatten(
        definition,
        [leaves[0], jnp.zeros_like(leaves[1]), leaves[2]],
    )

    np.testing.assert_array_equal(rebuilt.base_currents, np.zeros(2))
    np.testing.assert_array_equal(rebuilt.currents, np.zeros(2))
    assert np.isfinite(float(rebuilt.currents_scale))


def test_coils_json_roundtrip_preserves_physical_current_scale(tmp_path):
    curves = Curves(jnp.ones((2, 3, 5)), n_segments=12, stellsym=False)
    coils = Coils(curves, jnp.asarray([1.0e5, 2.0e5]))
    path = tmp_path / "coils.json"
    coils.to_json(path)
    rebuilt = Coils_from_json(path)

    np.testing.assert_allclose(
        rebuilt.base_currents, coils.base_currents, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(rebuilt.currents, coils.currents, rtol=0.0, atol=0.0)


def test_coils_json_supports_scale_aware_and_legacy_current_formats(tmp_path):
    common = {
        "nfp": 1,
        "stellsym": False,
        "order": 2,
        "n_segments": 12,
        "dofs_curves": np.ones((2, 3, 5)).tolist(),
    }
    scale_aware = tmp_path / "scale-aware.json"
    scale_aware.write_text(
        json.dumps(
            {
                **common,
                "dofs_currents": [2.0 / 3.0, 4.0 / 3.0],
                "currents_scale": 1.5e5,
            }
        )
    )
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({**common, "dofs_currents": [1.0e5, 2.0e5]}))

    for path in (scale_aware, legacy):
        coils = Coils_from_json(path)
        np.testing.assert_allclose(coils.base_currents, [1.0e5, 2.0e5])


def test_coils_json_rejects_inconsistent_redundant_currents(tmp_path):
    path = tmp_path / "inconsistent.json"
    path.write_text(
        json.dumps(
            {
                "nfp": 1,
                "stellsym": False,
                "order": 2,
                "n_segments": 12,
                "dofs_curves": np.ones((2, 3, 5)).tolist(),
                "dofs_currents": [2.0 / 3.0, 4.0 / 3.0],
                "currents_scale": 1.5e5,
                "base_currents": [1.0e5, 3.0e5],
            }
        )
    )

    with pytest.raises(ValueError, match="inconsistent current representations"):
        Coils_from_json(path)


if __name__ == "__main__":
    pytest.main()


def test_zero_physical_currents_have_finite_scale_and_field():
    from essos.coils import CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    coils = Coils(CreateEquallySpacedCurves(1, 2, 1.4, 0.2, n_segments=16), jnp.zeros(1))
    np.testing.assert_array_equal(coils.dofs_currents, [0.0])
    field = BiotSavart(coils)
    np.testing.assert_array_equal(jax.jit(field.B_covariant)(jnp.asarray([0.8, 0.2, 0.3])), np.zeros(3))
