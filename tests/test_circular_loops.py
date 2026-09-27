"""Closed-form circular-loop fields against Biot-Savart of finely discretised loops."""
import jax
import jax.numpy as jnp
import numpy as np

from essos.coils import Coils, Curves
from essos.fields import BiotSavart, CircularLoopsField

jax.config.update("jax_enable_x64", True)

RADII, HEIGHTS, CURRENTS = np.array([1.0, 2.5, 0.6]), np.array([0.0, -1.2, 0.8]), np.array([1.0e6, -3.0e5, 2.0e5])


def _discretised():
    dofs = np.zeros((3, 3, 3))
    dofs[:, 0, 2], dofs[:, 1, 1], dofs[:, 2, 0] = RADII, RADII, HEIGHTS
    return BiotSavart(Coils(Curves(jnp.asarray(dofs), n_segments=2048, nfp=1, stellsym=False),
                            jnp.asarray(CURRENTS), currents_scale=1.0e6))


def test_circular_loops_match_biot_savart_and_are_divergence_and_curl_free():
    exact, reference = CircularLoopsField(RADII, HEIGHTS, CURRENTS), _discretised()
    points = jnp.array([[1.6, 0.2, 0.3], [0.3, 1.7, -0.5], [-2.0, -0.4, 1.1], [3.1, 0.0, -1.0]])
    for point in points:
        assert jnp.allclose(exact.B(point), reference.B(point), rtol=1e-9, atol=1e-12)
        gradient = jax.jacfwd(exact.B)(point)
        assert abs(jnp.trace(gradient)) < 1e-9 * jnp.max(jnp.abs(gradient))
        assert jnp.allclose(gradient, gradient.T, atol=1e-9 * jnp.max(jnp.abs(gradient)))


def test_circular_loop_flux_generates_the_poloidal_field():
    field = CircularLoopsField(RADII, HEIGHTS, CURRENTS)
    for R, Z in [(1.6, 0.3), (2.0, -0.4), (3.1, -1.0)]:
        d_psi_dR, d_psi_dZ = jax.grad(field.psi, argnums=(0, 1))(R, Z)
        B_R, B_Z = field.B_cylindrical(R, Z)
        assert jnp.allclose(jnp.array([B_R, B_Z]), jnp.array([-d_psi_dZ / R, d_psi_dR / R]), rtol=1e-9)
