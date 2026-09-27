"""Connection lengths and their derivatives against closed-form field-line solutions.

Two exactly solvable geometries are used:

* ``CircularTokamakField`` with a horizontal divertor plate ``Z = -h``. Field lines
  stay on circles of minor radius ``r`` and advance ``dl/dtheta = sqrt(F^2 +
  (B_poloidal r)^2) / B_poloidal`` per radian of poloidal angle, so the lengths,
  strike points and all parameter derivatives are known exactly. Lines with
  ``r < h`` never reach the plate.
* A straight hyperbolic X-point, ``B = (k y, k x, B_z)``, with seeds on its stable
  manifold. The distance to a plate across the unstable direction diverges like
  ``(B_z / k) log(1 / u0)`` as the seed approaches the X-line.
"""
import diffrax
import jax
import jax.numpy as jnp
import pytest

from essos.dynamics import connection_length
from essos.fields import CircularTokamakField

jax.config.update("jax_enable_x64", True)

MAJOR_RADIUS, B_TOROIDAL, B_POLOIDAL, PLATE_DEPTH = 3.0, 2.0, 0.8, 0.4
TOLERANCE = 1e-10


def _plate(depth):
    return lambda xyz: xyz[2] + depth  # positive above the plate Z = -depth


def _toroidal_angle_advance(field, r, theta):
    """Integral of dphi/dtheta = F / (B_poloidal (R0 + r cos theta)) from 0 to theta."""
    k = jnp.sqrt(field.major_radius**2 - r**2)
    a = jnp.sqrt((field.major_radius - r) / (field.major_radius + r))
    branch = jnp.floor((theta + jnp.pi) / (2 * jnp.pi))
    primitive = 2.0 / k * (jnp.arctan(a * jnp.tan(theta / 2)) + jnp.pi * branch)
    return field.B_toroidal * field.major_radius / field.B_poloidal * primitive


def circular_tokamak_plate_solution(field, depth, seeds):
    """Forward/backward lengths ``(n, 2)`` and strike points ``(n, 2, 3)`` to ``Z = -depth``."""
    R = jnp.hypot(seeds[:, 0], seeds[:, 1])
    phi0 = jnp.arctan2(seeds[:, 1], seeds[:, 0])
    r = jnp.hypot(R - field.major_radius, seeds[:, 2])
    theta0 = jnp.arctan2(seeds[:, 2], R - field.major_radius)
    F = field.B_toroidal * field.major_radius
    length_per_radian = jnp.sqrt(F**2 + (field.B_poloidal * r) ** 2) / field.B_poloidal
    below = jnp.arcsin(depth / r)
    theta_hit = jnp.stack([jnp.pi + below, -below], axis=1)
    lengths = length_per_radian[:, None] * jnp.abs(theta_hit - theta0[:, None])
    phi_hit = phi0[:, None] + (_toroidal_angle_advance(field, r[:, None], theta_hit)
                               - _toroidal_angle_advance(field, r, theta0)[:, None])
    R_hit = field.major_radius + r[:, None] * jnp.cos(theta_hit)
    points = jnp.stack([R_hit * jnp.cos(phi_hit), R_hit * jnp.sin(phi_hit),
                        jnp.full_like(R_hit, -depth)], axis=-1)
    return lengths, points


def _seed(r, theta, phi):
    R = MAJOR_RADIUS + r * jnp.cos(theta)
    return jnp.array([R * jnp.cos(phi), R * jnp.sin(phi), r * jnp.sin(theta)])


def test_circular_tokamak_field_keeps_lines_on_circles_and_is_divergence_free():
    field = CircularTokamakField(MAJOR_RADIUS, B_TOROIDAL, B_POLOIDAL)
    points = jnp.stack([_seed(0.3, 0.4, 0.1), _seed(0.7, -2.0, 2.5), _seed(0.9, 3.0, -1.0)])
    minor_radius = lambda xyz: jnp.hypot(jnp.hypot(xyz[0], xyz[1]) - MAJOR_RADIUS, xyz[2])  # noqa: E731
    for point in points:
        B = field.B(point)
        assert jnp.abs(jnp.dot(B, jax.grad(minor_radius)(point))) < 1e-14
        assert jnp.abs(jnp.trace(field.dB_by_dX(point))) < 1e-14
        R = jnp.hypot(point[0], point[1])
        assert jnp.allclose(jnp.dot(B, jnp.array([-point[1], point[0], 0.0]) / R),
                            B_TOROIDAL * MAJOR_RADIUS / R)


def test_connection_length_matches_circular_tokamak_plate_closed_form():
    field = CircularTokamakField(MAJOR_RADIUS, B_TOROIDAL, B_POLOIDAL)
    seeds = jnp.stack([_seed(0.5, 0.0, 0.0), _seed(0.7, 1.2, 0.8),
                       _seed(0.9, 2.9, -2.0), _seed(0.6, -0.5, 4.0)])
    max_length = 200.0
    result = connection_length(field, seeds, _plate(PLATE_DEPTH), max_length=max_length,
                               tolerance=TOLERANCE)
    lengths, points = circular_tokamak_plate_solution(field, PLATE_DEPTH, seeds)
    assert jnp.all(result["hit"])
    assert jnp.allclose(result["lengths"], lengths, rtol=1e-8, atol=0.0)
    assert jnp.allclose(result["strike_points"], points, rtol=0.0, atol=1e-7)
    assert jnp.all(lengths < max_length)


def test_connection_length_traces_only_the_requested_direction():
    field = CircularTokamakField(MAJOR_RADIUS, B_TOROIDAL, B_POLOIDAL)
    seeds = jnp.stack([_seed(0.5, 0.0, 0.0), _seed(0.7, 1.2, 0.8)])
    both = connection_length(field, seeds, _plate(PLATE_DEPTH), max_length=200.0, tolerance=TOLERANCE)
    for column, sign in enumerate((1.0, -1.0)):
        one = connection_length(field, seeds, _plate(PLATE_DEPTH), max_length=200.0,
                                tolerance=TOLERANCE, directions=(sign,))
        assert one["lengths"].shape == (2, 1)
        assert jnp.allclose(one["lengths"][:, 0], both["lengths"][:, column], rtol=1e-12)
        assert jnp.allclose(one["strike_points"][:, 0], both["strike_points"][:, column], atol=1e-12)
    with pytest.raises(ValueError, match="directions"):
        connection_length(field, seeds, _plate(PLATE_DEPTH), max_length=200.0, directions=(0.5,))


def test_connection_length_compiles_once_inside_an_outer_jit():
    seeds = jnp.stack([_seed(0.5, 0.0, 0.0), _seed(0.7, 1.2, 0.8)])

    @jax.jit
    def lengths(B_poloidal, depth):
        field = CircularTokamakField(MAJOR_RADIUS, B_TOROIDAL, B_poloidal)
        return connection_length(field, seeds, _plate(depth), max_length=200.0,
                                 tolerance=TOLERANCE, directions=(1.0,))["lengths"][:, 0]

    for B_poloidal, depth in ((B_POLOIDAL, PLATE_DEPTH), (0.9, 0.35)):
        field = CircularTokamakField(MAJOR_RADIUS, B_TOROIDAL, B_poloidal)
        expected = circular_tokamak_plate_solution(field, depth, seeds)[0][:, 0]
        assert jnp.allclose(lengths(B_poloidal, depth), expected, rtol=1e-8)
    assert lengths._cache_size() == 1


def test_connection_length_caps_circular_tokamak_lines_that_miss_the_plate():
    field = CircularTokamakField(MAJOR_RADIUS, B_TOROIDAL, B_POLOIDAL)
    seeds = jnp.stack([_seed(0.3, 0.0, 0.0), _seed(0.39, 2.0, 1.0)])
    result = connection_length(field, seeds, _plate(PLATE_DEPTH), max_length=60.0,
                               tolerance=TOLERANCE)
    assert not jnp.any(result["hit"])
    assert jnp.allclose(result["lengths"], 60.0)
    r = jnp.hypot(jnp.hypot(result["strike_points"][..., 0], result["strike_points"][..., 1])
                  - MAJOR_RADIUS, result["strike_points"][..., 2])
    assert jnp.allclose(r, jnp.array([[0.3, 0.3], [0.39, 0.39]]), atol=1e-8)


def test_circular_tokamak_connection_length_approaches_pi_q_R_at_large_aspect_ratio():
    # q = F / (B_poloidal sqrt(R0^2 - r^2)). From the top of the circle a plate grazing
    # its bottom (h -> 0) is a quarter poloidal turn away in each direction, so the
    # line spans half a poloidal turn and Lc -> pi q R0 as r / R0 -> 0.
    field = CircularTokamakField(100.0, B_TOROIDAL, B_POLOIDAL)
    r = 0.5
    q = B_TOROIDAL * 100.0 / (B_POLOIDAL * jnp.sqrt(100.0**2 - r**2))
    seed = jnp.array([[100.0, 0.0, r]])
    Lc = connection_length(field, seed, _plate(1e-6), max_length=5000.0,
                           tolerance=TOLERANCE)["connection_length"]
    assert jnp.allclose(Lc, jnp.pi * q * 100.0, rtol=1e-4)


# Reverse mode uses Diffrax's default checkpointed adjoint; forward mode needs ForwardMode.
DERIVATIVE_MODES = pytest.mark.parametrize("jacobian, adjoint", [
    (jax.jacrev, None), (jax.jacfwd, diffrax.ForwardMode())], ids=["reverse", "forward"])


@DERIVATIVE_MODES
def test_connection_length_gradients_match_circular_tokamak_closed_form(jacobian, adjoint):
    seeds = jnp.stack([_seed(0.55, 0.3, 0.2), _seed(0.8, -0.3, 1.0)])  # both above the plate

    def traced(parameters, seeds):
        B_poloidal, depth = parameters
        field = CircularTokamakField(MAJOR_RADIUS, B_TOROIDAL, B_poloidal)
        return connection_length(field, seeds, _plate(depth), max_length=200.0,
                                 tolerance=TOLERANCE, adjoint=adjoint)["lengths"]

    def exact(parameters, seeds):
        B_poloidal, depth = parameters
        field = CircularTokamakField(MAJOR_RADIUS, B_TOROIDAL, B_poloidal)
        return circular_tokamak_plate_solution(field, depth, seeds)[0]

    parameters = jnp.array([B_POLOIDAL, PLATE_DEPTH])
    for argnum in (0, 1):
        numerical = jacobian(traced, argnums=argnum)(parameters, seeds)
        expected = jax.jacfwd(exact, argnums=argnum)(parameters, seeds)
        assert jnp.allclose(numerical, expected, rtol=1e-6, atol=1e-8), argnum


class _HyperbolicXPointField:
    """Straight X-line ``B = (k y, k x, B_z)``: ``x + y`` grows and ``x - y`` decays along +B."""

    def __init__(self, shear, axial):
        self.shear, self.axial = shear, axial

    def B_contravariant(self, xyz):
        return jnp.array([self.shear * xyz[1], self.shear * xyz[0], self.axial])


def _x_point_length(shear, axial, u0, plate):
    """Arclength from ``(u0/2, u0/2, 0)`` along +B to ``x + y = plate``."""
    b = 0.5 * (shear / axial) ** 2

    def primitive(u):
        s = jnp.sqrt(1.0 + b * u**2)
        return s + jnp.log(u) - jnp.log1p(s)

    return axial / shear * (primitive(plate) - primitive(u0))


def test_connection_length_resolves_logarithmic_divergence_near_an_x_line():
    shear, axial, plate, max_length = 0.7, 1.0, 1.0, 80.0
    u0 = jnp.logspace(-1, -8, 8)
    seeds = jnp.stack([u0 / 2, u0 / 2, jnp.zeros_like(u0)], axis=1)
    result = connection_length(_HyperbolicXPointField(shear, axial), seeds,
                               lambda xyz: plate - xyz[0] - xyz[1], max_length=max_length,
                               tolerance=TOLERANCE)
    expected = _x_point_length(shear, axial, u0, plate)
    assert jnp.all(result["hit"][:, 0]) and not jnp.any(result["hit"][:, 1])
    # The absolute tolerance acts on the distance u0 to the X-line, and an error du0
    # shifts the length by (B_z / k) du0 / u0, so accuracy degrades like tolerance / u0.
    error_bound = 1e-8 * expected + axial / shear * TOLERANCE / u0
    assert jnp.all(jnp.abs(result["lengths"][:, 0] - expected) < error_bound)
    assert jnp.allclose(result["lengths"][:, 1], max_length)
    # Each decade closer to the X-line adds (B_z / k) log(10) of connection length.
    assert jnp.allclose(jnp.diff(result["lengths"][:, 0]), axial / shear * jnp.log(10.0), rtol=1e-3)


@DERIVATIVE_MODES
def test_connection_length_seed_gradient_matches_x_line_closed_form(jacobian, adjoint):
    shear, axial, plate = 0.7, 1.0, 1.0
    field = _HyperbolicXPointField(shear, axial)

    def traced(u0):
        seeds = jnp.stack([u0 / 2, u0 / 2, jnp.zeros_like(u0)], axis=1)
        return connection_length(field, seeds, lambda xyz: plate - xyz[0] - xyz[1],
                                 max_length=80.0, tolerance=TOLERANCE,
                                 adjoint=adjoint)["lengths"][:, 0]

    u0 = jnp.array([1e-2, 1e-4, 1e-6])
    numerical = jnp.diag(jacobian(traced)(u0))
    expected = jax.vmap(jax.grad(lambda u: _x_point_length(shear, axial, u, plate)))(u0)
    assert jnp.allclose(numerical, expected, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
