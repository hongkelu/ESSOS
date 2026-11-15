import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.planar_coils import (FourierBoundary, PlanarCoils, biot_savart, convexity_loss, encircling_coils,
                                minimum_distance, rotation_matrix, separation_loss, shaping_coils)

jax.config.update("jax_enable_x64", True)


def _family(order=3, **kwargs):
    return PlanarCoils([[1.0, 0.2, 0.1], [0.4, 1.1, -0.2]], [[0.0, 1.0, 0.0], [0.6, 0.0, 0.8]], [0.5, 0.3],
                       [2.0e5, -1.0e5], order=order, **kwargs)


def test_rotation_matrix_is_a_rotation_and_smooth_at_zero():
    for w in ([0.3, -0.5, 0.9], [1e-9, 0.0, -2e-9], [0.0, 0.0, 0.0]):
        R = np.asarray(rotation_matrix(jnp.array(w)))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(R) == pytest.approx(1.0)
    jacobian = jax.jacfwd(rotation_matrix)(jnp.zeros(3))
    assert np.all(np.isfinite(jacobian)) and float(jacobian[2, 1, 0]) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation_matrix(jnp.array([0.0, 0.0, np.pi / 2])) @ jnp.array([1.0, 0.0, 0.0]),
                               [0.0, 1.0, 0.0], atol=1e-15)


def test_reference_coils_are_planar_circles_with_curvature_one_over_radius():
    coils = _family()
    p = coils.initial_parameters()
    gamma = np.asarray(coils.gamma(p))
    np.testing.assert_allclose(np.linalg.norm(gamma - coils.centres[:, None], axis=-1), coils.radius[:, None] + 0 * gamma[..., 0])
    np.testing.assert_allclose(np.einsum("nsk,nk->ns", gamma - coils.centres[:, None], np.asarray(coils.normals(p))), 0, atol=1e-15)
    np.testing.assert_allclose(coils.signed_curvature(p), (1 / coils.radius)[:, None] + 0 * gamma[..., 0])
    np.testing.assert_allclose(coils.length(p), 2 * np.pi * coils.radius)
    np.testing.assert_allclose(coils.currents(p), [2.0e5, -1.0e5])
    assert float(convexity_loss(coils.signed_curvature(p), 5.0)) == 0.0


def test_shaped_tilted_coil_stays_planar_and_matches_finite_differences():
    coils = _family(n_segments=400)
    p = coils.initial_parameters()
    p["shift"] = jnp.array([[0.01, -0.02, 0.03], [0.0, 0.05, 0.0]])
    p["tilt"] = jnp.array([[0.2, -0.1, 0.3], [-0.4, 0.2, 0.1]])
    p["size"] = jnp.array([0.1, -0.05])
    p["shape"] = jnp.array([[0.05, 0.02, -0.03, 0.04, 0.01, -0.02], [0.0, 0.1, 0.02, 0.0, -0.01, 0.03]])
    gamma, dash = np.asarray(coils.gamma(p)), np.asarray(coils.gamma_dash(p))
    centred = gamma - np.asarray(coils.centre_points(p))[:, None]
    np.testing.assert_allclose(np.einsum("nsk,nk->ns", centred, np.asarray(coils.normals(p))), 0, atol=1e-14)
    spectral = np.fft.ifft(2j * np.pi * np.fft.fftfreq(400, 1 / 400)[None, :, None] * np.fft.fft(gamma, axis=1), axis=1).real
    np.testing.assert_allclose(dash, spectral, atol=1e-10)
    second = np.fft.ifft(2j * np.pi * np.fft.fftfreq(400, 1 / 400)[None, :, None] * np.fft.fft(dash, axis=1), axis=1).real
    normal = np.asarray(coils.normals(p))[:, None]
    expected = np.einsum("nsk,nsk->ns", np.cross(dash, second), normal + 0 * dash) / np.linalg.norm(dash, axis=-1) ** 3
    np.testing.assert_allclose(coils.signed_curvature(p), expected, rtol=1e-9, atol=1e-9)


def test_signed_curvature_detects_a_concave_coil():
    coils = _family()
    p = coils.initial_parameters()
    p["shape"] = p["shape"].at[0, 4].set(0.3)                # a strong k = 3 lobe: three concave arcs
    kappa = coils.signed_curvature(p)
    assert float(kappa[0].min()) < 0 < float(kappa[1].min())
    assert float(convexity_loss(kappa, 5.0)) > 0
    assert float(convexity_loss(coils.signed_curvature(coils.initial_parameters()), 2.5)) > 0   # 1/0.3 > 2.5


def test_biot_savart_matches_the_axis_field_of_a_circular_loop():
    loop = PlanarCoils([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], 0.4, 3.0e5, n_segments=64)
    p = loop.initial_parameters()
    z = np.array([0.0, 0.3, -0.7])
    B = np.asarray(biot_savart(jnp.stack((0 * z, 0 * z, z), axis=1), loop.gamma(p), loop.gamma_dash(p), loop.currents(p)))
    np.testing.assert_allclose(B[:, 2], 4e-7 * np.pi * 3.0e5 * 0.4**2 / (2 * (0.4**2 + z**2) ** 1.5), rtol=1e-12)
    np.testing.assert_allclose(B[:, :2], 0, atol=1e-15)


def test_symmetry_expansion_gives_a_stellarator_symmetric_field():
    coils = _family(nfp=2, stellsym=True)
    p = coils.initial_parameters()
    gamma, dash, currents = coils.expanded(p)
    assert gamma.shape == (8, 64, 3) and dash.shape == (8, 64, 3) and currents.shape == (8,)
    x = jnp.array([[1.1, 0.3, 0.2]])
    mirror = x * jnp.array([1.0, -1.0, -1.0])
    B, B_mirror = np.asarray(coils.B(p, x))[0], np.asarray(coils.B(p, mirror))[0]
    np.testing.assert_allclose(B_mirror, B * np.array([-1.0, 1.0, 1.0]), rtol=1e-12, atol=1e-18)
    rotated = np.asarray(coils.B(p, x * jnp.array([-1.0, -1.0, 1.0])))[0]
    np.testing.assert_allclose(rotated, B * np.array([-1.0, -1.0, 1.0]), rtol=1e-12, atol=1e-18)


def test_separation_loss_and_minimum_distance():
    a = jnp.zeros((1, 4, 3)).at[0, :, 0].set(jnp.arange(4.0))
    b = a + jnp.array([0.0, 0.5, 0.0])
    assert float(minimum_distance(a, b)) == pytest.approx(0.5)
    assert float(separation_loss(a, b, 0.4)) == 0.0
    assert float(separation_loss(a, b, 1.0)) == pytest.approx(4 * 0.25 / 4)
    both = jnp.concatenate((a, b))
    assert float(minimum_distance(both, both, same=True)) == pytest.approx(0.5)
    assert float(separation_loss(both, both, 1.0, same=True)) == pytest.approx(2 * 4 * 0.25 / 8)
    gradient = jax.grad(lambda q: separation_loss(q, q, 1.0, same=True))(both)       # zero self-distances are masked
    assert np.all(np.isfinite(gradient)) and float(jnp.max(jnp.abs(gradient))) > 0


def test_seeds_clear_the_boundary_and_face_it():
    # Rotating ellipse, two field periods.
    boundary = FourierBoundary([1.0, 0.2, 0.05], [0.0, 0.2, -0.05], [0, 1, 1], [0, 0, 2])
    theta = 2 * np.pi * np.arange(64)[:, None] / 64
    phi = np.pi * (np.arange(16)[None, :] + 0.5) / 32
    points, normals = boundary(theta, phi)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1.0)
    centre = points.mean(axis=0, keepdims=True)
    assert np.all(np.sum(normals * (points - centre), axis=-1) > 0)
    tf = encircling_coils(boundary, 2, 3, 0.15, 1.0e5, order=2)
    p = tf.initial_parameters()
    distance = np.linalg.norm(np.asarray(tf.gamma(p))[:, :, None, None] - points[None, None], axis=-1)
    assert distance.min() > 0.12 and len(tf) == 3
    np.testing.assert_allclose(np.asarray(tf.centre_points(p))[:, 2], 0.0, atol=1e-12)
    dipoles = shaping_coils(boundary, 2, 0.12, 0.03, 0.02, 2.0e4)
    q = dipoles.initial_parameters()
    assert len(dipoles) > 20 and float(jnp.max(jnp.abs(dipoles.currents(q)))) == 0.0
    centres = np.asarray(dipoles.centre_points(q))
    assert np.linalg.norm(centres[:, None, None] - points[None], axis=-1).min() > 0.11
    every = dipoles.expanded(q)[0]
    assert every.shape[0] == 4 * len(dipoles)
    all_centres = np.asarray(every).mean(axis=1)
    apart = np.linalg.norm(all_centres[:, None] - all_centres[None], axis=-1) + 10 * np.eye(len(all_centres))
    assert apart.min() >= 2 * 0.03 + 0.02 - 1e-12
    q["current"] = q["current"].at[3].set(1.0)
    assert float(dipoles.currents(q)[3]) == pytest.approx(2.0e4)


def test_field_is_differentiable_in_every_variable():
    coils = _family(nfp=2)
    p = coils.initial_parameters()
    x = jnp.array([[0.2, 0.1, 0.05], [0.9, -0.4, 0.3]])
    jacobian = jax.jacfwd(lambda q: coils.B(q, x))(p)
    for name, value in jacobian.items():
        assert np.all(np.isfinite(value)) and float(jnp.max(jnp.abs(value))) > 0, name


def test_dipole_field_matches_a_small_current_loop():
    from essos.planar_coils import dipole_field
    loop = PlanarCoils([[0.1, -0.2, 0.3]], [[0.3, 0.4, np.sqrt(0.75)]], 1e-3, 2.0e4, n_segments=64, stellsym=False)
    p = loop.initial_parameters()
    x = jnp.array([[0.6, 0.1, 0.2], [-0.3, 0.4, 0.9]])
    moment = 2.0e4 * np.pi * 1e-6 * np.asarray(loop.normals(p))
    np.testing.assert_allclose(dipole_field(x, jnp.asarray(loop.centres), jnp.asarray(moment)), loop.B(p, x), rtol=1e-4)


def test_current_potential_reproduces_a_potential_and_hands_it_to_coils():
    from essos.planar_coils import CurrentPotential, dipole_field, offset_surface
    boundary = FourierBoundary([1.0, 0.2, 0.05], [0.0, 0.2, -0.05], [0, 1, 1], [0, 0, 2])
    sheet = CurrentPotential(offset_surface(boundary, 0.15), 2, 3, 2, n_theta=48, n_phi=24)
    theta = (np.arange(16)[:, None] + 0.5) * 2 * np.pi / 16
    phi = (np.arange(8)[None, :] + 0.5) * np.pi / 16
    points, normals = (v.reshape(-1, 3) for v in boundary(theta, phi))
    matrix = sheet.normal_field_matrix(points, normals)
    coefficients = np.zeros(len(sheet.modes))
    coefficients[[3, 11]] = [4.0e4, -2.5e4]
    # The matrix is the dipole layer of density Phi along the surface normal.
    moments = (sheet.potential(coefficients) * sheet.area)[:, None] * sheet.normals
    direct = np.sum(np.asarray(dipole_field(jnp.asarray(points), jnp.asarray(sheet.points), jnp.asarray(moments))) * normals, axis=1)
    np.testing.assert_allclose(matrix @ coefficients, direct, rtol=1e-10, atol=1e-16)
    # An unregularized solve of its own field returns the potential; regularization shrinks it.
    np.testing.assert_allclose(sheet.solve(matrix, direct, 0.0), coefficients, rtol=1e-6, atol=1e-3)
    assert np.linalg.norm(sheet.solve(matrix, direct, 1e-12)) < np.linalg.norm(coefficients)
    # Stellarator symmetry: Phi is odd under (theta, phi) -> (-theta, -phi).
    np.testing.assert_allclose(sheet.potential(coefficients, -theta, -phi), -sheet.potential(coefficients, theta, phi), atol=1e-9)
    # Coils take the moment of their cells: the total moment along the normals is kept.
    dipoles = shaping_coils(boundary, 2, 0.15, 0.03, 0.02, 1.0)
    currents = sheet.coil_currents(coefficients, dipoles)
    assert currents.shape == (len(dipoles),) and np.all(np.isfinite(currents)) and np.abs(currents).max() > 0
    local = sheet.potential(coefficients, dipoles.theta, dipoles.phi)                # the potential under each coil
    assert np.corrcoef(local, currents)[0, 1] > 0.9
