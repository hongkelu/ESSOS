"""Planar coils for a two-family coil set: convex encircling coils and circular shaping coils.

Every coil lies in a plane. With centre c, in-plane unit vectors (e1, e2) and radius r(t), t in [0, 2 pi),

    x(t) = c + r(t) (cos(t) e1 + sin(t) e2),   r(t) = r0 (1 + size + sum_k a_k cos(k t) + b_k sin(k t)).

The optimizer moves a coil by a centre shift, a rotation vector that tilts its plane, the relative size
and the relative harmonics (a_k, b_k), and its current. A planar curve has a signed curvature, so
"convex everywhere" is the inequality kappa(t) >= 0 and needs no winding-number penalty.

A coil set is one :class:`PlanarCoils` per family (encircling, shaping), each with its own symmetry,
resolution and currents; :func:`biot_savart` evaluates a family and the fields add.
"""
import jax
import jax.numpy as jnp
import numpy as np

from essos.coils import apply_symmetries_to_currents, apply_symmetries_to_gammas


def rotation_matrix(w):
    """Rotation matrix of the rotation vector ``w`` (Rodrigues), smooth through ``w = 0``."""
    theta2 = jnp.sum(w * w)
    small = theta2 < 1e-12
    safe = jnp.where(small, 1.0, theta2)
    theta = jnp.sqrt(safe)
    a = jnp.where(small, 1.0 - theta2 / 6.0, jnp.sin(theta) / theta)
    b = jnp.where(small, 0.5 - theta2 / 24.0, (1.0 - jnp.cos(theta)) / safe)
    K = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    return jnp.eye(3) + a * K + b * (K @ K)


def frames_from_normals(normals):
    """Right-handed orthonormal rows (e1, e2, n) for each normal, with e1 horizontal."""
    normals = np.atleast_2d(np.asarray(normals, dtype=float))
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    e1 = np.cross([0.0, 0.0, 1.0], normals)
    vertical = np.linalg.norm(e1, axis=1) < 1e-8          # a horizontal plane: any horizontal e1 will do
    e1[vertical] = [1.0, 0.0, 0.0]
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return np.stack((e1, np.cross(normals, e1), normals), axis=1)


class PlanarCoils:
    """One family of planar coils, the base coils of a (field-period, stellarator) symmetric set.

    ``centres`` and ``normals`` are (n, 3), ``radius`` and ``currents`` scalars or (n,). ``order`` is the
    number of radius harmonics. The reference geometry is fixed; :meth:`initial_parameters` returns the
    dimensionless variables every other method takes.
    """

    def __init__(self, centres, normals, radius, currents, *, order=0, nfp=1, stellsym=True, n_segments=64):
        self.centres = np.atleast_2d(np.asarray(centres, dtype=float))
        n = self.centres.shape[0]
        self.frames = frames_from_normals(normals)
        if self.frames.shape[0] != n:
            raise ValueError("centres and normals must have one row per coil")
        self.radius = np.broadcast_to(np.asarray(radius, dtype=float), (n,)).copy()
        currents = np.broadcast_to(np.asarray(currents, dtype=float), (n,)).copy()
        self.current_scale = float(np.max(np.abs(currents))) or 1.0
        self._currents = currents / self.current_scale
        self.order, self.nfp, self.stellsym, self.n_segments = int(order), int(nfp), bool(stellsym), int(n_segments)
        self.t = 2 * np.pi * np.arange(self.n_segments) / self.n_segments

    def __len__(self):
        return self.centres.shape[0]

    def initial_parameters(self):
        """Variables of the reference coils: shift [m], tilt [rad], size and shape (relative), current (scaled)."""
        n = len(self)
        return dict(shift=jnp.zeros((n, 3)), tilt=jnp.zeros((n, 3)), size=jnp.zeros(n),
                    shape=jnp.zeros((n, 2 * self.order)), current=jnp.asarray(self._currents))

    def _radius(self, p):
        """r(t) and its first two t-derivatives, each (n, n_segments)."""
        k = jnp.arange(1, self.order + 1)
        cosine, sine = jnp.cos(jnp.outer(k, self.t)), jnp.sin(jnp.outer(k, self.t))   # (order, n_segments)
        a, b = p["shape"][:, 0::2], p["shape"][:, 1::2]
        r0 = jnp.asarray(self.radius)[:, None]
        r = r0 * (1.0 + p["size"][:, None] + a @ cosine + b @ sine)
        dr = r0 * ((b * k) @ cosine - (a * k) @ sine)
        ddr = -r0 * ((a * k * k) @ cosine + (b * k * k) @ sine)
        return r, dr, ddr

    def _axes(self, p):
        rotation = jax.vmap(rotation_matrix)(p["tilt"])
        return jnp.einsum("nij,nkj->nki", rotation, jnp.asarray(self.frames))        # rows e1, e2, normal

    def normals(self, p):
        """Unit normals of the coil planes, (n, 3)."""
        return self._axes(p)[:, 2]

    def centre_points(self, p):
        """Coil centres, (n, 3)."""
        return jnp.asarray(self.centres) + p["shift"]

    def gamma(self, p):
        """Base-coil points, (n, n_segments, 3)."""
        r, _, _ = self._radius(p)
        axes = self._axes(p)
        cos, sin = jnp.cos(self.t)[None, :, None], jnp.sin(self.t)[None, :, None]
        return self.centre_points(p)[:, None, :] + r[..., None] * (cos * axes[:, None, 0] + sin * axes[:, None, 1])

    def gamma_dash(self, p):
        """Derivative of the base-coil points with respect to the curve parameter in [0, 1)."""
        r, dr, _ = self._radius(p)
        axes = self._axes(p)
        cos, sin = jnp.cos(self.t)[None, :], jnp.sin(self.t)[None, :]
        return 2 * jnp.pi * ((dr * cos - r * sin)[..., None] * axes[:, None, 0]
                             + (dr * sin + r * cos)[..., None] * axes[:, None, 1])

    def length(self, p):
        """Coil lengths, (n,)."""
        return jnp.mean(jnp.linalg.norm(self.gamma_dash(p), axis=-1), axis=1)

    def signed_curvature(self, p):
        """Signed in-plane curvature, (n, n_segments): positive where the coil is convex, 1/r for a circle."""
        r, dr, ddr = self._radius(p)
        return (r * r + 2 * dr * dr - r * ddr) / (r * r + dr * dr) ** 1.5

    def currents(self, p):
        """Base-coil currents in ampere, (n,)."""
        return p["current"] * self.current_scale

    def expanded(self, p):
        """Points, parameter derivatives and currents of every coil after the symmetries."""
        return (apply_symmetries_to_gammas(self.gamma(p), self.nfp, self.stellsym),
                apply_symmetries_to_gammas(self.gamma_dash(p), self.nfp, self.stellsym),
                apply_symmetries_to_currents(self.currents(p), self.nfp, self.stellsym))

    def B(self, p, points):
        """Field of the whole family at ``points`` (m, 3)."""
        return biot_savart(points, *self.expanded(p))


def biot_savart(points, gamma, gamma_dash, currents):
    """Field at ``points`` (m, 3) of closed filaments ``gamma`` (n, s, 3) with ``currents`` (n,)."""
    def one(x):
        d = x - gamma
        integrand = jnp.cross(gamma_dash, d) / jnp.linalg.norm(d, axis=-1, keepdims=True) ** 3
        return 1e-7 * jnp.einsum("n,nsk->k", currents, integrand) / gamma.shape[1]
    return jax.vmap(one)(points)


def convexity_loss(curvature, curvature_limit):
    """Mean squared violation of 0 <= kappa <= curvature_limit, in units of the limit."""
    low, high = jnp.maximum(-curvature, 0.0), jnp.maximum(curvature - curvature_limit, 0.0)
    return jnp.mean(low * low + high * high) / curvature_limit**2


def separation_loss(a, b, minimum, *, same=False):
    """Squared shortfall of the point distances between two sets of coils below ``minimum``.

    ``a`` and ``b`` are (n, s, 3) point sets; ``same=True`` compares a set with itself and skips each
    coil against itself. Summed over the pairs, per point of ``a``, in units of ``minimum``.
    """
    distance = jnp.linalg.norm(a[:, None, :, None, :] - b[None, :, None, :, :], axis=-1)   # (na, nb, sa, sb)
    shortfall = jnp.maximum(minimum - distance, 0.0) / minimum
    if same:
        shortfall = shortfall * (1.0 - jnp.eye(a.shape[0]))[:, :, None, None]
    return jnp.sum(shortfall * shortfall) / (a.shape[0] * a.shape[1])


def minimum_distance(a, b, *, same=False):
    """Smallest point distance between two sets of coils (diagnostic)."""
    distance = jnp.linalg.norm(a[:, None, :, None, :] - b[None, :, None, :, :], axis=-1)
    if same:
        distance = distance + 1e30 * jnp.eye(a.shape[0])[:, :, None, None]
    return jnp.min(distance)


class FourierBoundary:
    """A stellarator-symmetric boundary R = sum rmnc cos(m theta - n phi), Z = sum zmns sin(m theta - n phi).

    ``xn`` carries the field-period factor, as in a VMEC wout file. Calling it with broadcastable ``theta``
    (varying along the first axis) and ``phi`` returns the points and the unit outward normals.
    """

    def __init__(self, rmnc, zmns, xm, xn):
        self.rmnc, self.zmns, self.xm, self.xn = (np.asarray(v, dtype=float) for v in (rmnc, zmns, xm, xn))

    def __call__(self, theta, phi):
        theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
        angle = theta[..., None] * self.xm - phi[..., None] * self.xn
        cos, sin = np.cos(angle), np.sin(angle)
        R, Z = cos @ self.rmnc, sin @ self.zmns
        R_theta, Z_theta = -sin @ (self.rmnc * self.xm), cos @ (self.zmns * self.xm)
        R_phi, Z_phi = sin @ (self.rmnc * self.xn), -cos @ (self.zmns * self.xn)
        c, s = np.cos(phi), np.sin(phi)
        points = np.stack((R * c, R * s, Z), axis=-1)
        d_theta = np.stack((R_theta * c, R_theta * s, Z_theta), axis=-1)
        d_phi = np.stack((R_phi * c - R * s, R_phi * s + R * c, Z_phi), axis=-1)
        normal = np.cross(d_phi, d_theta)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        # Outward: away from the cross-section centroid of the same toroidal plane.
        centre = np.stack((np.mean(R, axis=0) * c, np.mean(R, axis=0) * s, np.mean(Z, axis=0) + 0 * c), axis=-1)
        sign = np.sign(np.sum(np.sum(normal * (points - centre), axis=-1), axis=0, keepdims=True))
        return points, normal * sign[..., None]


def encircling_coils(boundary, nfp, n_coils, clearance, current, *, order=4, n_segments=96, ntheta=256):
    """Circular coils in ``n_coils`` poloidal planes of a half field period, around the cross-section.

    Each circle is centred on the cross-section centroid and clears the farthest boundary point of its
    plane by ``clearance``. Cell-centred toroidal angles keep the coils off the symmetry planes.
    """
    phi = (np.arange(n_coils) + 0.5) * np.pi / (nfp * n_coils)
    theta = 2 * np.pi * np.arange(ntheta) / ntheta
    points, _ = boundary(theta[:, None], phi[None, :])                               # (ntheta, n_coils, 3)
    centres = points.mean(axis=0)
    radius = np.max(np.linalg.norm(points - centres, axis=-1), axis=0) + clearance
    normals = np.stack((-np.sin(phi), np.cos(phi), 0 * phi), axis=1)
    return PlanarCoils(centres, normals, radius, current, order=order, nfp=nfp, stellsym=True, n_segments=n_segments)


def shaping_coils(boundary, nfp, n_phi, n_theta, offset, radius, current_scale, *, n_segments=16):
    """Circles of one ``radius`` on the surface ``offset`` outside the boundary, facing the plasma.

    A cell-centred ``n_phi`` by ``n_theta`` grid over a half field period. Currents start at zero and
    are measured in units of ``current_scale`` ampere.
    """
    phi = (np.arange(n_phi) + 0.5) * np.pi / (nfp * n_phi)
    theta = (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta
    points, normals = boundary(theta[:, None], phi[None, :])
    centres = (points + offset * normals).reshape(-1, 3)
    coils = PlanarCoils(centres, normals.reshape(-1, 3), radius, current_scale, nfp=nfp, stellsym=True,
                        n_segments=n_segments)
    coils._currents = np.zeros(len(coils))
    return coils
