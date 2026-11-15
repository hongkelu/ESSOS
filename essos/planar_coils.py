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


def coil_loads(groups):
    """Lorentz loads on closed filaments from the field of all the *other* filaments.

    ``groups`` is a sequence of ``(gamma, gamma_dash, currents)`` with shapes (n, s, 3), (n, s, 3), (n,); the groups may
    have different numbers of points per coil. A coil's own field is left out: it gives no net force or torque on a
    rigid loop (only the hoop stress, which needs the conductor size). Returns one dict per group with the force per
    length [N/m] (n, s, 3), the net force [N] (n, 3) and the torque about the coil centroid [N m] (n, 3).
    """
    loads = []
    for a, (gamma, gamma_dash, currents) in enumerate(groups):
        def field(x, own):
            B = jnp.zeros(3)
            for b, (g, d, c) in enumerate(groups):
                r = x - g
                r2 = jnp.sum(r * r, axis=-1)
                other = jnp.ones(len(c)) if b != a else (jnp.arange(len(c)) != own).astype(r2.dtype)
                r2 = jnp.where(other[:, None] > 0, r2, 1.0)
                B = B + 1e-7 * jnp.einsum("n,nsk->k", c * other, jnp.cross(d, r) / r2[..., None] ** 1.5) / g.shape[1]
            return B
        B = jax.vmap(lambda xs, own: jax.vmap(lambda x: field(x, own))(xs))(gamma, jnp.arange(len(currents)))
        pull = currents[:, None, None] * jnp.cross(gamma_dash, B)                   # per unit curve parameter
        arm = gamma - jnp.mean(gamma, axis=1, keepdims=True)
        loads.append(dict(force_per_length=pull / jnp.linalg.norm(gamma_dash, axis=-1, keepdims=True),
                          force=jnp.mean(pull, axis=1), torque=jnp.mean(jnp.cross(arm, pull), axis=1)))
    return loads


def convexity_loss(curvature, curvature_limit):
    """Mean squared violation of 0 <= kappa <= curvature_limit, in units of the limit."""
    low, high = jnp.maximum(-curvature, 0.0), jnp.maximum(curvature - curvature_limit, 0.0)
    return jnp.mean(low * low + high * high) / curvature_limit**2


def _distances(a, b, same):
    """Point distances (na, nb, sa, sb); a coil against itself is pushed far away, with a zero gradient."""
    squared = jnp.sum((a[:, None, :, None, :] - b[None, :, None, :, :]) ** 2, axis=-1)
    if same:
        squared = squared + 1e12 * jnp.eye(a.shape[0])[:, :, None, None]
    return jnp.sqrt(squared)


def separation_loss(a, b, minimum, *, same=False):
    """Squared shortfall of the point distances between two sets of coils below ``minimum``.

    ``a`` and ``b`` are (n, s, 3) point sets; ``same=True`` compares a set with itself and skips each
    coil against itself. Summed over the pairs, per point of ``a``, in units of ``minimum``.
    """
    shortfall = jnp.maximum(minimum - _distances(a, b, same), 0.0) / minimum
    return jnp.sum(shortfall * shortfall) / (a.shape[0] * a.shape[1])


def minimum_distance(a, b, *, same=False):
    """Smallest point distance between two sets of coils (diagnostic)."""
    return jnp.min(_distances(a, b, same))


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


def shaping_coils(boundary, nfp, offset, radius, spacing, current_scale, *, n_segments=16, n_phi=24, n_theta=120):
    """Circles on the surface ``offset`` outside the boundary, facing the plasma.

    ``radius`` is one value or several: the sizes are placed in turn, largest first, so smaller coils fill
    the gaps the larger ones leave. Candidate centres are a cell-centred ``n_phi`` by ``n_theta`` grid over
    a half field period; a candidate is kept when its circle stays at least ``spacing`` from every circle
    already kept and from all symmetric images. Currents start at zero and are measured in units of
    ``current_scale`` ampere.
    """
    phi = (np.arange(n_phi) + 0.5) * np.pi / (nfp * n_phi)
    theta = (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta
    points, normals = boundary(theta[:, None], phi[None, :])
    candidates = np.swapaxes(points + offset * normals, 0, 1).reshape(-1, 3)        # plane by plane
    candidate_normals = np.swapaxes(normals, 0, 1).reshape(-1, 3)
    candidate_angles = np.stack(np.broadcast_arrays(theta[None, :], phi[:, None]), axis=-1).reshape(-1, 2)
    kept, radii, images, image_radii = [], [], np.zeros((0, 3)), np.zeros(0)
    for size in sorted(np.atleast_1d(np.asarray(radius, dtype=float)), reverse=True):
        for index, centre in enumerate(candidates):
            own = np.asarray(apply_symmetries_to_gammas(jnp.asarray(centre)[None, None, :], nfp, True))[:, 0]
            if np.min(np.linalg.norm(own[1:] - centre, axis=1)) < 2 * size + spacing:
                continue
            if images.shape[0] and np.min(np.linalg.norm(images - centre, axis=1) - image_radii) < size + spacing:
                continue
            kept.append(index)
            radii.append(size)
            images = np.concatenate((images, own))
            image_radii = np.concatenate((image_radii, np.full(own.shape[0], size)))
    coils = PlanarCoils(candidates[kept], candidate_normals[kept], np.array(radii), current_scale, nfp=nfp, stellsym=True,
                        n_segments=n_segments)
    coils._currents = np.zeros(len(coils))
    coils.theta, coils.phi = candidate_angles[kept].T                                # where each coil sits on the surface
    return coils


def dipole_field(points, positions, moments):
    """Field at ``points`` (m, 3) of point dipoles at ``positions`` (n, 3) with ``moments`` (n, 3) in A m^2."""
    def one(x):
        r = x - positions
        d = jnp.linalg.norm(r, axis=-1, keepdims=True)
        return 1e-7 * jnp.sum(3 * jnp.sum(moments * r, axis=-1, keepdims=True) * r / d**5 - moments / d**3, axis=0)
    return jax.vmap(one)(points)


class CurrentPotential:
    """A single-valued current potential on a winding surface, the continuous limit of a shaping-coil array.

    Phi(theta, phi) = sum_mn c_mn sin(m theta - n nfp phi) [ampere] is a stellarator-symmetric sheet of
    surface current n x grad(Phi), equivalent to a layer of magnetic dipoles of density Phi along the
    surface normal. Its normal field on the plasma is linear in c, so the best sheet for a given
    regularization is one linear solve; it bounds what any array of shaping coils on that surface can
    do and tells how much current they need. ``surface(theta, phi)`` returns points and unit normals.
    """

    def __init__(self, surface, nfp, mpol, ntor, *, n_theta=64, n_phi=32):
        self.nfp = int(nfp)
        theta = (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta
        phi = (np.arange(nfp * n_phi) + 0.5) * 2 * np.pi / (nfp * n_phi)             # the whole torus
        points, normals = surface(theta[:, None], phi[None, :])
        d_theta = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / 2
        d_phi = (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)) / 2
        self.area = np.linalg.norm(np.cross(d_theta, d_phi), axis=-1).ravel()        # of each cell
        self.points, self.normals = points.reshape(-1, 3), normals.reshape(-1, 3)
        self.theta, self.phi = (v.ravel() for v in np.broadcast_arrays(theta[:, None], phi[None, :]))
        self.modes = np.array([(m, n) for m in range(mpol + 1) for n in range(-ntor, ntor + 1) if m > 0 or n > 0])
        self.basis = self._basis(self.theta, self.phi)

    def _basis(self, theta, phi):
        m, n = self.modes.T
        return np.sin(np.asarray(theta)[..., None] * m - np.asarray(phi)[..., None] * n * self.nfp)

    def normal_field_matrix(self, points, normals):
        """Matrix from the coefficients to B.n at ``points`` with unit ``normals``."""
        positions, moments = jnp.asarray(self.points), jnp.asarray(self.normals * self.area[:, None])

        def row(x, normal):                                 # B.n at x of a unit dipole density in each cell
            r = x - positions
            d = jnp.linalg.norm(r, axis=-1)
            return 1e-7 * (3 * jnp.sum(moments * r, axis=-1) * (r @ normal) / d**5 - (moments @ normal) / d**3)
        return np.asarray(jax.vmap(row)(jnp.asarray(points), jnp.asarray(normals))) @ self.basis

    def solve(self, matrix, target, regularization):
        """Coefficients minimizing mean (B.n - target)^2 + regularization * surface mean of (Phi / Phi_0)^2.

        Phi_0 = 1 A, so ``regularization`` has units T^2 / A^2; scan it to trade field error against
        the size of the potential.
        """
        weight = np.sqrt(regularization * self.area / self.area.sum())[:, None] * self.basis
        system = np.concatenate((matrix / np.sqrt(matrix.shape[0]), weight))
        rhs = np.concatenate((np.asarray(target) / np.sqrt(matrix.shape[0]), np.zeros(weight.shape[0])))
        return np.linalg.lstsq(system, rhs, rcond=None)[0]

    def potential(self, coefficients, theta=None, phi=None):
        """Phi on the surface grid, or at the given angles."""
        basis = self.basis if theta is None else self._basis(theta, phi)
        return basis @ np.asarray(coefficients)

    def coil_currents(self, coefficients, coils):
        """Currents of an array of shaping coils that carries the potential.

        Each coil takes the dipole moment of the surface cells nearest to it (or to one of its symmetric
        images, with the sign of the image) and turns it into a current through its own area.
        """
        centres = np.asarray(apply_symmetries_to_gammas(jnp.asarray(coils.centres)[:, None, :], coils.nfp,
                                                        coils.stellsym))[:, 0]
        owner = np.empty(self.points.shape[0], dtype=int)
        for start in range(0, self.points.shape[0], 2048):  # nearest coil of each cell, in blocks
            block = self.points[start:start + 2048]
            owner[start:start + 2048] = np.argmin(np.linalg.norm(block[:, None] - centres[None], axis=-1), axis=1)
        sign = np.asarray(apply_symmetries_to_currents(np.ones(len(coils)), coils.nfp, coils.stellsym))
        moment = np.bincount(owner, weights=self.potential(coefficients) * self.area, minlength=centres.shape[0])
        base = np.arange(centres.shape[0]) % len(coils)
        moments = np.bincount(base, weights=moment * sign, minlength=len(coils)) / (centres.shape[0] // len(coils))
        return moments / (np.pi * coils.radius**2)


def offset_surface(boundary, offset):
    """The surface ``offset`` outside a :class:`FourierBoundary`, with the boundary normals."""
    def surface(theta, phi):
        points, normals = boundary(theta, phi)
        return points + offset * normals, normals
    return surface


def _symmetric_copies(gamma, gamma_dash, nfp):
    """A base coil and its field-period and stellarator-symmetric images: points, derivatives, current signs."""
    flip = jnp.array([1.0, -1.0, -1.0])
    points, derivatives, signs = [], [], []
    for k in range(nfp):
        angle = 2 * jnp.pi * k / nfp
        rotation = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0.0], [jnp.sin(angle), jnp.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        for mirrored in (False, True):
            g, d = (gamma * flip, gamma_dash * flip) if mirrored else (gamma, gamma_dash)
            points.append(g @ rotation.T)
            derivatives.append(d @ rotation.T)
            signs.append(-1.0 if mirrored else 1.0)
    return jnp.stack(points), jnp.stack(derivatives), jnp.array(signs)


def planar_coil_set_cost(gamma, normal, B, nfp, *, n_encircling, sheet_distance, encircling_clearance,
                         mpol=8, ntor=8, regularization=2.5e-3, n_segments=64):
    """How well circular encircling coils plus a current sheet can make the field ``B`` on a boundary.

    A differentiable measure of how hard a plasma is for planar encircling coils with an array of shaping
    coils, for use inside a plasma optimization (the boundary may be traced). ``gamma``, ``normal`` and
    ``B`` are (3, nphi, ntheta) on one field period of a stellarator-symmetric boundary, on uniform grids
    that start at zero angle. The coil set follows the boundary:

    * ``n_encircling`` circles per half field period in poloidal planes, each centred on its cross-section
      and clearing it by ``encircling_clearance``; their currents are free;
    * a single-valued current potential (dipole layer) on the surface ``sheet_distance`` outside the boundary.

    The currents and the potential are eliminated by one linear solve of

        cost = mean |B_coils - B|^2 / |B|^2  +  regularization * mean (Phi / I0)^2 ,

    I0 being the mean encircling-coil current the poloidal flux function of ``B`` asks for. Returns the
    cost, ``residuals`` whose sum of squares is the cost, and its parts: ``field_error`` (rms |dB|/|B|), ``potential_rms`` (in units of I0), ``currents``.
    """
    points = jnp.moveaxis(gamma, 0, -1)                                             # (nphi, ntheta, 3)
    normals = jnp.moveaxis(normal, 0, -1)
    normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
    field = jnp.moveaxis(B, 0, -1)
    nphi, ntheta = points.shape[:2]

    def turned(v, k):                                                               # rotate by k field periods
        angle = 2 * jnp.pi * k / nfp
        rotation = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0.0], [jnp.sin(angle), jnp.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        return v @ rotation.T
    torus = jnp.concatenate([turned(points, k) for k in range(nfp)])                # (nfp nphi, ntheta, 3)
    torus_normals = jnp.concatenate([turned(normals, k) for k in range(nfp)])

    # The sheet: a dipole layer of density Phi along the boundary normal, on the offset surface.
    sheet = torus + sheet_distance * torus_normals
    d_phi = (jnp.roll(sheet, -1, axis=0) - jnp.roll(sheet, 1, axis=0)) / 2
    d_theta = (jnp.roll(sheet, -1, axis=1) - jnp.roll(sheet, 1, axis=1)) / 2
    area = jnp.linalg.norm(jnp.cross(d_phi, d_theta), axis=-1).reshape(-1)
    positions, moments = sheet.reshape(-1, 3), torus_normals.reshape(-1, 3) * area[:, None]
    phi = 2 * np.pi * np.arange(nfp * nphi) / (nfp * nphi)
    theta = 2 * np.pi * np.arange(ntheta) / ntheta
    modes = np.array([(m, n) for m in range(mpol + 1) for n in range(-ntor, ntor + 1) if m > 0 or n > 0])
    basis = jnp.asarray(np.sin(theta[None, :, None] * modes[:, 0] - phi[:, None, None] * modes[:, 1] * nfp).reshape(-1, len(modes)))

    half = slice(0, nphi // 2)                                                      # stellarator symmetry
    targets = points[half].reshape(-1, 3)
    wanted = field[half].reshape(-1, 3)
    strength = jnp.linalg.norm(wanted, axis=-1)

    def dipole_rows(x):                                                             # field at x of unit density in each cell
        r = x - positions
        d = jnp.linalg.norm(r, axis=-1, keepdims=True)
        return 1e-7 * (3 * jnp.sum(moments * r, axis=-1, keepdims=True) * r / d**5 - moments / d**3)
    sheet_matrix = jnp.einsum("pck,cm->pkm", jax.vmap(dipole_rows)(targets), basis)   # (points, 3, modes)

    # Encircling circles around the cross-sections nearest to equally spaced planes of the half period.
    planes = np.rint((np.arange(n_encircling) + 0.5) * nphi / (2 * n_encircling)).astype(int)
    t = 2 * np.pi * np.arange(n_segments) / n_segments
    radial = np.stack((np.cos(2 * np.pi * planes / (nfp * nphi)), np.sin(2 * np.pi * planes / (nfp * nphi)), 0 * planes), axis=1)
    centres = jnp.mean(points[planes], axis=1)
    radii = jnp.max(jnp.linalg.norm(points[planes] - centres[:, None], axis=-1), axis=1) + encircling_clearance
    in_plane = (jnp.cos(t)[None, :, None] * jnp.asarray(radial)[:, None] + jnp.sin(t)[None, :, None] * jnp.array([0.0, 0.0, 1.0]))
    tangent = (-jnp.sin(t)[None, :, None] * jnp.asarray(radial)[:, None] + jnp.cos(t)[None, :, None] * jnp.array([0.0, 0.0, 1.0]))
    circles = centres[:, None] + radii[:, None, None] * in_plane                     # (coils, segments, 3)
    circle_dash = 2 * jnp.pi * radii[:, None, None] * tangent

    def coil_column(g, d):                                                          # field per ampere of a coil and its images
        copies, derivatives, signs = _symmetric_copies(g, d, nfp)
        return biot_savart(targets, copies, derivatives, signs)
    coil_matrix = jnp.moveaxis(jax.vmap(coil_column)(circles, circle_dash), 0, -1)    # (points, 3, coils)

    radius = jnp.linalg.norm(targets[:, :2], axis=-1)
    unit_current = 2 * jnp.pi * jnp.mean(radius * strength) / (4e-7 * jnp.pi) / (2 * nfp * n_encircling)   # I0
    n_points, n_modes = targets.shape[0], len(modes)
    scale = 1.0 / (strength[:, None, None] * jnp.sqrt(n_points))
    fit_rows = (jnp.concatenate((sheet_matrix, coil_matrix), axis=-1) * unit_current * scale).reshape(3 * n_points, -1)
    smooth_rows = jnp.concatenate((jnp.sqrt(regularization * area / jnp.sum(area))[:, None] * basis,
                                   jnp.zeros((basis.shape[0], n_encircling))), axis=1)
    system = jnp.concatenate((fit_rows, smooth_rows))
    rhs = jnp.concatenate(((wanted * scale[:, :, 0]).reshape(-1), jnp.zeros(basis.shape[0])))
    solution = jnp.linalg.solve(system.T @ system + 1e-12 * jnp.eye(system.shape[1]), system.T @ rhs)
    misfit = fit_rows @ solution - rhs[:3 * n_points]
    potential = basis @ solution[:n_modes]
    potential_mean_square = jnp.sum(potential**2 * area) / jnp.sum(area)
    # The same cost as a sum of squares, for least-squares optimizers: the field misfit and the potential in its modes.
    smooth = jnp.linalg.cholesky(smooth_rows[:, :n_modes].T @ smooth_rows[:, :n_modes] + 1e-14 * jnp.eye(n_modes)).T
    return dict(cost=jnp.sum(misfit**2) + regularization * potential_mean_square,
                residuals=jnp.concatenate((misfit, smooth @ solution[:n_modes])),
                field_error=jnp.sqrt(jnp.sum(misfit**2)), potential_rms=jnp.sqrt(potential_mean_square),
                currents=solution[n_modes:], unit_current=unit_current, radii=radii)
