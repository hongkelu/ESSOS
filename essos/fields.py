from dataclasses import dataclass
from typing import Any

import jax
jax.config.update("jax_enable_x64", True)
from jax import vmap
from essos.coils import Curves
import jax.numpy as jnp
from functools import partial
from jax import jit, jacfwd, grad, vmap, tree_util, lax
from essos.surfaces import SurfaceRZFourier, BdotN_over_B, SurfaceClassifier
from essos.plot import fix_matplotlib_3d
from essos.util import newton

class MagneticField():
    def __init__(self):
        pass

    @jit
    def sqrtg(self, points):
        raise NotImplementedError("sqrtg method not implemented")

    @jit
    def B(self, points):
        raise NotImplementedError("B method not implemented")

    @jit
    def B_covariant(self, points):
        return self.B(points)

    @jit
    def B_contravariant(self, points):
        return self.B(points)
    
    @jit
    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points))
    
    @jit
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)
    
    @jit
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)
    
    @jit
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)
    
    @jit
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] - grad_B_cov[1][2],
                          grad_B_cov[0][2] - grad_B_cov[2][0],
                          grad_B_cov[1][0] - grad_B_cov[0][1]])/self.sqrtg(points)

    @jit
    def curl_b(self, points):
        return self.curl_B(points) / self.AbsB(points) + jnp.cross(self.B_covariant(points), jnp.array(self.dAbsB_by_dX(points))) / self.AbsB(points)**2 / self.sqrtg(points)
    
    @jit
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points), self.curl_b(points)) * self.sqrtg(points) / self.AbsB(points)
    
    @jit
    def to_xyz(self, points):
        raise NotImplementedError("to_xyz method not implemented")

@jax.custom_jvp
def _regularized_filament_terms(
    gamma_dash: Any,
    displacement: Any,
    weights: Any,
) -> Any:
    """Return weighted filament terms with a finite numerical core."""

    radius_squared = jnp.sum(displacement * displacement, axis=-1)
    radius_floor = jnp.asarray(1.0e-30, dtype=radius_squared.dtype)
    outside_core = radius_squared > radius_floor
    safe_radius_squared = jnp.where(
        outside_core,
        radius_squared,
        jnp.ones_like(radius_squared),
    )
    inverse_radius = jax.lax.rsqrt(safe_radius_squared)
    unit_displacement = displacement * inverse_radius[..., None]
    weighted_cross = jnp.cross(gamma_dash, unit_displacement) * weights
    terms = weighted_cross * inverse_radius[..., None] * inverse_radius[..., None]
    return jnp.where(outside_core[..., None], terms, jnp.zeros_like(terms))


@_regularized_filament_terms.defjvp
def _regularized_filament_terms_jvp(primals, tangents):
    gamma_dash, displacement, weights = primals
    gamma_dash_tangent, displacement_tangent, weights_tangent = tangents

    radius_squared = jnp.sum(displacement * displacement, axis=-1)
    radius_floor = jnp.asarray(1.0e-30, dtype=radius_squared.dtype)
    outside_core = radius_squared > radius_floor
    safe_radius_squared = jnp.where(
        outside_core,
        radius_squared,
        jnp.ones_like(radius_squared),
    )
    inverse_radius = jax.lax.rsqrt(safe_radius_squared)
    unit_displacement = displacement * inverse_radius[..., None]
    radial_velocity = jnp.sum(
        unit_displacement * displacement_tangent,
        axis=-1,
    )
    unit_displacement_tangent = (
        displacement_tangent - unit_displacement * radial_velocity[..., None]
    ) * inverse_radius[..., None]
    inverse_radius_tangent = -inverse_radius * inverse_radius * radial_velocity

    cross = jnp.cross(gamma_dash, unit_displacement)
    cross_tangent = jnp.cross(
        gamma_dash_tangent,
        unit_displacement,
    ) + jnp.cross(gamma_dash, unit_displacement_tangent)
    weighted_cross = cross * weights
    weighted_cross_tangent = cross_tangent * weights + cross * weights_tangent
    inverse_radius_squared = inverse_radius * inverse_radius
    terms = weighted_cross * inverse_radius_squared[..., None]
    radial_term = (
        weighted_cross * (2 * inverse_radius)[..., None]
    ) * inverse_radius_tangent[..., None]
    terms_tangent = (
        weighted_cross_tangent * inverse_radius_squared[..., None]
        + radial_term
    )
    zero = jnp.zeros_like(terms)
    return (
        jnp.where(outside_core[..., None], terms, zero),
        jnp.where(
            outside_core[..., None],
            terms_tangent,
            jnp.zeros_like(terms_tangent),
        ),
    )


class _CartesianFieldOperators:
    """Cartesian field operations derived from a differentiable ``B`` method.

    ESSOS tracing and objective functions use this small method contract for
    fields expressed in Cartesian coordinates.  Concrete fields only need to
    provide ``B(points)``; JAX supplies the spatial derivatives below.
    """

    def sqrtg(self, points: Any) -> Any:
        """Return the Cartesian-coordinate Jacobian determinant."""

        del points
        return jnp.asarray(1.0)

    def B_covariant(self, points: Any) -> Any:
        """Return Cartesian covariant components of the magnetic field."""

        return self.B(points)

    def B_contravariant(self, points: Any) -> Any:
        """Return Cartesian contravariant components of the magnetic field."""

        return self.B(points)

    def AbsB(self, points: Any) -> Any:
        """Return the magnetic-field magnitude."""

        return jnp.linalg.norm(self.B(points), axis=-1)

    def dB_by_dX(self, points: Any) -> Any:
        """Return the Cartesian field Jacobian at one point."""

        xyz = jnp.asarray(points)
        if xyz.shape != (3,):
            raise ValueError(
                "spatial field derivatives require one point with shape (3,), "
                f"got {xyz.shape}"
            )
        return jacfwd(self.B)(xyz)

    def dAbsB_by_dX(self, points: Any) -> Any:
        """Return the Cartesian gradient of the field magnitude."""

        xyz = jnp.asarray(points)
        if xyz.shape != (3,):
            raise ValueError(
                "spatial field derivatives require one point with shape (3,), "
                f"got {xyz.shape}"
            )
        return grad(self.AbsB)(xyz)

    def grad_B_covariant(self, points: Any) -> Any:
        """Return the Cartesian covariant-field Jacobian at one point."""

        xyz = jnp.asarray(points)
        if xyz.shape != (3,):
            raise ValueError(
                "spatial field derivatives require one point with shape (3,), "
                f"got {xyz.shape}"
            )
        return jacfwd(self.B_covariant)(xyz)

    def curl_B(self, points: Any) -> Any:
        """Return the Cartesian curl of the magnetic field."""

        gradient = self.grad_B_covariant(points)
        return jnp.stack(
            (
                gradient[2, 1] - gradient[1, 2],
                gradient[0, 2] - gradient[2, 0],
                gradient[1, 0] - gradient[0, 1],
            )
        ) / self.sqrtg(points)

    def curl_b(self, points: Any) -> Any:
        """Return the Cartesian curl of the unit field vector."""

        magnitude = self.AbsB(points)
        return self.curl_B(points) / magnitude + jnp.cross(
            self.B_covariant(points), self.dAbsB_by_dX(points)
        ) / (magnitude * magnitude * self.sqrtg(points))

    def kappa(self, points: Any) -> Any:
        """Return magnetic-field-line curvature in Cartesian coordinates."""

        return (
            -jnp.cross(self.B_contravariant(points), self.curl_b(points))
            * self.sqrtg(points)
            / self.AbsB(points)
        )

    def to_xyz(self, points: Any) -> Any:
        """Return Cartesian coordinates unchanged."""

        return points


@dataclass(frozen=True, eq=False)
class FilamentaryBiotSavart(_CartesianFieldOperators):
    """Differentiable filamentary magnetic field.

    ``gamma`` and ``gamma_dash`` contain the points and periodic-parameter
    derivatives of every physical filament. ``currents`` contains the matching
    symmetry-expanded currents in amperes. All three arrays are JAX pytree
    leaves, so both current and coil-shape derivatives remain visible when the
    field crosses another JAX transformation boundary.
    """

    gamma: Any
    gamma_dash: Any
    currents: Any

    @classmethod
    def from_coils(cls, coils: Any) -> "FilamentaryBiotSavart":
        """Create a field from an ESSOS coil collection."""

        return cls(
            gamma=jnp.asarray(coils.gamma),
            gamma_dash=jnp.asarray(coils.gamma_dash),
            currents=jnp.asarray(coils.currents),
        )

    def B(self, points: Any) -> Any:
        """Return Cartesian ``(B_x, B_y, B_z)`` at points of shape ``(..., 3)``.

        Samples with squared distance at or below ``1e-30 m^2`` contribute
        zero. This numerical convention keeps transformed evaluations finite;
        values inside that core should not be interpreted as the singular
        physical filament field.
        """

        xyz = jnp.asarray(points)
        gamma = jnp.asarray(self.gamma)
        gamma_dash = jnp.asarray(self.gamma_dash)
        currents = jnp.asarray(self.currents)
        if xyz.ndim < 1 or xyz.shape[-1] != 3:
            raise ValueError(f"points must have shape (..., 3), got {xyz.shape}")
        if gamma.ndim != 3 or gamma.shape[-1] != 3:
            raise ValueError(
                f"gamma must have shape (n_coils, n_segments, 3), got {gamma.shape}"
            )
        if gamma_dash.shape != gamma.shape:
            raise ValueError(
                "gamma_dash must have the same shape as gamma, got "
                f"{gamma_dash.shape} and {gamma.shape}"
            )
        if currents.shape != (gamma.shape[0],):
            raise ValueError(
                "currents must have one entry per physical coil, got "
                f"{currents.shape} for {gamma.shape[0]} coils"
            )

        displacement = xyz[..., None, None, :] - gamma
        point_ndim = xyz.ndim - 1
        current_shape = (1,) * point_ndim + (-1, 1, 1)
        weights = jnp.reshape(currents, current_shape) * (1.0e-7 / gamma.shape[-2])
        terms = _regularized_filament_terms(
            gamma_dash,
            displacement,
            weights,
        )
        return jnp.sum(terms, axis=(-3, -2))

    def __call__(self, points: Any) -> Any:
        """Alias for :meth:`B`."""

        return self.B(points)

    def b_cyl(self, r: Any, phi: Any, z: Any) -> tuple[Any, Any, Any]:
        """Return cylindrical ``(B_R, B_phi, B_Z)`` at broadcastable points."""

        rr, pp, zz = jnp.broadcast_arrays(
            jnp.asarray(r),
            jnp.asarray(phi),
            jnp.asarray(z),
        )
        cos_phi = jnp.cos(pp)
        sin_phi = jnp.sin(pp)
        xyz = jnp.stack((rr * cos_phi, rr * sin_phi, zz), axis=-1)
        bxyz = self.B(xyz)
        br = cos_phi * bxyz[..., 0] + sin_phi * bxyz[..., 1]
        bphi = -sin_phi * bxyz[..., 0] + cos_phi * bxyz[..., 1]
        return br, bphi, bxyz[..., 2]


tree_util.register_dataclass(
    FilamentaryBiotSavart,
    data_fields=["gamma", "gamma_dash", "currents"],
    meta_fields=[],
)


class BiotSavart(MagneticField):
    def __init__(self, coils):
        self.coils = coils
        self._r_axis = None
        self._z_axis = None
    
    @property
    def dofs(self):
        return self.coils.dofs
    
    @dofs.setter
    def dofs(self, new_dofs):
        self.coils.dofs = new_dofs

    @jit
    def sqrtg(self, points):
        return 1.
    
    @jit
    def B(self, points):
        dif_R = (jnp.array(points) - self.coils.gamma).T
        dB = jnp.cross(self.coils.gamma_dash.T, dif_R, axisa=0, axisb=0, axisc=0) / jnp.linalg.norm(dif_R, axis=0)**3
        dB_sum = jnp.einsum("i,bai", self.coils.currents*1e-7, dB, optimize="greedy")
        return jnp.mean(dB_sum, axis=0)

    @jit
    def b_cyl(self, R, phi, Z):
        """Return ``(B_R, B_phi, B_Z)`` on broadcast cylindrical arrays.

        This field-provider interface lets VMEC/NESTOR evaluate ESSOS coils
        directly on a changing plasma boundary without writing an mgrid file.
        It uses the same traceable Biot--Savart graph as :meth:`B`, so coil
        shape and current derivatives are retained.
        """
        R, phi, Z = jnp.broadcast_arrays(R, phi, Z)
        xyz = jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1)
        B = vmap(self.B)(xyz.reshape((-1, 3))).reshape(xyz.shape)
        br = B[..., 0] * jnp.cos(phi) + B[..., 1] * jnp.sin(phi)
        bp = -B[..., 0] * jnp.sin(phi) + B[..., 1] * jnp.cos(phi)
        return br, bp, B[..., 2]

    @property
    def r_axis(self):
        if self._r_axis is None:
            self._r_axis = jnp.mean(jnp.sqrt(vmap(lambda dofs: dofs[0, 0]**2 + dofs[1, 0]**2)(self.coils.dofs_curves)))
        return self._r_axis

    @property
    def z_axis(self):
        if self._z_axis is None:
            self._z_axis = jnp.mean(vmap(lambda dofs: dofs[2, 0])(self.coils.dofs_curves))
        return self._z_axis    

    @jit
    def to_xyz(self, points):
        return points
    
    def _tree_flatten(self):
        children = (self.coils,)
        aux_data = {}
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

tree_util.register_pytree_node(BiotSavart,
                               BiotSavart._tree_flatten,
                               BiotSavart._tree_unflatten)
    
@jit
def d_dtheta_fft(f_theta):
    ntheta = f_theta.shape[-1]
    k = jnp.fft.fftfreq(ntheta, d=1.0/ntheta)     # integer modes
    Fk = jnp.fft.fft(f_theta, axis=-1)
    dF = (1j * k) * Fk
    return jnp.fft.ifft(dF, axis=-1).real * (2*jnp.pi)

@jit
def d2_dtheta2_fft(f_theta):
    ntheta = f_theta.shape[-1]
    k = jnp.fft.fftfreq(ntheta, d=1.0/ntheta)     # integer modes
    Fk = jnp.fft.fft(f_theta, axis=-1)
    d2F = -(k**2) * Fk
    return jnp.fft.ifft(d2F, axis=-1).real * (2*jnp.pi)**2

@jit
def gamma_dash_from_gamma(gamma):
    return jnp.stack([
        d_dtheta_fft(gamma[..., 0]),
        d_dtheta_fft(gamma[..., 1]),
        d_dtheta_fft(gamma[..., 2]),
    ], axis=-1)

@jit
def gamma_dashdash_from_gamma(gamma):
    return jnp.stack([
        d2_dtheta2_fft(gamma[..., 0]),
        d2_dtheta2_fft(gamma[..., 1]),
        d2_dtheta2_fft(gamma[..., 2]),
    ], axis=-1)

class BiotSavart_from_gamma(MagneticField):
    def __init__(self, gamma, gamma_dash=None, gamma_dashdash=None, currents=None):
        self.currents = currents
        self.gamma = gamma
        self._gamma_dash = gamma_dash
        self._gamma_dashdash = gamma_dashdash

        self.coils_length = None
        self.coils_curvature = None
        self.r_axis = None
        self.z_axis = None

    @property
    def gamma_dash(self):
        if self._gamma_dash is None:
            self._gamma_dash = gamma_dash_from_gamma(self.gamma)
        return self._gamma_dash

    @property
    def gamma_dashdash(self):
        if self._gamma_dashdash is None:
            self._gamma_dashdash = gamma_dashdash_from_gamma(self.gamma)
        return self._gamma_dashdash

    @property
    def coils_length(self):
        if self.coils_length is None:
            self.coils_length = jnp.array([jnp.mean(jnp.linalg.norm(d1gamma, axis=1)) for d1gamma in self.gamma_dash])
        return self.coils_length

    @property
    def coils_curvature(self):
        if self._coils_curvature is None:
            self._coils_curvature = vmap(Curves.compute_curvature)(self.gamma_dash, self.gamma_dashdash)
        return self._coils_curvature
    
    @property
    def r_axis(self):
        if self._r_axis is None:
            self._r_axis = jnp.average(jnp.linalg.norm(jnp.average(self.gamma, axis=1)[:, 0:2], axis=1))
        return self._r_axis
    
    @property
    def z_axis(self):
        if self._z_axis is None:
            self._z_axis = jnp.average(jnp.average(self.gamma, axis=1)[:, 2])
        return self._z_axis
    
    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        return 1.
    
    @partial(jit, static_argnames=['self'])
    def B(self, points):
        dif_R = (jnp.array(points) - self.gamma).T
        dB = jnp.cross(self.gamma_dash.T, dif_R, axisa=0, axisb=0, axisc=0) / jnp.linalg.norm(dif_R, axis=0)**3
        dB_sum = jnp.einsum("i,bai", self.currents*1e-7, dB, optimize="greedy")
        return jnp.mean(dB_sum, axis=0)
    
    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        return points

class Vmec():
    def __init__(self, wout_filename, ntheta=50, nphi=50, close=True, range_torus='full torus'):
        self.wout_filename = wout_filename
        from netCDF4 import Dataset
        self.nc = Dataset(self.wout_filename)
        self.nfp = int(self.nc.variables["nfp"][0])
        self.bmnc = jnp.array(self.nc.variables["bmnc"][:])
        self.xm = jnp.array(self.nc.variables["xm"][:])
        self.xn = jnp.array(self.nc.variables["xn"][:])
        self.rmnc = jnp.array(self.nc.variables["rmnc"][:])
        self.zmns = jnp.array(self.nc.variables["zmns"][:])
        self.bsubsmns = jnp.array(self.nc.variables["bsubsmns"][:])
        self.bsubumnc = jnp.array(self.nc.variables["bsubumnc"][:])
        self.bsubvmnc = jnp.array(self.nc.variables["bsubvmnc"][:])
        self.bsupumnc = jnp.array(self.nc.variables["bsupumnc"][:])
        self.bsupvmnc = jnp.array(self.nc.variables["bsupvmnc"][:])
        self.gmnc = jnp.array(self.nc.variables["gmnc"][:])
        self.xm_nyq = jnp.array(self.nc.variables["xm_nyq"][:])
        self.xn_nyq = jnp.array(self.nc.variables["xn_nyq"][:])
        self.len_xm_nyq = len(self.xm_nyq)
        self.ns = self.nc.variables["ns"][0]
        self.s_full_grid = jnp.linspace(0, 1, self.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds
        self.r_axis = self.rmnc[0, 0]
        self.z_axis=self.zmns[0,0]
        self.mpol = int(jnp.max(self.xm))
        self.ntor = int(jnp.max(jnp.abs(self.xn)) / self.nfp)
        self.range_torus = range_torus
        self._surface = SurfaceRZFourier.from_vmec(self, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus)
        self.Aminor_p = jnp.array(self.nc.variables["Aminor_p"][:])
        #self._classifier=SurfaceClassifier(self._surface,p=1,h=0.05)
        
    @property
    def surface(self):
        return self._surface
        
    @partial(jit, static_argnames=['self'])
    def B_covariant(self, points):
        s, theta, phi = points
        bsubsmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.bsubsmns)
        bsubumnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsubumnc[1:])
        bsubvmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsubvmnc[1:])
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        sinangle_nyq = jnp.sin(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sub_s = jnp.dot(bsubsmns_interp, sinangle_nyq)
        B_sub_theta = jnp.dot(bsubumnc_interp, cosangle_nyq)
        B_sub_phi = jnp.dot(bsubvmnc_interp, cosangle_nyq)
        return jnp.array([B_sub_s, B_sub_theta, B_sub_phi])
    
    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, points):
        s, theta, phi = points
        bsupumnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsupumnc[1:])
        bsupvmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsupvmnc[1:])
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sup_theta = jnp.dot(bsupumnc_interp, cosangle_nyq)
        B_sup_phi = jnp.dot(bsupvmnc_interp, cosangle_nyq)
        return jnp.array([0*B_sup_theta, B_sup_theta, B_sup_phi])
 
    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        s, theta, phi = points
        gmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.gmnc[1:])
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        sqrt_g_vmec = jnp.dot(gmnc_interp, cosangle_nyq)
        return sqrt_g_vmec



    @partial(jit, static_argnames=['self'])
    def B(self, points):
        s, theta, phi = points
        gmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.gmnc[1:])
        rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
        zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)
        d_rmnc_d_s_interp = vmap(lambda row: grad(lambda s: jnp.interp(s, self.s_full_grid, row))(s), in_axes=1)(self.rmnc)
        d_zmns_d_s_interp = vmap(lambda row: grad(lambda s: jnp.interp(s, self.s_full_grid, row))(s), in_axes=1)(self.zmns)
        
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sub_s, B_sub_theta, B_sub_phi = self.B_covariant(points)
        sqrt_g_vmec = jnp.dot(gmnc_interp, cosangle_nyq)
        
        cosangle  = jnp.cos(self.xm * theta - self.xn * phi)
        sinangle  = jnp.sin(self.xm * theta - self.xn * phi)
        msinangle = self.xm * sinangle
        nsinangle = self.xn * sinangle
        mcosangle = self.xm * cosangle
        ncosangle = self.xn * cosangle
        
        sinphi = jnp.sin(phi)
        cosphi = jnp.cos(phi)
        
        R = jnp.dot(rmnc_interp, cosangle)
        d_R_d_theta = jnp.dot(rmnc_interp, -msinangle)
        d_R_d_phi   = jnp.dot(rmnc_interp, nsinangle)
        d_R_d_s     = jnp.dot(d_rmnc_d_s_interp, cosangle)
        
        d_X_d_theta = d_R_d_theta * cosphi
        d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
        d_X_d_s = d_R_d_s * cosphi

        d_Y_d_theta = d_R_d_theta * sinphi
        d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
        d_Y_d_s = d_R_d_s * sinphi
        
        d_Z_d_s = jnp.dot(d_zmns_d_s_interp, sinangle)
        d_Z_d_theta = jnp.dot(zmns_interp, mcosangle)
        d_Z_d_phi = jnp.dot(zmns_interp, -ncosangle)

        grad_s_X = (d_Y_d_theta * d_Z_d_phi - d_Z_d_theta * d_Y_d_phi) / sqrt_g_vmec
        grad_s_Y = (d_Z_d_theta * d_X_d_phi - d_X_d_theta * d_Z_d_phi) / sqrt_g_vmec
        grad_s_Z = (d_X_d_theta * d_Y_d_phi - d_Y_d_theta * d_X_d_phi) / sqrt_g_vmec

        grad_theta_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
        grad_theta_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
        grad_theta_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec

        grad_phi_X = (d_Y_d_s * d_Z_d_theta - d_Z_d_s * d_Y_d_theta) / sqrt_g_vmec
        grad_phi_Y = (d_Z_d_s * d_X_d_theta - d_X_d_s * d_Z_d_theta) / sqrt_g_vmec
        grad_phi_Z = (d_X_d_s * d_Y_d_theta - d_Y_d_s * d_X_d_theta) / sqrt_g_vmec
        
        return jnp.array([B_sub_s * grad_s_X + B_sub_theta * grad_theta_X + B_sub_phi * grad_phi_X,
                          B_sub_s * grad_s_Y + B_sub_theta * grad_theta_Y + B_sub_phi * grad_phi_Y,
                          B_sub_s * grad_s_Z + B_sub_theta * grad_theta_Z + B_sub_phi * grad_phi_Z])
        
    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        s, theta, phi = points
        bmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bmnc[1:, :])
        cos_values = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        return jnp.dot(bmnc_interp, cos_values)
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)


    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)
    
    @partial(jit, static_argnames=['self'])
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)    
 
    @partial(jit, static_argnames=['self'])
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] -grad_B_cov[1][2],
                          grad_B_cov[0][2] -grad_B_cov[2][0],
                          grad_B_cov[1][0] -grad_B_cov[0][1]])/self.sqrtg(points)
    
    
    @partial(jit, static_argnames=['self'])
    def curl_b(self, points):
        return self.curl_B(points)/self.AbsB(points)+jnp.cross(self.B_covariant(points),jnp.array(self.dAbsB_by_dX(points)))/self.AbsB(points)**2/self.sqrtg(points)

    @partial(jit, static_argnames=['self'])
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points),self.curl_b(points))*self.sqrtg(points)/self.AbsB(points)

    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        s, theta, phi = points
        rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
        zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)
        cosangle = jnp.cos(self.xm * theta - self.xn * phi)
        sinangle = jnp.sin(self.xm * theta - self.xn * phi)
        R = jnp.dot(rmnc_interp, cosangle)
        Z = jnp.dot(zmns_interp, sinangle)
        X = R * jnp.cos(phi)
        Y = R * jnp.sin(phi)
        return jnp.array([X, Y, Z])

class near_axis:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "The 'near_axis' class has been migrated to the standalone 'pyQSC_JAX' repository. "
            "Please run 'pip install git+https://github.com/uwplasma/pyQSC_JAX.git' "
            "and import it via 'from pyqsc_jax.near_axis import near_axis'."
        )


class CombinedField(MagneticField):
    """Sum of several magnetic fields, traced as one.

    The usual case is a coil field plus a plasma contribution: ``B`` and
    ``B_contravariant`` add over the fields, while the geometry helpers
    ``sqrtg`` and ``to_xyz`` come from the first field, which is the one that
    carries the coordinate system.
    """

    def __init__(self, *fields):
        if len(fields) < 1:
            raise ValueError("CombinedField needs at least one field")
        self.fields = fields

    @jit
    def B(self, points):
        return sum(field.B(points) for field in self.fields)

    @jit
    def B_contravariant(self, points):
        return sum(field.B_contravariant(points) for field in self.fields)

    @jit
    def sqrtg(self, points):
        return self.fields[0].sqrtg(points)

    @jit
    def to_xyz(self, points):
        return self.fields[0].to_xyz(points)

    def _tree_flatten(self):
        return (self.fields,), {}

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[0], **aux_data)


tree_util.register_pytree_node(CombinedField,
                               CombinedField._tree_flatten,
                               CombinedField._tree_unflatten)

