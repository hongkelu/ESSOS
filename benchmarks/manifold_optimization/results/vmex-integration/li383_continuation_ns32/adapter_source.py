"""Validated VMEX exterior-field snapshots for pyna/Cyna production tracing.

VMEX owns equilibrium and virtual casing. The caller's domain predicate must
identify points strictly outside the LCFS and within the external-field domain.
"""
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class ExteriorValidation:
    points: int
    maximum_error_estimate: float
    error_normalization: str = "rms_surface_field"


class VmexExteriorField:
    """Host-side adapter for one explicitly accepted VMEX equilibrium.

    Parameter responses use the backend's public B_vjp in its declared DOF
    order. Use a parameterized equilibrium/surface backend to include plasma
    response; loading a fixed wout alone cannot provide it. Quadrature checks
    do not establish equilibrium/coil consistency or derivative accuracy.

    This is not a JAX PyTree. Stored-point queries mutate VMEX's query cache;
    do not share an instance between concurrent evaluations. An explicit
    control_jacobian maps physical controls into backend DOFs locally; e.g.
    stack two identity matrices to tie plasma and direct-coil parameter blocks.
    """
    def __init__(self, backend, *, definition, domain_check,
                 equilibrium_valid, maximum_error=1e-5, control_jacobian=None):
        from vmex.core.extender import VmecExtender
        if not isinstance(backend, VmecExtender):
            raise TypeError("A VMEX VmecExtender is required")
        if not equilibrium_valid:
            raise ValueError("Equilibrium acceptance is required before field sampling")
        if not callable(domain_check):
            raise TypeError("An exterior-domain predicate is required")
        if not np.isfinite(maximum_error) or maximum_error <= 0:
            raise ValueError("maximum_error must be finite and positive")
        if backend.uses_near_surface_continuation:
            raise ValueError("Near-surface continuation needs a separately qualified error budget")
        if definition.field_mode == "self_consistent_equilibrium" and not backend.uses_virtual_casing:
            raise ValueError("Self-consistent exterior field requires a plasma contribution")
        backend_controls = tuple(backend.dof_names)
        if control_jacobian is None:
            # Nonparameterized snapshots remain useful for field sampling.
            matrix = None
        else:
            matrix = np.asarray(control_jacobian, dtype=float)
            if matrix.shape != (len(backend_controls), len(definition.controls)) or not np.all(np.isfinite(matrix)):
                raise ValueError("Invalid backend-to-physical control Jacobian")
            matrix = np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)
        self.control_jacobian = matrix
        self.backend_controls = backend_controls
        self.controls = tuple(definition.controls)
        self.backend = backend
        self.definition = definition
        self.domain_check = domain_check
        self.maximum_error = float(maximum_error)

    def _points(self, xyz):
        points = np.asarray(xyz, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.all(np.isfinite(points)):
            raise ValueError("Expected finite, nonempty Cartesian points with shape (n, 3)")
        domain = np.asarray(self.domain_check(points))
        if domain.shape != (len(points),) or domain.dtype != np.bool_ or not np.all(domain):
            raise ValueError("Points leave the declared exterior-field domain")
        return points

    def sample(self, xyz):
        """Return B and an eager quadrature report, including after JIT use."""
        points = self._points(xyz)
        if self.backend.uses_virtual_casing:
            error = np.asarray(self.backend.B_error_estimate(jnp.asarray(points)))
            if error.shape != (len(points),) or not np.all(np.isfinite(error)) or np.any(error < 0):
                raise ValueError("VMEX returned an invalid field error estimate")
            maximum = float(np.max(error))
            if maximum > self.maximum_error:
                raise ValueError(f"Exterior quadrature error {maximum:.3e} exceeds {self.maximum_error:.3e}")
        else:
            maximum = 0.0
        value = np.asarray(self.backend.B(jnp.asarray(points)))
        if value.shape != points.shape or not np.all(np.isfinite(value)):
            raise ValueError("VMEX returned an invalid Cartesian field")
        return value, ExteriorValidation(len(points), maximum)

    def B(self, xyz):
        """Host-side Cartesian query accepting either (3,) or (n,3)."""
        single = np.shape(xyz) == (3,)
        values, _ = self.sample(np.asarray(xyz)[None] if single else xyz)
        return values[0] if single else values

    def _response(self, xyz):
        value, _ = self.sample(xyz)
        if tuple(self.backend.dof_names) != self.backend_controls or tuple(self.definition.controls) != self.controls:
            raise ValueError("VMEX response DOFs no longer match the declared controls")
        if self.control_jacobian is None and self.backend_controls != self.controls:
            raise ValueError("VMEX response DOFs do not match the declared controls")
        self.backend.set_points(jnp.asarray(xyz))
        # These queries stay host-side. Let the first real cotangent populate
        # VMEX's lazy caches; a zero-cotangent probe would pay another complete
        # coupled-equilibrium adjoint without contributing to the result.
        return jnp.zeros_like(jnp.asarray(value))

    def _backend_vjp(self, cotangent):
        result = np.asarray(self.backend.B_vjp(jnp.asarray(cotangent)))
        if result.shape != (len(self.backend_controls),) or not np.all(np.isfinite(result)):
            raise ValueError("Invalid VMEX equilibrium response")
        return result

    def vjp(self, xyz, cotangent):
        zero = self._response(xyz)
        weight = np.asarray(cotangent, dtype=float)
        if weight.shape != zero.shape or not np.all(np.isfinite(weight)):
            raise ValueError("Invalid Cartesian field cotangent")
        result = self._backend_vjp(weight)
        if self.control_jacobian is not None:
            result = self.control_jacobian.T @ result
        if not np.all(np.isfinite(result)):
            raise ValueError("VMEX response is nonfinite")
        return result

    def jvp(self, xyz, direction):
        """Apply a reduced reference JVP using one adjoint per field component.

        VMEX's equilibrium adjoint may contain host callbacks/iterative solves
        that JAX cannot transpose again. Public VJP queries avoid that failure.
        Cost scales with the number of query components; use this for reduced
        qualification, not large-grid gradient assembly.
        """
        zero = self._response(xyz)
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (len(self.definition.controls),) or not np.all(np.isfinite(direction)):
            raise ValueError("Invalid VMEX parameter direction")
        backend_direction = direction if self.control_jacobian is None else self.control_jacobian @ direction
        result = np.empty(zero.shape, dtype=float)
        for index in range(result.size):
            cotangent = np.zeros(zero.shape, dtype=float)
            cotangent.flat[index] = 1.
            result.flat[index] = np.dot(self._backend_vjp(cotangent), backend_direction)
        if not np.all(np.isfinite(result)):
            raise ValueError("VMEX response is nonfinite")
        return result

    def to_pyna_grid(self, R, Z, Phi, *, nfp=1, batch_size=256):
        """Validate every sample before handing an exterior-only grid to Cyna.

        Grid interpolation and trajectory error require separate refinement.
        Rectangular grids containing interior points are deliberately rejected.
        """
        from essos.manifold import _sampling_axis, _positive_integer
        from pyna.fields import VectorFieldCylind
        R = _sampling_axis(R, "R", minimum_size=2)
        Z = _sampling_axis(Z, "Z", minimum_size=2)
        Phi = _sampling_axis(Phi, "Phi", minimum_size=2)
        if np.any(R <= 0):
            raise ValueError("R must be positive")
        nfp = _positive_integer(nfp, "nfp")
        batch_size = _positive_integer(batch_size, "batch_size")
        r, z, phi = np.meshgrid(R, Z, Phi, indexing="ij")
        points = np.stack((r*np.cos(phi), r*np.sin(phi), z), axis=-1).reshape(-1, 3)
        values = np.empty_like(points)
        maximum = 0.0
        for start in range(0, len(points), batch_size):
            values[start:start+batch_size], report = self.sample(points[start:start+batch_size])
            maximum = max(maximum, report.maximum_error_estimate)
        bx, by, bz = values.T
        cs, sn = np.cos(phi.ravel()), np.sin(phi.ravel())
        grid = VectorFieldCylind(R, Z, Phi, BR=(bx*cs+by*sn).reshape(r.shape),
            BZ=bz.reshape(r.shape), BPhi=(-bx*sn+by*cs).reshape(r.shape),
            nfp=nfp, name="VMEX total exterior field", units="T")
        return grid, ExteriorValidation(len(points), maximum)
