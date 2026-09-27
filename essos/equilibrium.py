"""Equilibrium field responses for finite-beta topology optimization.

The implicit equilibrium/field contract, VMEX exterior (coil plus
virtual-casing) fields with their LCFS-exterior domain, and the VMEC-JAX
material-coordinate field response.
"""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass,replace
from pathlib import Path
import hashlib


# ----------------------------------------------------------------------------
# From essos/equilibrium_response.py: Equilibrium-aware field response with explicit primal and linear-solve gates.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EquilibriumDefinition:
    identity: str
    boundary_model: str
    controls: tuple[str,...]
    held_fixed: tuple[str,...]
    spatial_domain: str
    field_mode: str = 'self_consistent_equilibrium'

    def __post_init__(self):
        if self.field_mode not in ('vacuum_coils','prescribed_background','self_consistent_equilibrium'):raise ValueError('Unknown field mode')
        if self.boundary_model not in ('fixed_boundary','free_boundary','manufactured'):raise ValueError('Unknown equilibrium boundary model')
        if not self.identity or not self.spatial_domain or not self.controls:raise ValueError('Equilibrium model, domain, and controls must be explicit')
        object.__setattr__(self,'controls',tuple(self.controls));object.__setattr__(self,'held_fixed',tuple(self.held_fixed))


@dataclass(frozen=True)
class EquilibriumState:
    definition: EquilibriumDefinition
    parameters: np.ndarray
    state: np.ndarray
    residual_norm: float
    condition: float
    minimum_singular_value: float
    valid: bool
    status: str

    def __post_init__(self):
        for name in ('parameters','state'):
            a=np.asarray(getattr(self,name),dtype=float)
            object.__setattr__(self,name,np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape))


class ImplicitEquilibriumField:
    """Reference total-response adapter around a qualified primal solver.

    solve_fn(c, initial) returns the equilibrium unknown vector. residual_fn(z,c)
    is the *gauged physical equilibrium residual*, with declared normalization.
    field_fn(x,z,c) returns Cartesian B at a fixed physical position x. Supplying
    field values at moving material grid points violates this contract; those
    require the coordinate response as well. This class does not infer gauges,
    plasma profiles, or exterior coupling from a solver's name.
    """
    def __init__(self,definition,solve_fn,residual_fn,field_fn,*,residual_tolerance=1e-9,
                 maximum_condition=1e10,minimum_singular_value=1e-10):
        controls=np.array([residual_tolerance,maximum_condition,minimum_singular_value])
        if not np.all(np.isfinite(controls)) or np.any(controls<=0):raise ValueError('Invalid equilibrium response controls')
        self.definition=definition;self.solve_fn=solve_fn;self.residual_fn=residual_fn;self.field_fn=field_fn
        self.residual_tolerance=residual_tolerance;self.maximum_condition=maximum_condition;self.minimum_singular_value=minimum_singular_value
        self._linearization=jax.jit(jax.jacfwd(residual_fn,0))

    def solve(self,parameters,initial=None):
        c=np.asarray(parameters,dtype=float)
        if c.shape!=(len(self.definition.controls),) or not np.all(np.isfinite(c)):raise ValueError('Invalid equilibrium controls')
        z=np.asarray(self.solve_fn(c.copy(),None if initial is None else np.array(initial,copy=True)),dtype=float)
        r=np.asarray(self.residual_fn(z,c));a=np.asarray(self._linearization(z,c))
        if z.ndim!=1 or r.shape!=z.shape or a.shape!=(z.size,z.size):raise ValueError('Equilibrium residual must be a square gauged system')
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(a)):
            return EquilibriumState(self.definition,c,z,float('inf'),float('inf'),0.,False,'equilibrium_nonfinite')
        norm=float(np.linalg.norm(r,np.inf));sv=np.linalg.svd(a,compute_uv=False)
        condition=float(sv[0]/sv[-1]) if sv[-1]>0 else float('inf')
        if norm>self.residual_tolerance:status='equilibrium_not_converged'
        elif condition>self.maximum_condition or sv[-1]<self.minimum_singular_value:status='equilibrium_response_singular'
        else:status='valid'
        return EquilibriumState(self.definition,c,z,norm,condition,float(sv[-1]),status=='valid',status)

    def _check(self,solution):
        if solution.definition!=self.definition or not solution.valid:raise ValueError('Invalid equilibrium response state: '+solution.status)
        error=np.linalg.norm(self.residual_fn(solution.state,solution.parameters),np.inf)
        if not np.isfinite(error) or error>self.residual_tolerance:raise ValueError('Equilibrium state no longer solves the declared residual')

    def field(self,solution,xyz):
        self._check(solution)
        return self.field_fn(jnp.asarray(xyz),jnp.asarray(solution.state),jnp.asarray(solution.parameters))

    def state_jvp(self,solution,parameter_tangent):
        self._check(solution)
        z=jnp.asarray(solution.state);c=jnp.asarray(solution.parameters);dc=jnp.asarray(parameter_tangent)
        rhs=jax.jvp(lambda p:self.residual_fn(z,p),(c,),(dc,))[1]
        a=np.asarray(self._linearization(z,c))
        return np.linalg.solve(a,-np.asarray(rhs))

    def jvp(self,solution,xyz,parameter_tangent):
        dz=self.state_jvp(solution,parameter_tangent)
        z=jnp.asarray(solution.state);c=jnp.asarray(solution.parameters)
        return jax.jvp(lambda state,p:self.field_fn(jnp.asarray(xyz),state,p),
                       (z,c),(jnp.asarray(dz),jnp.asarray(parameter_tangent)))[1]

    def vjp(self,solution,xyz,field_cotangent):
        self._check(solution)
        z=jnp.asarray(solution.state);c=jnp.asarray(solution.parameters)
        _,pullback=jax.vjp(lambda state,p:self.field_fn(jnp.asarray(xyz),state,p),z,c)
        bz,bc=pullback(jnp.asarray(field_cotangent))
        a=np.asarray(self._linearization(z,c));adjoint=np.linalg.solve(a.T,np.asarray(bz))
        _,residual_pullback=jax.vjp(lambda p:self.residual_fn(z,p),c)
        return np.asarray(bc)-np.asarray(residual_pullback(jnp.asarray(adjoint))[0])


# ----------------------------------------------------------------------------
# From essos/vmex_domain.py: Exterior classification for a simple Fourier LCFS in cylindrical sections.
# ----------------------------------------------------------------------------

class FourierLCFSExteriorDomain:
    """Reject interior points and unresolved boundary proximity.

    Modes use the VMEC wout convention ``m*theta - xn*phi`` (xn includes nfp).
    Every constant-phi boundary must be a simple closed RZ curve. The polygon's
    chord error is bounded by max|d²(R,Z)/dtheta²| * dtheta² / 8. Clearance here
    is in a cylindrical section, NOT shortest three-dimensional wall distance.
    """
    def __init__(self, xm, xn, rmnc, zmns, *, rmns=None, zmnc=None,
                 ntheta=1024, minimum_section_clearance=0.):
        arrays = [np.asarray(a, float) for a in (xm,xn,rmnc,zmns)]
        if (arrays[0].ndim != 1 or not len(arrays[0])
                or any(a.shape != arrays[0].shape or not np.all(np.isfinite(a)) for a in arrays)):
            raise ValueError('Expected matching finite Fourier mode vectors')
        if np.any(arrays[0] != np.round(arrays[0])) or np.any(arrays[1] != np.round(arrays[1])):
            raise ValueError('Fourier modes must be integers for a closed periodic LCFS')
        if isinstance(ntheta, bool) or int(ntheta) != ntheta or ntheta < 16:
            raise ValueError('ntheta must be an integer >= 16')
        if not np.isfinite(minimum_section_clearance) or minimum_section_clearance < 0:
            raise ValueError('Invalid minimum section clearance')
        for value in (rmns,zmnc):
            a = np.zeros_like(arrays[0]) if value is None else np.asarray(value,float)
            if a.shape != arrays[0].shape or not np.all(np.isfinite(a)):
                raise ValueError('Invalid asymmetric Fourier coefficients')
            arrays.append(a)
        self.xm,self.xn,self.rmnc,self.zmns,self.rmns,self.zmnc = (
            np.frombuffer(a.tobytes(),dtype=float) for a in arrays)
        self.ntheta = int(ntheta)
        self.minimum_section_clearance = float(minimum_section_clearance)
        r2 = np.sum(self.xm**2*(abs(self.rmnc)+abs(self.rmns)))
        z2 = np.sum(self.xm**2*(abs(self.zmnc)+abs(self.zmns)))
        self.chord_error_bound = float(np.hypot(r2,z2)*(2*np.pi/self.ntheta)**2/8)

    @classmethod
    def from_wout(cls, wout, **kwargs):
        asym = bool(wout.lasym)
        return cls(wout.xm,wout.xn,np.asarray(wout.rmnc)[-1],np.asarray(wout.zmns)[-1],
            rmns=np.asarray(wout.rmns)[-1] if asym else None,
            zmnc=np.asarray(wout.zmnc)[-1] if asym else None,**kwargs)

    def section(self, phi):
        theta = np.arange(self.ntheta)*2*np.pi/self.ntheta
        phase = theta[:,None]*self.xm-self.xn*phi
        cs,sn = np.cos(phase),np.sin(phase)
        return np.column_stack((cs@self.rmnc+sn@self.rmns,cs@self.zmnc+sn@self.zmns))

    def classify(self, xyz):
        points = np.asarray(xyz,float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError('Expected finite Cartesian points (n,3)')
        outside = np.zeros(len(points),bool)
        clearance = np.empty(len(points))
        for i,(x,y,z) in enumerate(points):
            r = np.hypot(x,y)
            polygon = self.section(np.arctan2(y,x))
            if np.min(polygon[:,0]) <= self.chord_error_bound:
                raise ValueError('Boundary section approaches the cylindrical coordinate axis')
            a,b = polygon,np.roll(polygon,-1,axis=0)
            edge = b-a
            lengths2 = np.sum(edge*edge,axis=1)
            if np.any(lengths2 == 0):raise ValueError('Degenerate boundary section')
            t = np.clip(np.sum((np.array([r,z])-a)*edge,axis=1)/lengths2,0,1)
            distance = np.min(np.linalg.norm(np.array([r,z])-a-t[:,None]*edge,axis=1))
            # Ray crossing uses only straddling edges, avoiding horizontal-edge division.
            crossing = (a[:,1]>z) != (b[:,1]>z)
            aa,bb = a[crossing],b[crossing]
            intersections = aa[:,0]+(z-aa[:,1])*(bb[:,0]-aa[:,0])/(bb[:,1]-aa[:,1])
            interior = np.count_nonzero(intersections>r)%2 == 1
            clearance[i] = (-distance if interior else distance)-self.chord_error_bound
            outside[i] = r>0 and clearance[i]>self.minimum_section_clearance
        return outside,clearance

    def __call__(self, xyz):
        return self.classify(xyz)[0]


# ----------------------------------------------------------------------------
# From essos/vmex_field.py: Validated VMEX exterior-field snapshots for pyna/Cyna production tracing.
# ----------------------------------------------------------------------------

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
        # VMEX's eager estimator creates lax.cond closures on each call. A
        # stable outer jit prevents per-batch recompilation during large grid
        # builds. Domain/finite/error-budget checks remain eager in sample().
        # These callables belong to this one accepted equilibrium snapshot.
        self._sample_value = jax.jit(backend.B)
        self._sample_error = jax.jit(backend.B_error_estimate) if backend.uses_virtual_casing else None

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
            error = np.asarray(self._sample_error(jnp.asarray(points)))
            if error.shape != (len(points),) or not np.all(np.isfinite(error)) or np.any(error < 0):
                raise ValueError("VMEX returned an invalid field error estimate")
            maximum = float(np.max(error))
            if maximum > self.maximum_error:
                raise ValueError(f"Exterior quadrature error {maximum:.3e} exceeds {self.maximum_error:.3e}")
        else:
            maximum = 0.0
        value = np.asarray(self._sample_value(jnp.asarray(points)))
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
        from essos.topology import _sampling_axis, _positive_integer
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


# ----------------------------------------------------------------------------
# From essos/vmec_response.py: Optional VMEC-JAX response at fixed flux-grid labels.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class VmecMaterialEvaluation:
    field: np.ndarray
    jacobian: np.ndarray | None
    parameters: np.ndarray
    converged: bool
    force_residuals: tuple[float,float,float]
    coordinate_convention: str = 'fixed_vmec_flux_grid_labels'

    def __post_init__(self):
        from pyna.topo.snapshot import immutable_array
        for name in ('field','jacobian','parameters'):
            value=getattr(self,name)
            if value is not None:object.__setattr__(self,name,immutable_array(value))


class VmecJaxMaterialField:
    """Use actual VMEC solves and its discrete state-response implementation.

    The pinned solver's discrete response is treated as provisional until it
    agrees with independent complete equilibrium finite differences. Primal
    force residuals are checked separately from response qualification.
    """
    def __init__(self,input_path,workdir,*,ns=16,controls=(('rc',1,0),),
                 max_iterations=2500,force_tolerance=1e-13,s_index=None,response_method="discrete"):
        if response_method not in ("discrete","implicit_residual"):raise ValueError("Unknown VMEC response method")
        self.response_method=response_method
        import jax
        import jax.numpy as jnp
        import vmec_jax as vj
        from vmec_jax.optimization import BoundaryParamSpec,FixedBoundaryExactOptimizer
        from vmec_jax.field import b_cartesian_from_state
        if not jax.config.x64_enabled:raise ValueError('VMEC response requires float64')
        self.input_path=Path(input_path);self.workdir=Path(workdir);self.workdir.mkdir(parents=True,exist_ok=True)
        self.ns=ns;self.max_iterations=max_iterations;self.force_tolerance=force_tolerance
        cfg,indata=vj.load_config(self.input_path);cfg=replace(cfg,ns=ns)
        self.static=vj.build_static(cfg);self.indata=indata
        self.s_index=ns//2 if s_index is None else s_index
        boundary=vj.boundary_from_indata(indata,self.static.modes,apply_m1_constraint=False)
        boundary_input=vj.boundary_input_from_indata(indata,self.static.modes)
        self.boundary_input=boundary_input
        specs=[]
        for kind,m,n in controls:
            indices=np.flatnonzero((np.asarray(self.static.modes.m)==m)&(np.asarray(self.static.modes.n)==n))
            if len(indices)!=1:raise ValueError('Requested boundary Fourier mode is unavailable')
            specs.append(BoundaryParamSpec(f'{kind}({m},{n})',kind,int(indices[0]),m,n))
        # The field helper infers no changing profiles: all input profiles are
        # held fixed; only the selected physical boundary modes vary.
        initial=vj.run_fixed_boundary(self.input_path,ns_override=ns,max_iter=max_iterations,multigrid=False,verbose=False)
        self.signgs=initial.signgs
        self._field=lambda state:b_cartesian_from_state(state,self.static,indata=self.indata,signgs=self.signgs,s_index=self.s_index)
        self.optimizer=FixedBoundaryExactOptimizer(self.static,indata,boundary,specs,
                         lambda state:jnp.ravel(self._field(state)),boundary_input=boundary_input,
                         inner_max_iter=max_iterations,inner_ftol=force_tolerance,
                         trial_max_iter=max_iterations,trial_ftol=force_tolerance,exact_path='tape')
        self.specs=tuple(specs)
        self.parameter_names=tuple(spec.name for spec in specs)

    def solve_primal(self,parameters):
        import vmec_jax as vj
        parameters=np.asarray(parameters,dtype=float)
        if parameters.shape!=(len(self.parameter_names),) or not np.all(np.isfinite(parameters)):raise ValueError('Invalid VMEC boundary parameters')
        tag=hashlib.sha256(parameters.tobytes()).hexdigest()[:20]
        path=self.workdir/f'input-{tag}.vmec'
        self.optimizer.save_input(path,parameters)
        from vmec_jax.namelist import InData,write_indata
        _,candidate_indata=vj.load_config(path)
        scalars=dict(candidate_indata.scalars)
        scalars['FTOL_ARRAY']=[self.force_tolerance]
        scalars['NS_ARRAY']=[self.ns]
        write_indata(path,InData(scalars=scalars,indexed=dict(candidate_indata.indexed)))
        run=vj.run_fixed_boundary(path,ns_override=self.ns,max_iter=self.max_iterations,multigrid=False,verbose=False)
        self._last_primal_run=run
        residuals=tuple(float(np.asarray(getattr(run.result,k))[-1]) for k in ('fsqr2_history','fsqz2_history','fsql2_history'))
        valid=bool(run.result.diagnostics.get('converged',False) and np.all(np.isfinite(residuals)) and max(residuals)<=self.force_tolerance)
        return VmecMaterialEvaluation(np.asarray(self._field(run.state)),None,parameters,valid,residuals)

    def response(self,parameters):
        primal=self.solve_primal(parameters)
        if not primal.converged:raise ValueError('VMEC primal equilibrium did not converge')
        if self.response_method=="discrete":
            field,columns=self.optimizer.b_cartesian_tangent_columns_fun(parameters,s_index=self.s_index)
        else:
            field,columns=self._implicit_response(parameters)
        if not np.all(np.isfinite(field)) or not np.all(np.isfinite(columns)):raise ValueError('VMEC response is nonfinite')
        discrepancy=np.linalg.norm(field-primal.field)/max(np.linalg.norm(primal.field),1e-30)
        if discrepancy>1e-5:raise ValueError(f'VMEC response base state differs from independently solved equilibrium: {discrepancy}')
        return VmecMaterialEvaluation(np.asarray(field),np.asarray(columns),np.asarray(parameters),True,primal.force_residuals)

    def _implicit_response(self,parameters):
        import jax
        import jax.numpy as jnp
        import vmec_jax as vj
        from vmec_jax.boundary import boundary_from_input_convention
        from vmec_jax.implicit import ImplicitFixedBoundaryOptions,solve_fixed_boundary_state_implicit_vmec_residual
        state0=self._last_primal_run.state
        def edges(c):
            raw=vj.apply_boundary_params(self.boundary_input,self.specs,c)
            boundary=boundary_from_input_convention(raw,self.static.modes,lasym=bool(self.static.cfg.lasym),apply_m1_constraint=False)
            return boundary.R_cos,boundary.R_sin,boundary.Z_cos,boundary.Z_sin
        # The implicit backend constructs an initial guess from BoundaryCoeffs;
        # its edge arguments are Fourier boundary coefficients, not normalized
        # rows of the solver state (these differ for nonaxisymmetric modes).
        options=ImplicitFixedBoundaryOptions(damping=0.,cg_tol=1e-11,cg_max_iter=500,
                                            residual_tangent_mode='chunked',jac_chunk_size=16)
        def material(c):
            e=edges(c)
            state=solve_fixed_boundary_state_implicit_vmec_residual(state0,self.static,indata=self.indata,
                       signgs=self.signgs,state0_host=state0,max_iter=self.max_iterations,
                       ftol=self.force_tolerance,step_size=self.indata.get_float('DELT',1.),implicit=options,
                       edge_Rcos=e[0],edge_Rsin=e[1],edge_Zcos=e[2],edge_Zsin=e[3])
            return self._field(state)
        c=jnp.asarray(parameters);columns=[];field=None
        for direction in np.eye(len(parameters)):
            field,column=jax.jvp(material,(c,),(jnp.asarray(direction),))
            columns.append(np.asarray(column))
        return np.asarray(field),np.stack(columns,axis=-1)

    @staticmethod
    def jvp(evaluation,direction):
        if not evaluation.converged or evaluation.jacobian is None:raise ValueError('Qualified primal and response required')
        return np.tensordot(evaluation.jacobian,np.asarray(direction),axes=1)

    @staticmethod
    def vjp(evaluation,cotangent):
        if not evaluation.converged or evaluation.jacobian is None:raise ValueError('Qualified primal and response required')
        return evaluation.jacobian.reshape(-1,evaluation.parameters.size).T@np.asarray(cotangent).ravel()
