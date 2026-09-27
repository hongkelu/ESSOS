"""Equilibrium-aware field response with explicit primal and linear-solve gates.

The adapter describes E(z,c)=0 and B(x,z,c). Its backend may solve E using an
external, non-JAX code. Dense reference linearizations are provided here for
reduced problems; production backends can implement the same JVP/VJP contract
using their native linear/adjoint solvers. No frozen-background response is
silently promoted to an equilibrium response.
"""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp


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
