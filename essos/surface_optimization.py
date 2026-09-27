"""Closed-surface objectives for the shared refreshed-merit optimizer."""
from dataclasses import dataclass
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
from essos.topology_optimizer import Evaluation


@dataclass(frozen=True)
class CircleObjective:
    """Fixed observables, targets, and scaling, with separately refreshed tori.

    observables(unknowns, parameters) must compute the same physical quantities
    in model and refresh calls. Map omega is an angle per declared map span;
    callers convert it to transform using that span, never a hardcoded period.
    The torus problem chooses the physical label (currently signed section
    area). It must remain fixed throughout one optimization problem.
    """
    problem: object
    observables: object
    target: tuple[float,...]
    scales: tuple[float,...]
    initial_omega: float
    identity: str

    def __post_init__(self):
        target=np.asarray(self.target);scales=np.asarray(self.scales)
        if not self.identity or target.ndim!=1 or scales.shape!=target.shape or not np.all(np.isfinite(target)) or not np.all(np.isfinite(scales)) or np.any(scales<=0):raise ValueError('Invalid surface objective definition')
        object.__setattr__(self,'target',tuple(map(float,target)));object.__setattr__(self,'scales',tuple(map(float,scales)))

    @property
    def fingerprint(self):
        data=repr((self.identity,self.target,self.scales,self.initial_omega,self.problem.area,self.problem.length_scale,
                   self.problem.tolerance,self.problem.dense_tolerance,self.problem.maximum_condition,self.problem.minimum_speed)).encode()+self.problem.reference.tobytes()
        return hashlib.sha256(data).hexdigest()

    def _evaluate(self,parameters,snapshot,*,derivatives):
        if snapshot and snapshot.get('definition')!=self.fingerprint:raise ValueError('Surface snapshot definition changed')
        initial=snapshot.get('unknowns') if snapshot else None
        omega=snapshot.get('omega',self.initial_omega) if snapshot else self.initial_omega
        solution=self.problem.solve(parameters,omega=omega,initial=initial)
        diagnostics=dict(circle_status=solution.status,dense_invariance_residual=solution.dense_residual,
                         root_condition=solution.condition,minimum_speed=solution.minimum_speed,
                         counterterm=solution.counterterm,minimum_divisor=solution.minimum_divisor)
        if not solution.valid:
            return Evaluation(np.zeros(len(self.target)),snapshot,valid=False,status=solution.status,diagnostics=diagnostics)
        def residual(z,c):return (jnp.asarray(self.observables(z,c))-jnp.asarray(self.target))/jnp.asarray(self.scales)
        r=np.asarray(residual(solution.unknowns,parameters));jacobian=None
        if derivatives:
            dz=np.column_stack([self.problem.jvp(solution,d) for d in np.eye(len(parameters))])
            jacobian=np.asarray(jax.jacfwd(residual,0)(solution.unknowns,parameters))@dz+np.asarray(jax.jacfwd(residual,1)(solution.unknowns,parameters))
        updated=dict(definition=self.fingerprint,unknowns=solution.unknowns.tolist(),omega=solution.omega,
                     parameters=np.asarray(parameters).tolist(),label='signed_section_area',area=self.problem.area)
        return Evaluation(r,updated,jacobian=jacobian,diagnostics=diagnostics)

    def model(self,parameters,snapshot):
        return self._evaluate(parameters,snapshot,derivatives=True)

    def refresh(self,parameters,snapshot):
        return self._evaluate(parameters,snapshot,derivatives=False)
