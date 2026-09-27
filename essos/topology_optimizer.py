"""Trust-region Gauss-Newton design optimization with refreshed acceptance.

Candidates are accepted only after an independent physical refresh; accepted
states are checkpointed atomically and identified by content fingerprints.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from pathlib import Path
import os
import tempfile
import operator
from scipy.optimize import minimize


# ----------------------------------------------------------------------------
# From essos/candidate_cache.py: Explicit content identities for physical evaluations and current-only grids.
# ----------------------------------------------------------------------------

def content_identity(*,currents,geometry,wall,numerics,topology,target):
    """Hash all semantic inputs; callers must provide actual content, not IDs."""
    digest=hashlib.sha256()
    def add(value):
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:raise TypeError('Object arrays cannot identify numerical caches')
            digest.update(b'array');digest.update(str((value.dtype.str,value.shape)).encode());digest.update(value.tobytes())
        elif isinstance(value,dict):
            digest.update(b'mapping');digest.update(str(len(value)).encode()+b':')
            if not all(isinstance(k,str) for k in value):raise TypeError('Cache mapping keys must be strings')
            for key in sorted(value):add(key);add(value[key])
        elif isinstance(value,(tuple,list)):
            digest.update(b'list');digest.update(str(len(value)).encode()+b':')
            for item in value:add(item)
        elif isinstance(value,np.generic):add(value.item())
        else:
            encoded=json.dumps(value,allow_nan=False,sort_keys=True).encode()
            digest.update(str(len(encoded)).encode()+b':'+encoded)
    for name,value in locals().copy().items():
        if name in ('currents','geometry','wall','numerics','topology','target'):add(name);add(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class CurrentGridBasis:
    """Fields per ampere on one fixed grid with fixed coil geometry.

    Coils may be symmetry-linked current groups. Grid coordinates, coil
    quadrature, units, and symmetry conventions belong in geometry_grid_key.
    Shape changes require rebuilding the basis; wall changes invalidate the
    candidate evaluation but do not alter magnetic field values on this grid.
    """
    fields_per_ampere: np.ndarray
    geometry_grid_key: str

    def __post_init__(self):
        a=np.asarray(self.fields_per_ampere,dtype=float)
        if not self.geometry_grid_key or a.ndim<3 or a.shape[-1]!=3 or not np.all(np.isfinite(a)):raise ValueError('Finite vector-field basis and identity required')
        object.__setattr__(self,'fields_per_ampere',np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape))

    def evaluate(self,currents,*,geometry_grid_key):
        if geometry_grid_key!=self.geometry_grid_key:raise ValueError('Coil geometry or sampling grid changed; rebuild current basis')
        currents=jnp.asarray(currents)
        if currents.shape!=(self.fields_per_ampere.shape[0],):raise ValueError('Current groups must match field basis')
        return jnp.tensordot(currents,jnp.asarray(self.fields_per_ampere),axes=1)


# ----------------------------------------------------------------------------
# From essos/topology_optimizer.py: Scaled constrained optimization with mandatory refreshed physical acceptance.
# ----------------------------------------------------------------------------

class _FrozenArray(np.ndarray):
    def __setattr__(self,name,value):
        if name in {"shape","strides","dtype","data"}:
            raise ValueError("Accepted numerical metadata is immutable")
        super().__setattr__(name,value)


def _array(value):
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape).view(_FrozenArray)


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _diagnostics(value):
    """Keep invalid numerical diagnostics JSON-safe without changing physics."""
    if isinstance(value,np.ndarray):return _diagnostics(value.tolist())
    if isinstance(value,np.generic):return _diagnostics(value.item())
    if isinstance(value,dict):return {k:_diagnostics(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_diagnostics(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    return value


@dataclass(frozen=True)
class DesignProblem:
    identity: str
    reference: np.ndarray
    scales: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    equality_tolerance: float = 1e-8

    def __post_init__(self):
        for name in ('reference','scales','lower','upper'):
            object.__setattr__(self,name,_array(getattr(self,name)))
        n=self.reference.size
        if not self.identity or n==0 or any(getattr(self,k).shape!=(n,) for k in ('reference','scales','lower','upper')):
            raise ValueError('Problem requires an identity and aligned parameter vectors')
        if not np.all(np.isfinite(self.reference)) or not np.all(np.isfinite(self.scales)) or np.any(self.scales<=0):
            raise ValueError('Reference and positive parameter scales must be finite')
        if np.any(np.isnan(self.lower)) or np.any(np.isnan(self.upper)) or np.any(self.lower>self.upper):
            raise ValueError('Invalid physical bounds')
        if not np.isfinite(self.equality_tolerance) or self.equality_tolerance<0:
            raise ValueError('Invalid equality tolerance')

    @property
    def fingerprint(self):
        digest=hashlib.sha256(self.identity.encode())
        for value in (self.reference,self.scales,self.lower,self.upper):
            digest.update(value.tobytes())
        digest.update(str(self.equality_tolerance).encode())
        return digest.hexdigest()

    def physical(self,q):
        return self.reference+self.scales*np.asarray(q)


@dataclass
class Evaluation:
    residual: np.ndarray
    snapshot: dict = field(default_factory=dict)
    valid: bool = True
    status: str = 'valid'
    jacobian: np.ndarray | None = None
    inequalities: np.ndarray = field(default_factory=lambda: np.empty(0))
    equalities: np.ndarray = field(default_factory=lambda: np.empty(0))
    inequality_jacobian: np.ndarray | None = None
    equality_jacobian: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def objective(self):
        r=np.asarray(self.residual,dtype=float)
        return float(0.5*np.dot(r,r))

    def feasible(self,problem):
        arrays=[np.asarray(a,dtype=float) for a in (self.residual,self.inequalities,self.equalities)]
        return bool(self.valid and all(a.ndim==1 and np.all(np.isfinite(a)) for a in arrays)
                    and np.all(arrays[1]<=0) and np.all(np.abs(arrays[2])<=problem.equality_tolerance))


@dataclass(frozen=True)
class AcceptedState:
    problem_id: str
    q: np.ndarray
    objective: float
    snapshot_json: str
    history_json: str = '[]'
    iteration: int = 0
    radius: float = 0.25
    controls_json: str = '{}'

    def __post_init__(self):
        object.__setattr__(self,'q',_array(self.q))
        if self.q.ndim!=1 or not np.all(np.isfinite(self.q)) or not np.isfinite(self.objective):
            raise ValueError('Accepted state must be finite')
        if not np.isfinite(self.radius) or self.radius<=0 or self.iteration<0:
            raise ValueError('Invalid optimizer checkpoint controls')
        _json(json.loads(self.snapshot_json)); _json(json.loads(self.history_json)); _json(json.loads(self.controls_json))

    @property
    def snapshot(self):
        return json.loads(self.snapshot_json)

    @property
    def history(self):
        return json.loads(self.history_json)

    def save(self,path):
        path=Path(path)
        payload={'schema_version':2,'problem_id':self.problem_id,'q':self.q.tolist(),
                 'objective':self.objective,'snapshot':self.snapshot,'history':self.history,
                 'iteration':self.iteration,'radius':self.radius,'controls':json.loads(self.controls_json)}
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',delete=False) as stream:
                temporary=stream.name
                stream.write(_json(payload)+'\n');stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            if temporary and os.path.exists(temporary):os.unlink(temporary)


    @classmethod
    def load(cls,path,*,problem_id):
        p=json.loads(Path(path).read_text())
        if p.get('schema_version')!=2 or p.get('problem_id')!=problem_id:
            raise ValueError('Checkpoint schema or problem identity mismatch')
        return cls(p['problem_id'],p['q'],p['objective'],_json(p['snapshot']),_json(p['history']),p['iteration'],p['radius'],_json(p['controls']))


@dataclass(frozen=True)
class OptimizationResult:
    state: AcceptedState
    status: str
    trials: tuple


def optimize_topology(problem, initial_parameters, model, refresh, *, snapshot=None,
                      restart=None, max_iterations=50, initial_radius=0.25,
                      minimum_radius=1e-8, gradient_tolerance=1e-8,
                      acceptance_ratio=0.1, numerical_floor=1e-12, checkpoint=None):
    """Trust-region Gauss-Newton with refreshed feasibility and merit checks.

    refresh(c, snapshot) must reconstruct and physically validate the candidate;
    valid=False rejects. model(c, snapshot) supplies fixed-definition residuals
    and physical-parameter Jacobians. Programming/backend exceptions propagate.
    Every callback receives a fresh snapshot and parameter copy. A restart is
    refreshed again; target/scaling changes require a new problem identity.
    """
    controls=_json(dict(minimum_radius=minimum_radius,gradient_tolerance=gradient_tolerance,
                        acceptance_ratio=acceptance_ratio,numerical_floor=numerical_floor))
    if not np.isfinite(initial_radius) or not (0<minimum_radius<=initial_radius) or not 0<acceptance_ratio<1:
        raise ValueError('Invalid trust-region controls')
    max_iterations=operator.index(max_iterations)
    if max_iterations<1 or numerical_floor<0 or gradient_tolerance<0:
        raise ValueError('Invalid stopping controls')
    parameters=_array(initial_parameters)
    if parameters.shape!=problem.reference.shape:
        raise ValueError('Initial parameter shape differs from problem')
    q=(parameters-problem.reference)/problem.scales
    if restart is not None:
        if restart.problem_id!=problem.fingerprint or restart.controls_json!=controls:
            raise ValueError('Restart problem identity differs')
        q=restart.q.copy(); snapshot=restart.snapshot
    c=problem.physical(q)
    if c.shape!=problem.reference.shape or not np.all(np.isfinite(c)) or np.any(c<problem.lower) or np.any(c>problem.upper):
        raise ValueError('Initial parameters are outside physical bounds')
    initial=refresh(c.copy(),json.loads(_json(snapshot or {})))
    if not initial.feasible(problem):
        raise ValueError('Initial design must pass refreshed physical feasibility')
    if restart is not None and abs(initial.objective-restart.objective)>max(numerical_floor,1e-10*(1+abs(restart.objective))):
        raise ValueError('Restart refresh changed the saved physical objective')
    state=AcceptedState(problem.fingerprint,q,initial.objective,_json(initial.snapshot),
                        restart.history_json if restart else '[]',restart.iteration if restart else 0,
                        restart.radius if restart else initial_radius,controls)
    radius=state.radius; trials=[]; previous_history=state.history

    def finish(status):
        final=AcceptedState(state.problem_id,state.q,state.objective,state.snapshot_json,
                            _json(previous_history+trials),state.iteration,radius,controls)
        if checkpoint:final.save(checkpoint)
        return OptimizationResult(final,status,tuple(json.loads(_json(trials))))
    for _ in range(max_iterations):
        if state.objective<=numerical_floor:
            return finish("numerical_floor_reached")
        current=model(problem.physical(state.q).copy(),state.snapshot)
        if not current.feasible(problem):
            return finish('local_model_invalid')
        r=np.asarray(current.residual,dtype=float)
        jac=np.asarray(current.jacobian,dtype=float)
        if jac.shape!=(r.size,state.q.size) or not np.all(np.isfinite(jac)):
            raise ValueError('Model must supply a finite residual Jacobian')
        jac=jac*problem.scales
        gradient=jac.T@r
        lo=(problem.lower-problem.reference)/problem.scales-state.q
        hi=(problem.upper-problem.reference)/problem.scales-state.q
        projected=np.clip(-gradient,lo,hi)
        if np.linalg.norm(projected,np.inf)<=gradient_tolerance and not len(current.inequalities) and not len(current.equalities):
            return finish('converged')
        constraints=[{'type':'ineq','fun':lambda p:radius**2-np.dot(p,p),
                      'jac':lambda p:-2*p}]
        for values,derivative,kind in [(current.inequalities,current.inequality_jacobian,'ineq'),
                                       (current.equalities,current.equality_jacobian,'eq')]:
            values=np.asarray(values,dtype=float)
            if not values.size: continue
            a=np.asarray(derivative,dtype=float)
            if a.shape!=(values.size,state.q.size) or not np.all(np.isfinite(a)):
                raise ValueError('Active constraints require finite physical Jacobians')
            a=a*problem.scales
            sign=-1 if kind=='ineq' else 1
            # Solve inequalities just inside the boundary so floating-point
            # SLSQP roundoff cannot violate strict physical acceptance.
            if kind=='ineq':values=values+1e-12*(1+np.abs(values))
            constraints.append({'type':kind,'fun':lambda p,a=a,b=values,s=sign:s*(b+a@p),
                                'jac':lambda p,a=a,s=sign:s*a})
        sub=minimize(lambda p:0.5*np.dot(r+jac@p,r+jac@p),np.zeros_like(state.q),
                     jac=lambda p:jac.T@(r+jac@p),bounds=list(zip(lo,hi)),constraints=constraints,
                     method='SLSQP',options={'ftol':1e-12,'maxiter':100})
        if not sub.success or not np.all(np.isfinite(sub.x)):
            trials.append({'status':'subproblem_failed','message':str(sub.message),'radius':radius})
            radius*=0.5
            if radius<minimum_radius: return finish('feasible_stagnation')
            continue
        step=sub.x
        predicted=current.objective-0.5*np.dot(r+jac@step,r+jac@step)
        if predicted<=numerical_floor:
            # Project the negative gradient onto the linearized feasible set,
            # without the shrinking trust region. A small trust step alone is
            # not evidence of constrained first-order stationarity.
            projected_step=minimize(lambda p:.5*np.dot(p+gradient,p+gradient),np.zeros_like(state.q),
                jac=lambda p:p+gradient,bounds=list(zip(lo,hi)),constraints=constraints[1:],
                method='SLSQP',options={'ftol':1e-14,'maxiter':200})
            if projected_step.success and np.all(np.isfinite(projected_step.x)) and np.linalg.norm(projected_step.x,np.inf)<=gradient_tolerance:
                return finish('constrained_stationarity')
            return finish('model_decrease_unresolved')
        candidate=problem.physical(state.q+step)
        if np.any(candidate<problem.lower) or np.any(candidate>problem.upper):
            radius*=0.5
            continue
        physical=refresh(candidate.copy(),state.snapshot)
        feasible=physical.feasible(problem)
        actual=state.objective-physical.objective if feasible else None
        ratio=actual/predicted if actual is not None else None
        accepted=bool(feasible and actual>numerical_floor and ratio>=acceptance_ratio)
        record={'iteration':state.iteration,'q':(state.q+step).tolist(),'accepted':accepted,
                'status':'accepted' if accepted else physical.status if not feasible else 'merit_rejected',
                'predicted_decrease':float(predicted),'actual_decrease':actual,'ratio':ratio,'radius':radius,
                'diagnostics':_diagnostics(physical.diagnostics)}
        _json(record)
        trials.append(record)
        if accepted:
            # Snapshot serialization and checkpoint must succeed before committing.
            next_radius=2*radius if ratio>0.75 and np.linalg.norm(step)>0.8*radius else radius
            next_state=AcceptedState(problem.fingerprint,state.q+step,physical.objective,
                                     _json(physical.snapshot),_json(previous_history+trials),state.iteration+1,
                                     next_radius,controls)
            if checkpoint: next_state.save(checkpoint)
            state=next_state
            radius=next_radius
        else:
            radius*=0.5
        if radius<minimum_radius:
            return finish('feasible_stagnation')
    return finish('evaluation_budget')
