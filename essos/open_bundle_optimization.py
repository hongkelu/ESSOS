"""General open-line objectives with independent production acceptance.

The local event model supplies derivatives. The production callback must return
pyna's OpenBundleValidation after first-hit and whole-leg validation. Observables
receive (hit_RZPhi, connection_lengths, weights, parameters) in both paths.
Hit correspondence uses Cartesian distance in metres and a separate unwrapped
phase check in radians; these quantities are never combined into one norm.
"""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from pyna.topo.open_lines import trace_open_bundle
from pyna.topo.open_validation import OpenBundleValidation
from essos.candidate_cache import content_identity
from essos.topology_optimizer import Evaluation


@dataclass(frozen=True)
class OpenBundleObjective:
    bundle: object
    field_fn: object
    wall_fn: object
    wall_parameters: object
    observables: object
    production_validate: object
    target: tuple
    scales: tuple
    phi_guesses: tuple
    identity: str
    phi_start: float = 0.
    maximum_phi_shift: float = .2
    maximum_hit_discrepancy: float = 1e-3
    n_steps: int = 256
    launch_fn: object = None
    weights_fn: object = None

    def __post_init__(self):
        for name in ('target','scales','phi_guesses'):
            a=np.asarray(getattr(self,name),dtype=float)
            if a.ndim!=1 or not np.all(np.isfinite(a)):raise ValueError('Finite objective vectors required')
            object.__setattr__(self,name,tuple(map(float,a)))
        if not self.identity or not self.target or len(self.scales)!=len(self.target) or min(self.scales)<=0:raise ValueError('Invalid fixed objective definition')
        if len(self.phi_guesses)!=len(self.bundle.labels):raise ValueError('One event guess per launch required')
        if self.bundle.launch_convention!='fixed_physical' and self.launch_fn is None:raise ValueError('Moving launches require a physical launch function')
        if self.bundle.weights_move and self.weights_fn is None:raise ValueError('Moving weights require a physical weight function')
        if not np.all(np.isfinite([self.phi_start,self.maximum_phi_shift,self.maximum_hit_discrepancy])) or min(self.maximum_phi_shift,self.maximum_hit_discrepancy)<=0 or not isinstance(self.n_steps,int) or self.n_steps<2:raise ValueError('Invalid trace controls')

    @property
    def fingerprint(self):
        # identity versions callable field, wall, launch and observable definitions.
        return content_identity(currents=None,geometry=None,wall=None,numerics=None,topology=None,target=dict(coordinate_contract='cartesian_hits_unwrapped_phi_v1',identity=self.identity,labels=self.bundle.labels,launches=self.bundle.launch_RZ,
            weights=self.bundle.weights,convention=self.bundle.launch_convention,weights_move=self.bundle.weights_move,
            target=self.target,scales=self.scales,guesses=self.phi_guesses,phi_start=self.phi_start,
            shift=self.maximum_phi_shift,discrepancy=self.maximum_hit_discrepancy,n_steps=self.n_steps))

    def _check_snapshot(self,snapshot):
        if snapshot and snapshot.get('definition')!=self.fingerprint:raise ValueError('Open-bundle snapshot definition changed')

    def _weights(self,c):
        weights=jnp.asarray(self.bundle.weights) if self.weights_fn is None else self.weights_fn(c)
        return weights if self.bundle.weights_move else jax.lax.stop_gradient(weights)

    def _trace(self,c,snapshot):
        return trace_open_bundle(self.field_fn,c,self.bundle,self.wall_fn,self.wall_parameters(c),
            phi_start=self.phi_start,phi_guesses=snapshot.get('hit_phi',self.phi_guesses),
            maximum_phi_shift=self.maximum_phi_shift,n_steps=self.n_steps,
            launch_RZ=None if self.launch_fn is None else self.launch_fn(c),weights=self._weights(c))

    def _residual(self,hits,lengths,c):
        return (jnp.asarray(self.observables(hits,lengths,self._weights(c),c))-jnp.asarray(self.target))/jnp.asarray(self.scales)

    def refresh(self,c,snapshot):
        self._check_snapshot(snapshot)
        checked=self.production_validate(np.array(c,copy=True),dict(snapshot))
        if not isinstance(checked,OpenBundleValidation):raise TypeError('Production callback must return OpenBundleValidation')
        if not checked.valid:return Evaluation(np.zeros(len(self.target)),snapshot,valid=False,status=checked.status)
        local=self._trace(jnp.asarray(c),snapshot)
        hits=np.asarray(local.intersections.point_RZPhi)
        if hits.shape!=checked.hit_RZPhi.shape or checked.connection_lengths.shape!=(len(self.bundle.labels),):raise ValueError('Production launch identity/shape changed')
        physical_hits=np.column_stack((checked.hit_RZPhi[:,0]*np.cos(checked.hit_RZPhi[:,2]),
                                       checked.hit_RZPhi[:,0]*np.sin(checked.hit_RZPhi[:,2]),checked.hit_RZPhi[:,1]))
        discrepancy=float(np.max(np.linalg.norm(np.asarray(local.intersections.point_xyz)-physical_hits,axis=1)))
        phase_error=float(np.max(np.abs(hits[:,2]-checked.hit_RZPhi[:,2])))
        valid=bool(np.asarray(local.valid)) and np.isfinite(discrepancy) and discrepancy<=self.maximum_hit_discrepancy and phase_error<=self.maximum_phi_shift
        if snapshot and np.max(np.abs(checked.hit_RZPhi[:,2]-np.asarray(snapshot['hit_phi'])))>self.maximum_phi_shift:valid=False
        updated=dict(definition=self.fingerprint,hit_phi=checked.hit_RZPhi[:,2].tolist(),hits=checked.hit_RZPhi.tolist(),lengths=checked.connection_lengths.tolist(),parameters=np.asarray(c).tolist())
        return Evaluation(np.asarray(self._residual(checked.hit_RZPhi,checked.connection_lengths,c)),updated,
            valid=valid,status='valid' if valid else 'local_production_correspondence_failed',diagnostics={'hit_discrepancy_m':discrepancy,'unwrapped_phase_discrepancy_rad':phase_error})

    def model(self,c,snapshot):
        self._check_snapshot(snapshot)
        if not snapshot:raise ValueError('Production refresh required before local linearization')
        if not np.array_equal(c,np.asarray(snapshot['parameters'])):raise ValueError('Model parameters differ from accepted snapshot')
        def residual(parameters):
            local=self._trace(parameters,snapshot)
            hits=local.intersections.point_RZPhi
            return self._residual(hits,local.connection_length,parameters)
        local=self._trace(jnp.asarray(c),snapshot)
        # The model value uses accepted production observables; the smooth
        # model Jacobian is qualified separately against production differences.
        value=self._residual(jnp.asarray(snapshot['hits']),jnp.asarray(snapshot['lengths']),jnp.asarray(c))
        return Evaluation(np.asarray(value),snapshot,jacobian=np.asarray(jax.jacfwd(residual)(jnp.asarray(c))),valid=bool(np.asarray(local.valid)))
