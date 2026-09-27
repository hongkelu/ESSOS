"""Production-refreshed labelled manifold samples for the constrained optimizer."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import jax
import jax.numpy as jnp
from essos.topology_optimizer import Evaluation
from essos.manifold_optimization import trace_manifold_reference
from pyna.topo.manifold_correspondence import ManifoldBranchReference,compare_jax_manifold_branch
from pyna.topo.manifold_refresh import refresh_manifold_branch_reference_field


@dataclass(frozen=True)
class ManifoldSampleObjective:
    field_factory: object
    production_field_factory: object
    initial_branch: ManifoldBranchReference
    sample_label: object
    target: tuple[float,float]
    scales: tuple[float,float]
    identity: str
    maximum_anchor_displacement: float = .002
    minimum_direction_alignment: float = .9
    maximum_sample_displacement: float = .002
    correspondence_tolerance: float = 2e-4
    steps: int = 64
    derivative_mode: str = 'moving_linear_seed'

    def __post_init__(self):
        self.initial_branch.point(self.sample_label)
        for name in ('target','scales'):
            value=tuple(map(float,getattr(self,name)))
            if len(value)!=2 or not np.all(np.isfinite(value)):raise ValueError('Finite RZ target and scales required')
            object.__setattr__(self,name,value)
        if min(self.scales)<=0 or not self.identity:raise ValueError('Positive scales and identity required')
        if self.derivative_mode not in ('frozen_reference','moving_linear_seed'):raise ValueError('Unknown derivative mode')

    @property
    def fingerprint(self):
        controls=(self.identity,self.target,self.scales,self.sample_label.key,self.maximum_anchor_displacement,
                  self.minimum_direction_alignment,self.maximum_sample_displacement,self.correspondence_tolerance,
                  self.steps,self.derivative_mode)
        return hashlib.sha256(repr(controls).encode()+json.dumps(self.initial_branch.to_dict(),sort_keys=True,allow_nan=False).encode()).hexdigest()

    def _branch(self,snapshot):
        if not snapshot:return self.initial_branch
        if snapshot.get('definition')!=self.fingerprint:raise ValueError('Manifold objective definition changed')
        return ManifoldBranchReference.from_dict(snapshot['branch'])

    def _residual(self,parameters,branch):
        traced=trace_manifold_reference(self.field_factory(parameters),branch,n_steps_per_span=self.steps,
                                        bphi_floor=1e-8,derivative_mode=self.derivative_mode)
        point=traced.generations[self.sample_label.generation_index,branch.seed_index(self.sample_label.seed_order)]
        return (point-jnp.asarray(self.target))/jnp.asarray(self.scales)

    def model(self,parameters,snapshot):
        branch=self._branch(snapshot)
        local_residual=self._residual(jnp.asarray(parameters),branch)
        # Match the local model value to the independently refreshed physical
        # sample. The JAX-Cyna discretization offset is held fixed in this model.
        residual=(branch.point(self.sample_label)-np.asarray(self.target))/np.asarray(self.scales)
        jacobian=jax.jacfwd(lambda p:self._residual(p,branch))(jnp.asarray(parameters))
        valid=bool(np.all(np.isfinite(residual)) and np.all(np.isfinite(jacobian)))
        return Evaluation(np.asarray(residual),snapshot,valid=valid,status='valid' if valid else 'local_response_invalid',jacobian=np.asarray(jacobian),
                          diagnostics={'fixed_discretization_offset':(np.asarray(local_residual)-residual).tolist()})

    def refresh(self,parameters,snapshot):
        previous=self._branch(snapshot)
        production=self.production_field_factory(parameters)
        report=refresh_manifold_branch_reference_field(production,previous,
                    maximum_anchor_displacement_m=self.maximum_anchor_displacement,
                    minimum_direction_alignment=self.minimum_direction_alignment,
                    DPhi=abs(previous.map_span)/self.steps,fixed_point_residual_tolerance=1e-9)
        if not report.accepted:
            return Evaluation(np.zeros(2),snapshot,valid=False,status=report.rejection_reason or 'production_refresh_failed')
        branch=report.candidate_branch
        displacement=float(np.linalg.norm(branch.point(self.sample_label)-previous.point(self.sample_label)))
        trace=trace_manifold_reference(self.field_factory(parameters),branch,n_steps_per_span=self.steps,
                                       bphi_floor=1e-8,derivative_mode=self.derivative_mode)
        correspondence=compare_jax_manifold_branch(branch,trace.generations,absolute_tolerance_m=self.correspondence_tolerance,require_complete=True)
        valid=correspondence.accepted and displacement<=self.maximum_sample_displacement
        residual=(branch.point(self.sample_label)-np.asarray(self.target))/np.asarray(self.scales)
        updated=dict(definition=self.fingerprint,branch=branch.to_dict())
        status='valid' if valid else 'sample_displacement_exceeded' if displacement>self.maximum_sample_displacement else 'jax_cyna_correspondence_failed'
        return Evaluation(residual,updated,valid=valid,status=status,
                          diagnostics=dict(sample_displacement_m=displacement,jax_cyna_deviation_m=correspondence.max_deviation_m))
