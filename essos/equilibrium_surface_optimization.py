"""Joint equilibrium and closed-surface acceptance for reduced response models."""
from dataclasses import dataclass
import hashlib
import numpy as np
from essos.topology_optimizer import Evaluation


@dataclass(frozen=True)
class EquilibriumCircleObjective:
    """Compose an equilibrium response with a physical invariant-circle target.

    surface is a CircleObjective whose map and observables take the combined
    parameter vector concatenate((controls, equilibrium_unknowns)). Its map
    must use the declared Eulerian field and spatial domain. Every refresh
    re-solves the equilibrium and the surface; both validity gates must pass.
    The dense reference chain rule is intended for reduced problems. Material
    VMEC grid samples cannot be substituted for an Eulerian map here.
    """
    equilibrium: object
    surface: object
    identity: str

    def __post_init__(self):
        if not self.identity:raise ValueError('A versioned coupled problem identity is required')

    @property
    def fingerprint(self):
        return hashlib.sha256(repr((self.identity,self.equilibrium.definition,self.surface.fingerprint)).encode()).hexdigest()

    def _evaluate(self,c,snapshot,derivatives):
        if snapshot and snapshot.get('definition')!=self.fingerprint:raise ValueError('Coupled problem definition changed')
        equilibrium=self.equilibrium.solve(c,initial=snapshot.get('equilibrium_state'))
        diagnostics=dict(equilibrium_status=equilibrium.status,equilibrium_residual=equilibrium.residual_norm,
                         equilibrium_condition=equilibrium.condition,field_mode=self.equilibrium.definition.field_mode)
        if not equilibrium.valid:
            return Evaluation(np.zeros(len(self.surface.target)),snapshot,valid=False,status=equilibrium.status,diagnostics=diagnostics)
        combined=np.concatenate((np.asarray(c),equilibrium.state))
        surface_snapshot=snapshot.get('surface',{})
        result=(self.surface.model if derivatives else self.surface.refresh)(combined,surface_snapshot)
        if derivatives and result.valid:
            response=np.column_stack([self.equilibrium.state_jvp(equilibrium,d) for d in np.eye(len(c))])
            result.jacobian=result.jacobian@np.vstack((np.eye(len(c)),response))
        result.diagnostics={**result.diagnostics,**diagnostics}
        if result.valid:
            result.snapshot=dict(definition=self.fingerprint,parameters=np.asarray(c).tolist(),
                                 equilibrium_state=equilibrium.state.tolist(),surface=result.snapshot)
        else:result.snapshot=snapshot
        return result

    def model(self,c,snapshot):return self._evaluate(c,snapshot,True)

    def refresh(self,c,snapshot):return self._evaluate(c,snapshot,False)
