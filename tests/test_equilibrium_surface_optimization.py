import numpy as np
import jax.numpy as jnp
from scipy.optimize import brentq
from pyna.topo.torus_solver import InvariantCircleProblem
from essos.equilibrium import EquilibriumDefinition,ImplicitEquilibriumField
from essos.topology_objectives import CircleObjective
from essos.topology_objectives import EquilibriumCircleObjective
from essos.topology_optimizer import DesignProblem,optimize_topology


def test_coupled_optimizer_resolves_equilibrium_and_topology_at_acceptance():
    definition=EquilibriumDefinition('manufactured-equilibrium','manufactured',('c',),('profile',),'manufactured domain')
    equilibrium=ImplicitEquilibriumField(definition,lambda c,z:np.sqrt(1+c),lambda z,c:z*z-1-c,
                                         lambda x,z,c:jnp.array([0.,1.,z[0]]))
    theta=np.arange(9)*2*np.pi/9;reference=np.column_stack((2+.4*np.cos(theta),.4*np.sin(theta)))
    def mapping(x,p):
        v=x-jnp.array([2.,0.]);angle=.7+.1*p[0]+.2*p[1]+.2*jnp.dot(v,v)
        return jnp.array([2.,0.])+jnp.array([[jnp.cos(angle),-jnp.sin(angle)],[jnp.sin(angle),jnp.cos(angle)]])@v
    circle=InvariantCircleProblem(mapping,reference,area=np.pi*.4**2)
    surface=CircleObjective(circle,lambda z,p:jnp.array([z[-2]]),(.97,),(.04,),.932,'coupled-fixed-area')
    objective=EquilibriumCircleObjective(equilibrium,surface,'joint-model-v1')
    initial=objective.refresh(np.array([0.]),{})
    model=objective.model(np.array([0.]),initial.snapshot)
    np.testing.assert_allclose(model.jacobian,[[5.]],rtol=1e-10)
    problem=DesignProblem(objective.fingerprint,[0.],[.2],[-.5],[.5])
    result=optimize_topology(problem,[0.],objective.model,objective.refresh,max_iterations=10)
    expected=brentq(lambda c:.732+.1*c+.2*np.sqrt(1+c)-.97,0.,.5)
    np.testing.assert_allclose(problem.physical(result.state.q),[expected],atol=1e-6)
    assert result.state.objective<1e-12
    for trial in result.trials:
        if trial.get('accepted'):
            assert trial['diagnostics']['equilibrium_status']=='valid'
            assert trial['diagnostics']['circle_status']=='valid'
    saved=result.state.snapshot
    np.testing.assert_allclose(np.square(saved['equilibrium_state']),1+np.asarray(saved['parameters']),atol=1e-12)
