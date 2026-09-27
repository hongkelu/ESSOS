import numpy as np
import jax.numpy as jnp
from pyna.topo.torus_solver import InvariantCircleProblem
from essos.surface_optimization import CircleObjective
from essos.topology_optimizer import DesignProblem,optimize_topology


def test_closed_surface_only_optimization_refreshes_area_and_transform():
    theta=np.arange(11)*2*np.pi/11
    reference=np.column_stack((2+.4*np.cos(theta),.4*np.sin(theta)))
    def mapping(x,c):
        center=jnp.array([2.+c[0],0.]);v=x-center
        angle=.7+.2*jnp.dot(v,v)+.3*c[0]
        return center+jnp.array([[jnp.cos(angle),-jnp.sin(angle)],[jnp.sin(angle),jnp.cos(angle)]])@v
    circle=InvariantCircleProblem(mapping,reference,area=np.pi*.4**2)
    objective=CircleObjective(circle,lambda z,c:jnp.array([z[-2]]),(.75,),(.03,),.732,'fixed-area-transform')
    problem=DesignProblem(objective.fingerprint,[0.],[.1],[-.2],[.2])
    result=optimize_topology(problem,[0.],objective.model,objective.refresh,max_iterations=8)
    assert result.state.objective<1e-12
    np.testing.assert_allclose(problem.physical(result.state.q),[.06],atol=1e-6)
    assert all(t['diagnostics']['dense_invariance_residual']<1e-7 for t in result.trials if t.get('accepted'))
    assert result.state.snapshot['area']==circle.area
