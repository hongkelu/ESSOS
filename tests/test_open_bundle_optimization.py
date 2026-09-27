import numpy as np
import jax.numpy as jnp
import pytest
pytest.importorskip("pyna.topo.open_validation")
from pyna.topo.open_validation import OpenBundleValidation  # noqa: E402
from essos.open_bundle_optimization import LaunchBundle, OpenBundleObjective  # noqa: E402
from essos.topology_optimizer import DesignProblem,optimize_topology


def test_general_open_objective_independent_refresh_and_moving_wall():
    bundle=LaunchBundle(('a',),[[2.,0.]],[1.])
    def field(x,c):
        r=jnp.hypot(x[0],x[1]);return jnp.array([-x[1]/r,x[0]/r,c[0]/r])
    def wall(xyz,w):return w[0]-xyz[2]  # below the plane z = w, positive inside
    calls=[]
    def production(c,s):
        # Independent exact helix/plane intersection and analytic arc length.
        height=1.+c[1];phi=height/c[0];length=height*np.sqrt(1+(2/c[0])**2)
        calls.append(c.copy())
        return OpenBundleValidation(True,'analytic_verified',[[2.,height,phi]],[length],())
    objective=OpenBundleObjective(bundle,field,wall,lambda c:jnp.array([1.+c[1]]),
        lambda hits,lengths,w,c:jnp.array([hits[0,2],hits[0,1]]),production,
        (1.8,1.05),(.2,.1),(2.,),'analytic-moving-wall',maximum_phi_shift=.5,max_length=20.)
    initial=objective.refresh(np.array([.5,0.]),{})
    local=objective.model(np.array([.5,0.]),initial.snapshot)
    np.testing.assert_allclose(local.jacobian,[[-20.,10.],[0.,10.]],atol=1e-10)
    problem=DesignProblem(objective.fingerprint,[.5,0.],[.1,.1],[.3,-.1],[.8,.1])
    result=optimize_topology(problem,[.5,0.],objective.model,objective.refresh,max_iterations=12)
    assert result.state.objective<1e-10
    np.testing.assert_allclose(problem.physical(result.state.q),[1.05/1.8,.05],atol=1e-6)
    assert len(calls)>=result.state.iteration+1
    with pytest.raises(ValueError,match='definition'):
        objective.model(np.array([.5,0.]),{'definition':'changed'})
    with pytest.raises(ValueError,match='parameters'):
        objective.model(np.array([.6,0.]),initial.snapshot)

    # RZPhi components have different units. At R=2 m, a 0.001 rad phase
    # mismatch is a 0.002 m physical discrepancy, not 0.001 m.
    from dataclasses import replace
    def shifted_production(c,s):
        value=production(c,s);hits=np.array(value.hit_RZPhi,copy=True);hits[:,2]+=.001
        return OpenBundleValidation(True,'deliberate_mismatch',hits,value.connection_lengths,())
    shifted=replace(objective,production_validate=shifted_production,maximum_hit_discrepancy=.0015)
    rejected=shifted.refresh(np.array([.5,0.]),{})
    assert not rejected.valid
    assert rejected.diagnostics['hit_discrepancy_m']>.0019
