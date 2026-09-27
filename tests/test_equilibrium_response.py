import numpy as np
import jax.numpy as jnp
from scipy.optimize import root
import pytest
from essos.equilibrium_response import EquilibriumDefinition,ImplicitEquilibriumField


def test_external_primal_total_field_jvp_vjp_and_resolved_finite_difference():
    definition=EquilibriumDefinition('manufactured-response','manufactured',('a','b','coupling'),('gauge',),'all space')
    def residual(z,c):return jnp.array([z[0]**2+z[1]-c[0],z[1]+c[2]*z[0]-c[1]])
    def solve(c,initial):return root(lambda z:np.asarray(residual(z,c)),[1.,.2] if initial is None else initial).x
    def field(x,z,c):return jnp.array([-z[0]*x[1],z[0]*x[0],c[0]+z[1]])
    adapter=ImplicitEquilibriumField(definition,solve,residual,field)
    c=np.array([1.3,.4,.2]);solution=adapter.solve(c);assert solution.valid
    xyz=np.array([2.,.3,-.2]);direction=np.array([.3,-.4,.1]);eps=1e-5
    response=adapter.jvp(solution,xyz,direction)
    plus=adapter.solve(c+eps*direction);minus=adapter.solve(c-eps*direction)
    fd=(adapter.field(plus,xyz)-adapter.field(minus,xyz))/(2*eps)
    np.testing.assert_allclose(response,fd,rtol=1e-7,atol=1e-8)
    cotangent=np.array([.4,.7,-.2])
    np.testing.assert_allclose(cotangent@response,adapter.vjp(solution,xyz,cotangent)@direction,rtol=1e-11)
    assert np.linalg.norm(response-np.array([0,0,direction[0]]))>.1


def test_singular_equilibrium_is_rejected_without_regularization():
    definition=EquilibriumDefinition('fold','manufactured',('control',),(),'all space')
    adapter=ImplicitEquilibriumField(definition,lambda c,z:np.array([np.sqrt(c[0])]),lambda z,c:z*z-c,lambda x,z,c:z)
    state=adapter.solve([0.]);assert not state.valid
    assert state.status=='equilibrium_response_singular'
    with pytest.raises(ValueError,match='Invalid equilibrium'):adapter.jvp(state,[1,0,0],[1])
