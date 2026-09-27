import numpy as np
import pytest
import jax
import jax.numpy as jnp
from essos.topology_optimizer import content_identity,CurrentGridBasis
from essos.coils import Coils,Curves
from essos.fields import BiotSavart


def test_each_physical_definition_component_invalidates_candidate_identity():
    baseline=dict(currents=np.array([1.,2.]),geometry={'curve':np.ones((2,3))},wall={'r':2.},numerics={'steps':128},topology={'label':'u+'},target={'r':1.5})
    original=content_identity(**baseline)
    for key in baseline:
        changed=dict(baseline);changed[key]={'changed':True}
        assert content_identity(**changed)!=original
    assert content_identity(**baseline)==original


def test_current_basis_matches_fresh_biot_savart_values_and_jacobian():
    dofs=np.zeros((2,3,3));dofs[:,0,2]=[1.,1.2];dofs[:,1,1]=[1.,1.2];dofs[:,2,0]=[-.5,.5]
    curves=Curves(jnp.asarray(dofs),n_segments=32,nfp=1,stellsym=False)
    points=jnp.array([[1.4,.2,0.],[.3,.1,.3],[1.7,0.,.1]])
    def fresh(currents):return jax.vmap(BiotSavart(Coils(curves,currents,currents_scale=1.)).B)(points)
    basis=CurrentGridBasis(np.stack([np.asarray(fresh(e)) for e in jnp.eye(2)]),'geometry-grid-v1')
    currents=jnp.array([2e5,-3e5]);cached=lambda c:basis.evaluate(c,geometry_grid_key='geometry-grid-v1')
    np.testing.assert_allclose(cached(currents),fresh(currents),rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(jax.jacfwd(cached)(currents),jax.jacfwd(fresh)(currents),rtol=1e-13,atol=1e-18)
    with pytest.raises(ValueError,match='rebuild'):basis.evaluate(currents,geometry_grid_key='new-shape')
