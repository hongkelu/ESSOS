import numpy as np
import pytest
from essos.vmex_domain import FourierLCFSExteriorDomain


def test_torus_interior_exterior_and_unresolved_boundary():
    domain = FourierLCFSExteriorDomain([0,1],[0,0],[1.,.3],[0.,.3],ntheta=128)
    points = np.array([[1.,0.,0.],[1.4,0.,0.],[0.,0.,0.],[1.3,0.,0.],[.6,0.,0.]])
    np.testing.assert_array_equal(domain(points),[False,True,False,False,True])
    angles = np.arange(13)*2*np.pi/13
    xyz = np.column_stack((1.4*np.cos(angles),1.4*np.sin(angles),np.zeros(13)))
    np.testing.assert_array_equal(domain(xyz),True)
    assert domain.chord_error_bound>0
    assert not domain.rmnc.flags.writeable
    with pytest.raises(ValueError, match='integers'):
        FourierLCFSExteriorDomain([0,.5],[0,0],[1.,.3],[0.,.3])


def test_helical_boundary_and_clearance_guard():
    domain = FourierLCFSExteriorDomain([0,1,0],[0,0,3],[1.,.3,.1],[0.,.3,0.],
        minimum_section_clearance=.02)
    # The same R is exterior at phi=pi/3 and interior at phi=0.
    phi = np.pi/3
    points = [[1.35,0.,0.],[1.35*np.cos(phi),1.35*np.sin(phi),0.],[1.41,0.,0.]]
    np.testing.assert_array_equal(domain(points),[False,True,False])
    with pytest.raises(ValueError):domain([[np.nan,0,0]])
