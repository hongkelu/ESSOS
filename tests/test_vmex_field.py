"""Real VMEX virtual casing against independent filament Biot--Savart data.

This optional dependency lane is invoked explicitly with a no-skip audit.
The prescribed torus is a layer-potential fixture, not an MHD equilibrium.
"""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
pytest.importorskip("vmex")
from vmex.core.extender import VmecExtender
from vmex.core.virtual_casing import VmecSurfaceFieldData
from essos.equilibrium_response import EquilibriumDefinition
from essos.vmex_field import VmexExteriorField

jax.config.update('jax_enable_x64', True)


def ring(points, radius=1., current=3e5, segments=256):
    t = jnp.arange(segments)*2*jnp.pi/segments
    source = radius*jnp.stack((jnp.cos(t),jnp.sin(t),jnp.zeros_like(t)),axis=1)
    dl = radius*jnp.stack((-jnp.sin(t),jnp.cos(t),jnp.zeros_like(t)),axis=1)*2*jnp.pi/segments
    delta = points[:,None,:]-source[None]
    return 1e-7*current*jnp.sum(jnp.cross(dl[None],delta)/jnp.linalg.norm(delta,axis=-1)[...,None]**3,axis=1)


def toroidal(points, strength=1.):
    x,y,_ = points.T
    return strength*jnp.stack((-y,x,jnp.zeros_like(x)),axis=1)/(x*x+y*y)[:,None]


def surface(p, n=24):
    theta=jnp.arange(n)*2*jnp.pi/n;phi=theta
    ph,th=jnp.meshgrid(phi,theta,indexing='ij')
    major=p[1] if len(p)>1 else 1.
    R=major+.3*jnp.cos(th)
    gamma=jnp.stack((R*jnp.cos(ph),R*jnp.sin(ph),.3*jnp.sin(th)))
    et=jnp.stack((-.3*jnp.sin(th)*jnp.cos(ph),-.3*jnp.sin(th)*jnp.sin(ph),.3*jnp.cos(th)))
    ep=jnp.stack((-R*jnp.sin(ph),R*jnp.cos(ph),jnp.zeros_like(R)))
    area=jnp.cross(et,ep,axis=0);normal=area/jnp.linalg.norm(area,axis=0)
    xyz=gamma.reshape(3,-1).T
    total=ring(xyz,radius=major,current=3e5*p[0])+toroidal(xyz)
    return VmecSurfaceFieldData(gamma=gamma,B_total=total.T.reshape(gamma.shape),
        normal=normal,area_vector=area,theta=theta,phi=phi,nfp=1,
        stellsym=False,signgs=1,source_convention='analytic_filament_fixture')


def make_field(p=(1.,1.)):
    backend=VmecExtender.from_parameterized_surface_data(surface,jnp.array(p[:1]),
        external_parameters=jnp.array(p[1:]),
        external_field_from_parameters=lambda q:lambda xyz:toroidal(xyz,q[0]),
        external_dof_names=('external_strength',),dof_names=('internal_current',),
        digits=4,levels=((48,24),(96,48)),accuracy_check='off')
    definition=EquilibriumDefinition('known-current torus','manufactured',
        ('internal_current','external_strength'),('surface_geometry',),
        'outside circular torus R0=1,a=.3',field_mode='prescribed_background')
    return VmexExteriorField(backend,definition=definition,
        domain_check=lambda x:np.hypot(np.hypot(x[:,0],x[:,1])-1.,x[:,2])>.3,
        equilibrium_valid=True,maximum_error=1e-4)


def test_vmex_exterior_field_and_total_response():
    points=np.array([[1.9,0.,.2],[0.,1.85,-.1]])
    field=make_field();value,report=field.sample(points)
    expected=np.asarray(ring(points)+toroidal(points))
    np.testing.assert_allclose(value,expected,rtol=2e-4,atol=2e-5)
    assert report.maximum_error_estimate<1e-4
    direction=np.array([.3,-.2]);response=field.jvp(points,direction)
    exact=np.asarray(.3*ring(points)-.2*toroidal(points))
    np.testing.assert_allclose(response,exact,rtol=3e-4,atol=2e-5)
    h=1e-4
    fd=(make_field(np.ones(2)+h*direction).B(points)-make_field(np.ones(2)-h*direction).B(points))/(2*h)
    np.testing.assert_allclose(response,fd,rtol=3e-4,atol=2e-6)
    cotangent=np.arange(6).reshape(2,3)/7
    np.testing.assert_allclose(np.vdot(response,cotangent),np.vdot(direction,field.vjp(points,cotangent)),rtol=1e-10,atol=1e-10)
    tied=VmexExteriorField(field.backend,
        definition=replace(field.definition,controls=('common_current_factor',)),
        domain_check=field.domain_check,equilibrium_valid=True,maximum_error=1e-4,
        control_jacobian=np.ones((2,1)))
    np.testing.assert_allclose(tied.jvp(points,[1.]),expected,rtol=3e-4,atol=2e-5)
    np.testing.assert_allclose(tied.vjp(points,cotangent),[np.sum(field.vjp(points,cotangent))],rtol=1e-10)
    assert not tied.control_jacobian.flags.writeable
    # Changing points must invalidate backend spatial pullback caches.
    np.testing.assert_allclose(field.jvp(points[:1],direction),response[:1],rtol=1e-10,atol=1e-10)


def test_vmex_domain_and_accuracy_rejection():
    field=make_field()
    with pytest.raises(ValueError,match='domain'):field.B(np.array([1.,0.,0.]))
    field.maximum_error=1e-30
    with pytest.raises(ValueError,match='quadrature'):field.B(np.array([1.301,0.,0.]))
    with pytest.raises(ValueError,match='Equilibrium acceptance'):
        VmexExteriorField(field.backend,definition=field.definition,
            domain_check=field.domain_check,equilibrium_valid=False)
    field.definition=replace(field.definition,controls=('wrong','names'))
    field.maximum_error=1e-4
    with pytest.raises(ValueError,match='DOFs'):field.vjp(np.array([[1.9,0.,.2]]),np.ones((1,3)))


def test_vmex_grid_native_trace():
    from pyna._cyna import is_available
    from pyna.toroidal.flt import trace_wall_hits_twall_field, strike_line_from_wall_hits
    from pyna.toroidal.geometry import ToroidalWall
    from scipy.integrate import solve_ivp
    assert is_available()
    field=make_field()
    grid,report=field.to_pyna_grid(np.linspace(1.8,2.,13),np.linspace(-.15,.15,13),
                                 np.arange(4)*2*np.pi/4,batch_size=169)
    assert report.points==676 and report.maximum_error_estimate<=1e-4
    np.testing.assert_allclose(grid.BPhi[:,:,0],1/np.linspace(1.8,2.,13)[:,None]*np.ones((1,13)),rtol=2e-4,atol=2e-5)
    phi=np.arange(12)*2*np.pi/12
    wall=ToroidalWall(phi,np.tile([1.81,1.99,1.99,1.81],(12,1)),
                      np.tile([-.1,-.1,.1,.1],(12,1)))
    hits=trace_wall_hits_twall_field(grid,[1.9],[0.],0.,2.,.002,wall,direction='+')
    hit=strike_line_from_wall_hits(hits,direction='+')
    assert len(hit['R'])==1
    exact_field=jax.jit(lambda x:ring(x[None])[0]+toroidal(x[None])[0])
    def rhs(phi,rz):
        R,Z=rz;cs,sn=np.cos(phi),np.sin(phi)
        bx,by,bz=np.asarray(exact_field(jnp.array([R*cs,R*sn,Z])))
        bphi=-bx*sn+by*cs
        return R*np.array([bx*cs+by*sn,bz])/bphi
    def event(phi,rz):return rz[1]+.1
    event.terminal=True
    reference=solve_ivp(rhs,(0.,2.),[1.9,0.],rtol=1e-10,atol=1e-12,events=event)
    assert len(reference.t_events[0])==1
    np.testing.assert_allclose(hit['phi'],reference.t_events[0],atol=5e-4,rtol=0.)
    np.testing.assert_allclose(hit['R'],reference.y_events[0][:,0],atol=5e-4,rtol=0.)


def test_vmex_moving_surface_response():
    points=np.array([[1.9,.2,.1]])
    backend=VmecExtender.from_parameterized_surface_data(surface,jnp.array([1.,1.]),
        external_field=toroidal,dof_names=('internal_current','major_radius'),
        digits=4,levels=((48,24),(96,48)),accuracy_check='off')
    definition=EquilibriumDefinition('moving prescribed interface','manufactured',
        ('internal_current','major_radius'),('external_field',),'R > 1.5',
        field_mode='prescribed_background')
    field=VmexExteriorField(backend,definition=definition,equilibrium_valid=True,
        domain_check=lambda x:np.hypot(x[:,0],x[:,1])>1.5,maximum_error=1e-4)
    direction=np.array([.2,.1])
    exact=jax.jvp(lambda p:ring(jnp.asarray(points),radius=p[1],current=3e5*p[0]),
                  (jnp.ones(2),),(jnp.asarray(direction),))[1]
    np.testing.assert_allclose(field.jvp(points,direction),exact,rtol=5e-4,atol=2e-5)
