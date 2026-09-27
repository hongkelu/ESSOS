"""Required native integration for arbitrary open-line optimization."""
import numpy as np
import jax.numpy as jnp
from pyna.fields import VectorFieldCylind
from pyna.toroidal.geometry import ToroidalWall
from pyna.toroidal.control.strike_heat import StrikeSeedBundle
from pyna.topo.clearance3d import TriangleWall
from pyna.topo.open_validation import validate_open_bundle_3d
from essos.open_bundle_optimization import LaunchBundle, OpenBundleObjective
from essos.topology_optimizer import DesignProblem,optimize_topology


def test_general_open_optimizer_requires_native_first_hit_and_whole_leg():
    phi=np.linspace(0,2*np.pi,128,endpoint=False);theta=np.linspace(0,2*np.pi,32,endpoint=False)
    r=1.5+.4*np.cos(theta);z=.4*np.sin(theta)
    wall=ToroidalWall(phi,np.broadcast_to(r,(len(phi),len(r))).copy(),np.broadcast_to(z,(len(phi),len(z))).copy(),nfp=1)
    vertices=np.stack((wall.R*np.cos(phi[:,None]),wall.R*np.sin(phi[:,None]),wall.Z),axis=-1).reshape(-1,3)
    faces=[];patches=[];nt=len(theta)
    for i in range(len(phi)):
        for j in range(nt):
            a=i*nt+j;b=((i+1)%len(phi))*nt+j;c=((i+1)%len(phi))*nt+(j+1)%nt;d=i*nt+(j+1)%nt
            faces.extend([[a,b,c],[a,c,d]])
            patch='target' if min(np.cos(theta[j]),np.cos(theta[(j+1)%nt]))>.9 else 'vessel'
            patches.extend([patch,patch])
    mesh=TriangleWall(vertices,faces,tuple(patches))
    radius=np.linspace(1.,2.,16);height=np.linspace(-.5,.5,16);angles=np.linspace(0,2*np.pi,8,endpoint=False)
    rr,zz,pp=np.meshgrid(radius,height,angles,indexing='ij')
    seeds=StrikeSeedBundle('arbitrary','general_open',np.array([1.5]),np.array([0.]),np.array([0.]),'+',np.ones(1),'relative',np.zeros(1))
    validations=[]
    def production(c,snapshot):
        native=VectorFieldCylind(R=radius,Z=height,Phi=angles,BR=np.full_like(rr,c[0]),BZ=np.zeros_like(rr),BPhi=rr,nfp=1)
        checked=validate_open_bundle_3d(native,seeds,wall,mesh,target_patches=('target',),maximum_speed_m_per_rad=2.,
            required_clearance_m=.001,terminal_exclusion_length_m=.6,trace_error_m=1e-5,wall_error_m=.001,DPhi=.01,
            previous_hit_phi=snapshot.get('hit_phi'),maximum_hit_phase_shift=.5)
        validations.append(checked);return checked
    def field(x,c):
        r=jnp.hypot(x[0],x[1]);return jnp.array([c[0]*x[0]/r-x[1],c[0]*x[1]/r+x[0],0.])
    objective=OpenBundleObjective(LaunchBundle(('a',),[[1.5,0.]],[1.]),field,
        lambda xyz,w:w[0]-jnp.hypot(xyz[0],xyz[1]),lambda c:jnp.array([1.9]),lambda hits,lengths,w,c:hits[:,2],
        production,(3.8,),(.2,),(4.,),'native-arbitrary-open',maximum_phi_shift=.5,max_length=20.)
    problem=DesignProblem(objective.fingerprint,[.1],[.01],[.09],[.12])
    result=optimize_topology(problem,[.1],objective.model,objective.refresh,max_iterations=8)
    assert result.state.objective<1e-10
    np.testing.assert_allclose(problem.physical(result.state.q),[.4/3.8],atol=1e-7)
    assert all(v.valid and len(v.certificates)==1 for v in validations)
    assert len(validations)>=result.state.iteration+1
