#!/usr/bin/env python3
"""Numerical NCSX open-line current control with native whole-leg acceptance.

The launch, vessel and target patch are fixed. The target is an independently
traced nearby control setting, and bounds are numerical, not engineering limits.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=['implicit','finite_difference'],default='implicit')
    args=parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    import numpy as np
    import jax
    import jax.numpy as jnp
    from essos.coils import load_simsopt_xyz_coils
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.topology import essos_field_to_pyna_cylindrical_grid
    from essos.topology_objectives import LaunchBundle,OpenBundleObjective
    from essos.topology_optimizer import DesignProblem,optimize_topology
    from pyna.topo.open_validation import validate_open_bundle_3d
    from pyna.topo.clearance3d import TriangleWall
    from pyna.toroidal.control.strike_heat import StrikeSeedBundle,trace_wall_strikes_field
    from qualify_ncsx_open_inputs import load_ncsx_vessel
    source=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization'
    wall,mesh,metadata=load_ncsx_vessel(source)
    coils=load_simsopt_xyz_coils(source/'ncsx_coils.json',n_segments=128)
    currents=jnp.asarray(coils.currents);curves=coils.curves;span=2*np.pi/3
    refined_curves=load_simsopt_xyz_coils(source/'ncsx_coils.json',n_segments=256).curves
    def physical(c):return BiotSavart(Coils(curves,currents*jnp.tile(c,6),currents_scale=1e6))
    radii=np.linspace(1.45,1.85,41);heights=np.linspace(.4,.8,41);angles=np.linspace(0,span,128,endpoint=False)
    def grid(c,refined=False):
        field=BiotSavart(Coils(refined_curves,currents*jnp.tile(c,6),currents_scale=1e6)) if refined else physical(c)
        r=np.linspace(radii[0],radii[-1],81) if refined else radii
        z=np.linspace(heights[0],heights[-1],81) if refined else heights
        phi=np.linspace(0,span,256,endpoint=False) if refined else angles
        return essos_field_to_pyna_cylindrical_grid(field,r,z,phi,nfp=3,batch_size=256)
    seeds=StrikeSeedBundle('ncsx-vessel-launch','general_open',np.array([1.7]),np.array([.65]),np.array([0.]),'+',np.ones(1),'relative',np.zeros(1))
    def hits(native):return trace_wall_strikes_field(native,(seeds,),wall,max_turns=1,DPhi=.001)[0]
    start=time.perf_counter();basegrid=grid(np.ones(3));base=hits(basegrid)
    if len(base.R)!=1 or not 0<base.phi[0]<.4:raise ValueError('Launch does not have the qualified short first hit')
    hitxyz=np.array([base.R[0]*np.cos(base.phi[0]),base.R[0]*np.sin(base.phi[0]),base.Z[0]])
    centers=np.mean(mesh.vertices[mesh.faces],axis=1)
    patches=tuple('target' if d<.15 else 'vessel' for d in np.linalg.norm(centers-hitxyz,axis=1))
    mesh=TriangleWall(mesh.vertices,mesh.faces,patches)
    desired=np.array([1.001,.999,1.]);target_hits=hits(grid(desired))
    if len(target_hits.R)!=1:raise ValueError('Target control has no first hit')
    target=(float(target_hits.R[0]),float(target_hits.Z[0]))
    # Exact local level set of the production wall's RZ polygon interpolated
    # between toroidal sections. Integer cell/segment identities are piecewise
    # fixed and are checked through production correspondence and first hits.
    wall_r=jnp.asarray(wall.R);wall_z=jnp.asarray(wall.Z);nphi=len(wall.R);dphi=span/nphi
    def polygon_level(rz,phi):
        u=jnp.mod(phi,span)/dphi;lo=jnp.floor(u).astype(int);weight=u-lo
        points=jnp.column_stack(((1-weight)*wall_r[lo]+weight*wall_r[(lo+1)%nphi],
                                 (1-weight)*wall_z[lo]+weight*wall_z[(lo+1)%nphi]))
        edge=jnp.roll(points,-1,axis=0)-points
        t=jnp.clip(jnp.sum((rz-points)*edge,axis=1)/jnp.sum(edge*edge,axis=1),0,1)
        index=jnp.argmin(jnp.sum((rz-points-t[:,None]*edge)**2,axis=1))
        a=points[index];e=edge[index];return ((rz[0]-a[0])*e[1]-(rz[1]-a[1])*e[0])/jnp.linalg.norm(e)
    # connection_length needs a Cartesian level set that is positive inside the
    # vessel; orient the polygon distance by its sign at the (interior) launch.
    inside=float(jnp.sign(polygon_level(jnp.array([1.7,.65]),0.)))
    def wall_fn(xyz,unused):
        return inside*polygon_level(jnp.array([jnp.hypot(xyz[0],xyz[1]),xyz[2]]),jnp.arctan2(xyz[1],xyz[0]))
    # Bound the difference between the input triangles and the interpolated
    # cylindrical wall: bilinear-vs-triangle cross term plus cylindrical sag.
    phi_nodes=np.arange(nphi+1)*dphi
    rr=np.concatenate((np.asarray(wall.R),np.asarray(wall.R[:1])),axis=0)
    zz=np.concatenate((np.asarray(wall.Z),np.asarray(wall.Z[:1])),axis=0)
    xyz=np.stack((rr*np.cos(phi_nodes[:,None]),rr*np.sin(phi_nodes[:,None]),zz),axis=-1)
    cross_term=xyz[:-1]-xyz[1:]-np.roll(xyz[:-1],-1,axis=1)+np.roll(xyz[1:],-1,axis=1)
    wall_error=float(np.max(np.linalg.norm(cross_term,axis=-1))/4+np.max(rr)*dphi*dphi/8+np.max(np.abs(np.diff(rr,axis=0)))*dphi/4+3e-6)
    print(f'Conservative wall representation error bound: {wall_error} m',flush=True)
    validations=[]
    def production(c,snapshot,refined=False):
        native=grid(c,refined=refined)
        # Cyna's trilinear field interpolation is bounded by cell corner
        # values over this entire R/Z box and the qualified phi interval.
        nz=int(np.searchsorted(np.asarray(native.Phi),.4))+1
        bphi=np.asarray(native.BPhi)[:,:,:nz]
        if np.min(bphi)*np.max(bphi)<=0:raise ValueError('No usable toroidal-field lower bound in the declared open domain')
        minimum=float(np.min(np.abs(bphi)))
        br=float(np.max(np.abs(np.asarray(native.BR)[:,:,:nz])));bz=float(np.max(np.abs(np.asarray(native.BZ)[:,:,:nz])))
        speed=float(radii[-1]*np.sqrt(1+(br*br+bz*bz)/minimum**2))
        checked=validate_open_bundle_3d(native,seeds,wall,mesh,target_patches=('target',),maximum_speed_m_per_rad=speed,
            required_clearance_m=.001,terminal_exclusion_length_m=.2,trace_error_m=1e-5,wall_error_m=wall_error,
            DPhi=.0005 if refined else .001,max_turns=1,endpoint_tolerance_m=2e-5,previous_hit_phi=snapshot.get('hit_phi'),maximum_hit_phase_shift=.05)
        if checked.valid and checked.hit_RZPhi[0,2]>=.4:raise ValueError('Hit left the domain of the speed bound')
        print('Production validation:',checked.status,flush=True)
        if not checked.valid and len(checked.hit_RZPhi):
            rh,zh,ph=checked.hit_RZPhi[0];point=np.array([[rh*np.cos(ph),rh*np.sin(ph),zh]])
            print('Hit mesh distance and competing patch:',mesh.distance(point),mesh.distance(point,exclude_patch='target'),flush=True)
        validations.append(dict(valid=checked.valid,status=checked.status,speed_bound=speed,
            minimum_clearance_lower_bound=min((cert.minimum_lower_bound for cert in checked.certificates),default=None)))
        return checked
    objective=OpenBundleObjective(LaunchBundle(('fixed-vessel-launch',),[[1.7,.65]],[1.]),lambda x,c:physical(c).B(x),
        wall_fn,lambda c:jnp.zeros(0),lambda hits,lengths,w,c:hits[0,:2],production,target,(.001,.001),
        (float(base.phi[0]),),'ncsx-vessel-fixed-launch-v1',maximum_phi_shift=.05,maximum_hit_discrepancy=.002,max_length=10.)
    def constrained(value,c):
        value.equalities=np.array([np.sum(c)-3.]);value.equality_jacobian=np.ones((1,3));return value
    def refresh(c,s):return constrained(objective.refresh(c,s),c)
    def native_residual(c):
        value=hits(grid(c))
        if len(value.R)!=1 or abs(value.phi[0]-base.phi[0])>.05:raise ValueError('Finite-difference first-hit identity changed')
        return (np.array([value.R[0],value.Z[0]])-np.asarray(target))/.001
    def model(c,s):
        if args.method=='implicit':return constrained(objective.model(c,s),c)
        value=objective.refresh(c,s);step=1e-4
        value.jacobian=np.column_stack([(native_residual(c+step*d)-native_residual(c-step*d))/(2*step) for d in np.eye(3)])
        return constrained(value,c)
    initial=refresh(np.ones(3),{})
    if not initial.valid:raise ValueError('Initial whole-leg qualification failed: '+initial.status)
    local=objective.model(np.ones(3),initial.snapshot);direction=np.array([.3,-.2,-.1])
    response=local.jacobian@direction;derivative_checks=[]
    for step in (1e-3,3e-4,1e-4):
        fd=(native_residual(np.ones(3)+step*direction)-native_residual(np.ones(3)-step*direction))/(2*step)
        derivative_checks.append(dict(step=step,relative_error=float(np.linalg.norm(response-fd)/np.linalg.norm(fd))))
    if min(row['relative_error'] for row in derivative_checks)>.01:raise ValueError('Open-line response failed production finite differences')
    problem=DesignProblem(objective.fingerprint,np.ones(3),np.full(3,.005),np.full(3,.995),np.full(3,1.005))
    result=optimize_topology(problem,np.ones(3),model,refresh,max_iterations=8,initial_radius=.2,numerical_floor=1e-10,checkpoint=args.output/'checkpoint.json')
    final_parameters=problem.physical(result.state.q)
    refined=production(final_parameters,result.state.snapshot,refined=True)
    desired_refined=hits(grid(desired,refined=True))
    refined_error=float(np.linalg.norm(refined.hit_RZPhi[0,:2]-np.array([desired_refined.R[0],desired_refined.Z[0]]))) if refined.valid and len(desired_refined.R)==1 else None
    report=dict(case='NCSX vacuum general-open first-hit control',method=args.method,status=result.status,field_mode='vacuum_coils',
        launch_RZ_m=[1.7,.65],target_RZ_m=target,parameters=problem.physical(result.state.q).tolist(),
        initial_objective=initial.objective,final_objective=result.state.objective,trials=list(result.trials),
        validations=validations,derivative_checks=derivative_checks,refined_valid=refined.valid,refined_target_error_m=refined_error,mesh=metadata,elapsed_seconds=time.perf_counter()-start,
        current_bounds=[.995,1.005],sum_current_factors=3.,bounds_kind='numerical recovery benchmark',
        required_clearance_m=.001,terminal_exclusion_m=.2,wall_error_budget_m=wall_error,trace_error_budget_m=1e-5,
        limitation='Numerical recovery test; independent tracing/wall error budgets and engineering limits need further qualification')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if refined.valid and refined_error<1e-6 and result.state.objective<initial.objective and result.state.iteration>0 else 2


if __name__=='__main__':raise SystemExit(main())
