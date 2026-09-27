#!/usr/bin/env python3
"""Compare native Cyna tracing of a real VMEX exterior grid with direct DOP853.

A short, fixed-equilibrium trajectory in an explicitly exterior Cartesian
region; not a wall-target optimization or a near-LCFS continuation model.
"""
import argparse
import json
from pathlib import Path
import shutil
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equilibrium-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    shutil.copyfile(__file__,args.output/'qualification_source.py')
    import vmex_accepted_case as loader
    shutil.copyfile(loader.__file__,args.output/'loader_source.py')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.integrate import solve_ivp
    from vmex.core import virtual_casing as vc
    from vmex.core.extender import VmecExtender
    from essos.fields import BiotSavart
    from essos.equilibrium import VmexExteriorField
    from essos.equilibrium import FourierLCFSExteriorDomain
    from essos.equilibrium import EquilibriumDefinition
    from pyna.toroidal.flt import trace_orbit_along_phi_field
    if not jax.config.x64_enabled:raise ValueError('Float64 required')
    start=time.perf_counter()
    case=loader.load_accepted_case(args.equilibrium_run)
    domain=FourierLCFSExteriorDomain.from_wout(case.wout)
    zbound=float(np.sum(abs(domain.zmns)+abs(domain.zmnc)))
    raxis=float(domain.rmnc[(domain.xm==0)&(domain.xn==0)][0])
    seed=np.array([raxis,zbound+.12]);phi_start=.1;phi_end=.125
    direct=jax.jit(jax.vmap(BiotSavart(case.coils).B))
    def backend(n,levels):
        data=vc.surface_field_data_from_state(case.inp,case.state,nphi=n,ntheta=n)
        return VmecExtender.from_surface_data(data,external_field=direct,digits=5,levels=levels,accuracy_check='off')
    coarse_backend=backend(96,((192,96),(384,192)))
    fine_backend=backend(144,((384,192),(768,384)))
    definition=EquilibriumDefinition('LI383 accepted continuation','free_boundary',('common_NCSX_current_factor',),
        ('pressure','plasma_current','toroidal_flux'),f'Z > {zbound} m')
    outside=lambda x:np.asarray(x)[:,2]>zbound
    field=VmexExteriorField(coarse_backend,definition=definition,domain_check=outside,
        equilibrium_valid=True,maximum_error=1e-3)
    refined=VmexExteriorField(fine_backend,definition=definition,domain_check=outside,
        equilibrium_valid=True,maximum_error=1e-3)
    point_field=jax.jit(lambda xyz:fine_backend.B(xyz[None])[0])
    def rhs(phi,rz):
        r,z=rz;cs,sn=np.cos(phi),np.sin(phi)
        if z<=zbound:raise ValueError('Reference trajectory left exterior region')
        bx,by,bz=np.asarray(point_field(jnp.array([r*cs,r*sn,z])))
        bp=-bx*sn+by*cs
        if not np.isfinite(bp) or abs(bp)<1e-5:raise ValueError('Unusable toroidal field')
        return r*np.array([bx*cs+by*sn,bz])/bp
    report=dict(case='LI383 finite-beta exterior native trace',checkpoint_sha256=case.checkpoint_sha256,
        equilibrium_run=str(args.equilibrium_run.resolve()),seed_RZ_m=seed.tolist(),phi_interval=[phi_start,phi_end],
        exterior_Z_lower_bound_m=zbound,trace_tolerance_m=1e-4,step_tolerance_m=1e-6,
        response_policy='User accepts recorded 0.17–0.23 percent common-current discrepancy for proceeding',
        physical_design_qualified=False,rows=[])
    def save():
        report['elapsed_seconds']=time.perf_counter()-start
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        print('Direct VMEX DOP853 reference trajectory',flush=True)
        ref=solve_ivp(rhs,(phi_start,phi_end),seed,method='DOP853',rtol=1e-10,atol=1e-12,dense_output=True,max_step=.0025)
        if not ref.success:raise ValueError('Reference ODE failed')
        phi=np.linspace(phi_start,phi_end,101);rz=ref.sol(phi).T
        xyz=np.column_stack((rz[:,0]*np.cos(phi),rz[:,0]*np.sin(phi),rz[:,1]))
        fine_value,validation=refined.sample(xyz)
        coarse_value,_=field.sample(xyz)
        quad_change=float(np.max(np.linalg.norm(fine_value-coarse_value,axis=1)/np.linalg.norm(fine_value,axis=1)))
        report['trajectory_field_refinement_relative_difference']=quad_change
        report['trajectory_maximum_error_estimate']=validation.maximum_error_estimate
        if quad_change>1e-3:raise ValueError('Field quadrature is unresolved on reference trace')
        _,clearance=domain.classify(xyz)
        report['minimum_sampled_section_clearance_m']=float(clearance.min())
        rlo,rhi=float(rz[:,0].min()-.015),float(rz[:,0].max()+.015)
        zlo,zhi=float(rz[:,1].min()-.015),float(rz[:,1].max()+.015)
        if zlo<=zbound:raise ValueError('Grid box not wholly exterior')
        np.savez(args.output/'reference.npz',phi=phi,RZ=rz,B=fine_value)
        previous=None
        for n,nphi in ((5,32),(9,64),(13,128)):
            print(f'Building exterior grid {n}x{n}x{nphi}',flush=True)
            grid,gcheck=field.to_pyna_grid(np.linspace(rlo,rhi,n),np.linspace(zlo,zhi,n),
                np.arange(nphi)*2*np.pi/(3*nphi),nfp=3,batch_size=64)
            np.savez(args.output/f'grid-{n}-{nphi}.npz',R=grid.R,Z=grid.Z,Phi=grid.Phi,
                BR=grid.BR,BZ=grid.BZ,BPhi=grid.BPhi)
            mapped=[];last_error=None
            for steps in (32,64):
                dt=(phi_end-phi_start)/steps
                rr,zz,pp,_,alive=trace_orbit_along_phi_field(grid,*seed,phi_start,phi_end,dt,dphi_out=dt)
                rr,zz,pp=np.asarray(rr),np.asarray(zz),np.asarray(pp)
                if not np.all(alive) or abs(pp[-1]-phi_end)>1e-10:raise ValueError('Incomplete native trace')
                trajectory=np.column_stack((rr,zz));target=ref.sol(pp).T
                last_error=float(np.max(np.linalg.norm(trajectory-target,axis=1)))
                mapped.append(trajectory[-1])
                coordinates=np.column_stack((rr*np.cos(pp),rr*np.sin(pp),zz))
                if not np.all(domain(coordinates)):raise ValueError('Native trace left exterior domain')
                report['rows'].append(dict(grid=[n,n,nphi],steps=steps,maximum_trace_error_m=last_error,
                    maximum_grid_error_estimate=gcheck.maximum_error_estimate))
                np.savez(args.output/f'trace-{n}-{steps}.npz',phi=pp,RZ=trajectory)
                save()
            step_change=float(np.linalg.norm(mapped[1]-mapped[0]))
            grid_change=None if previous is None else float(np.linalg.norm(mapped[1]-previous))
            report['rows'][-1].update(endpoint_step_difference_m=step_change,endpoint_grid_difference_m=grid_change)
            report['native_trace_passed']=bool(previous is not None and last_error<1e-4 and step_change<1e-6 and grid_change<1e-4)
            previous=mapped[1];save()
            if report['native_trace_passed']:break
        return 0 if report['native_trace_passed'] else 2
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';save();raise

if __name__=='__main__':raise SystemExit(main())
