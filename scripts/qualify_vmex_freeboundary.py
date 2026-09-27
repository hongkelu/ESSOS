#!/usr/bin/env python3
"""Pilot LI383/NCSX free-boundary equilibrium, including edge convergence.

The seed boundary may move. The complete original pressure/current profiles
and toroidal flux are held fixed; coil matching is tested, not presumed.
A passing pilot still requires grid/coil/interface and response refinement.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=3000)
    parser.add_argument('--ns',type=int,default=16)
    parser.add_argument('--radial-ladder',type=int,nargs='+',help='Independent cold-to-fine continuation for every current factor; final rung must equal --ns')
    parser.add_argument('--grid',type=int,default=64)
    parser.add_argument('--planes',type=int,default=32)
    parser.add_argument('--coil-quadrature',type=int,default=128)
    parser.add_argument('--force-tolerance',type=float,default=1e-10)
    parser.add_argument('--response',action='store_true')
    args=parser.parse_args()
    if args.radial_ladder and args.radial_ladder[-1] != args.ns:
        parser.error('Final radial ladder rung must equal --ns')
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import vmex as vm
    from essos.coils import load_simsopt_xyz_coils
    from essos.fields import BiotSavart
    from essos.equilibrium import EquilibriumDefinition
    from essos.equilibrium import VmexExteriorField
    from vmex.core.extender import VmecExtender
    if not jax.config.x64_enabled:raise ValueError('Float64 required')
    shutil.copyfile(__file__,args.output/"qualification_source.py")
    import essos.equilibrium as adapter_module
    shutil.copyfile(adapter_module.__file__,args.output/"adapter_source.py")
    start=time.perf_counter()
    data=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization'
    report=dict(case='li383_ncsx_free_boundary_pilot',slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        versions={name:importlib.metadata.version(name) for name in ['vmex','virtual-casing-jax','jax','scipy']},
        input_sha256={name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in ['li383.vmec','ncsx_coils.json']},
        pressure_current_flux='unchanged from LI383 deck',include_edge_in_convergence=True,
        radial_resolution=args.ns,force_tolerance=args.force_tolerance,field_grid=dict(R=[.65,2.2,args.grid],Z=[-.85,.85,args.grid],nphi=args.planes,nfp=3),
        coil_quadrature=args.coil_quadrature,response_tolerance=1e-3,physical_design_qualified=False)
    runtime_root=Path(vm.__file__).resolve().parent
    report['vmex_runtime_sources']={str(path.relative_to(runtime_root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(runtime_root.rglob('*.py'))}
    def save():
        report['elapsed_seconds']=time.perf_counter()-start
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        coils=load_simsopt_xyz_coils(data/'ncsx_coils.json',n_segments=args.coil_quadrature)
        print('Tabulating NCSX external coil field',flush=True)
        external=vm.MgridField.from_coils(coils,rmin=.65,rmax=2.2,zmin=-.85,zmax=.85,
            ir=args.grid,jz=args.grid,kp=args.planes,nfp=3)
        inp=replace(vm.VmecInput.from_file(data/'li383.vmec'),lfreeb=True,
            mgrid_file='NCSX packaged coils, direct tabulation',nzeta=16,
            ns_array=[args.ns],niter_array=[args.iterations],ftol_array=[args.force_tolerance])
        print('Solving finite-pressure/current free-boundary equilibrium',flush=True)
        def field_from_current(c):return replace(external,extcur=jnp.asarray(c))
        continuation=None
        if args.radial_ladder:
            from vmex_radial_continuation import RadialContinuation
            import vmex_radial_continuation as continuation_module
            shutil.copyfile(continuation_module.__file__,args.output/'continuation_source.py')
            continuation=RadialContinuation(inp,field_from_current,args.radial_ladder,
                args.force_tolerance,args.iterations,args.output)
            report['radial_ladder']=args.radial_ladder
            result=continuation.solve(np.ones(1)).result
        else:
            result=vm.solve_free_boundary(inp,external_field=external,
                error_on_no_convergence=False,include_edge_in_convergence=True,edge_force_tolerance=args.force_tolerance)
        force=[float(getattr(result,name)) for name in ('fsqr','fsqz','fsql')]
        edge=float(result.fedge)
        finite=bool(np.all(np.isfinite(force+[edge])))
        report.update(primal_converged=bool(result.converged),iterations=int(result.iterations),
            force_residuals=force if finite else None,edge_residual=edge if finite else None)
        save()
        if not result.converged or not finite or max(force+[edge])>args.force_tolerance:return 2
        wout=vm.wout_from_state(inp=inp,state=result.state,fsqr=force[0],fsqz=force[1],fsql=force[2],
            niter=int(result.iterations),converged=True,vacuum_output=result.vacuum)
        report['volume_average_beta']=float(wout.betatotal)
        radius_bound=float(np.sum(np.abs(np.asarray(wout.rmnc)[-1])))
        points=np.array([[3.2,0.,.1]])
        if radius_bound>=3.2:raise ValueError('Exterior test point is not outside the boundary bound')
        direct=jax.jit(jax.vmap(BiotSavart(coils).B))
        backend=VmecExtender.from_wout(wout,external_field=direct,nphi=48,ntheta=48,
            digits=5,levels=((192,96),(384,192)),accuracy_check='off')
        definition=EquilibriumDefinition('LI383 NCSX pilot','free_boundary',('common_NCSX_current_factor',),
            ('pressure','current_profile','toroidal_flux'),f'R > {radius_bound}')
        field=VmexExteriorField(backend,definition=definition,equilibrium_valid=True,
            domain_check=lambda x:np.hypot(x[:,0],x[:,1])>radius_bound,maximum_error=1e-5)
        value,validation=field.sample(points)
        report['exterior_error_estimate']=validation.maximum_error_estimate
        report['primal_pilot_passed']=True
        np.savez(args.output/'exterior.npz',points=points,total=value,external=np.asarray(direct(points)))
        save()
        if not args.response:return 0
        print('Constructing coupled coil/equilibrium/exterior response',flush=True)
        from vmex.core import implicit as im
        from vmex.core import virtual_casing as vc
        parameters=im.params_from_input(inp)
        cfg=vm.make_free_boundary_config(inp,external,ns=args.ns,ftol=args.force_tolerance,
            max_iterations=args.iterations,adjoint_tol=1e-8,adjoint_solver='boundary_schur',
            include_edge_in_convergence=True,edge_force_tolerance=args.force_tolerance,
            field_from_parameters=field_from_current)
        runtime=im.runtime_from_params(parameters,cfg.implicit)
        report['primal_evaluation_policy']='fresh free-boundary configuration per surface evaluation; ordinary cold re-solve reference'
        continued_state=None if continuation is None else continuation.implicit_state(parameters,cfg)
        if continuation is not None:
            report['primal_evaluation_policy']='independent cold radial ladders per current; exact-control immutable accepted-root reuse for differentiation'
        def surface(c):
            # VMEX's free-boundary wrapper otherwise hot-restarts even at the
            # identical controls. A fresh identity-keyed config makes these
            # reference primal calls independent of call history, including
            # the separate value and derivative construction calls.
            evaluation_cfg=replace(cfg)
            if continued_state is None:
                state,status,_,_=vm.solve_free_boundary_implicit_status(parameters,c,evaluation_cfg)
            else:
                state,status=continued_state(c),0
            state=jax.tree.map(lambda a:jnp.where(status==0,a,jnp.full_like(a,jnp.nan)),state)
            return vc.surface_field_data_from_state(inp,state,runtime=runtime,nphi=48,ntheta=48)
        c=jnp.ones(1)
        if continued_state is None:
            state,status,fsq,ratio=vm.solve_free_boundary_implicit_status(parameters,c,cfg)
        else:
            state,status,fsq,ratio=continued_state(c),0,sum(force),sum(force)/args.force_tolerance
        report['implicit_status']=int(status);report['implicit_fsq']=float(fsq) if np.isfinite(float(fsq)) else None;save()
        if int(status)!=0 or not np.isfinite(float(fsq)):raise ValueError('Coupled implicit equilibrium rejected')
        response_backend=VmecExtender.from_parameterized_surface_data(surface,c,
            external_parameters=c,external_field_from_parameters=lambda u:lambda x:u[0]*direct(x),
            dof_names=('equilibrium_current_factor',),external_dof_names=('direct_current_factor',),
            digits=5,levels=((192,96),(384,192)),accuracy_check='off')
        coupled=VmexExteriorField(response_backend,definition=definition,equilibrium_valid=True,
            domain_check=lambda x:np.hypot(x[:,0],x[:,1])>radius_bound,maximum_error=1e-5,
            control_jacobian=np.ones((2,1)))
        report['implicit_primal_state_max_difference']=max(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(result.state)))
        # Match the live-state field definition used by the differentiated
        # backend. Keep the wout reconstruction discrepancy as a separate
        # representation check; it is not a derivative comparison.
        live_data=vc.surface_field_data_from_state(inp,result.state,runtime=runtime,nphi=48,ntheta=48)
        live=VmecExtender.from_surface_data(live_data,external_field=direct,
            digits=5,levels=((192,96),(384,192)),accuracy_check='off')
        live_validated=VmexExteriorField(live,definition=definition,equilibrium_valid=True,
            domain_check=lambda x:np.hypot(x[:,0],x[:,1])>radius_bound,maximum_error=1e-5)
        live_value=live_validated.B(points)
        report['wout_live_relative_difference']=float(np.linalg.norm(live_value-value)/max(np.linalg.norm(live_value),1e-30))
        report['base_field_relative_difference']=float(np.linalg.norm(coupled.B(points)-live_value)/max(np.linalg.norm(live_value),1e-30))
        if report['base_field_relative_difference']>1e-4:raise ValueError('Implicit and ordinary free-boundary base fields differ')
        response=coupled.jvp(points,[1.])
        report['response_controls']=['common_NCSX_current_factor']
        rows=[]
        for h in (1e-3,3e-4,1e-4):
            print(f'Independent free-boundary current response {h}',flush=True)
            pair=[];forces=[]
            for sign in (1,-1):
                factor=1+sign*h
                if continuation is None:
                    solved=vm.solve_free_boundary(inp,external_field=field_from_current(jnp.array([factor])),
                        error_on_no_convergence=False,include_edge_in_convergence=True,
                        edge_force_tolerance=args.force_tolerance)
                else:
                    solved=continuation.solve(np.array([factor]),fresh=True).result
                residuals=[float(getattr(solved,key)) for key in ('fsqr','fsqz','fsql','fedge')]
                forces.append(residuals)
                if not solved.converged or not np.all(np.isfinite(residuals)) or max(residuals)>args.force_tolerance:
                    raise ValueError('Perturbed free-boundary solve failed')
                out=vm.wout_from_state(inp=inp,state=solved.state,fsqr=residuals[0],fsqz=residuals[1],fsql=residuals[2],
                    niter=int(solved.iterations),converged=True,vacuum_output=solved.vacuum)
                bound=float(np.sum(np.abs(np.asarray(out.rmnc)[-1])))
                data=vc.surface_field_data_from_state(inp,solved.state,runtime=runtime,nphi=48,ntheta=48)
                fresh=VmecExtender.from_surface_data(data,external_field=lambda x:factor*direct(x),
                    digits=5,levels=((192,96),(384,192)),accuracy_check='off')
                validated=VmexExteriorField(fresh,definition=definition,equilibrium_valid=True,
                    domain_check=lambda x:np.hypot(x[:,0],x[:,1])>bound,maximum_error=1e-5)
                pair.append(validated.B(points))
            fd=(pair[0]-pair[1])/(2*h)
            rows.append(dict(step=h,relative_error=float(np.linalg.norm(response-fd)/max(np.linalg.norm(fd),1e-30)),force_residuals=forces))
            np.savez(args.output/f'fd-{h:.0e}.npz',response=response,finite_difference=fd,plus=pair[0],minus=pair[1])
            report['finite_differences']=rows;save()
        passed=[row['relative_error']<1e-3 for row in rows]
        report['response_qualified']=any(a and b for a,b in zip(passed,passed[1:]))
        save();return 0 if report['response_qualified'] else 2
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';save();raise


if __name__=='__main__':raise SystemExit(main())
