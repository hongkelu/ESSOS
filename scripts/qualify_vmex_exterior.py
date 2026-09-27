#!/usr/bin/env python3
"""Qualify VMEX exterior response at fixed Cartesian points, not moving labels.

This fixed-boundary diagnostic does not qualify a free-boundary coil design.
It holds the declared external field fixed while varying a boundary coefficient.
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
    parser.add_argument('--case',choices=('li383','tokamak'),default='li383')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ns',type=int,default=16)
    parser.add_argument('--response-tolerance',type=float,default=1e-3)
    parser.add_argument('--response',action='store_true')
    parser.add_argument('--frozen-diagnostic',action='store_true')
    parser.add_argument('--force-tolerance',type=float,default=1e-16)
    args=parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import vmex as vm
    from vmex import optimize as opt
    from vmex.core.extender import VmecExtender
    from vmex.core import virtual_casing as vc
    from essos.vmex_field import VmexExteriorField
    from essos.equilibrium_response import EquilibriumDefinition
    if not jax.config.x64_enabled:raise ValueError('Float64 required')
    start=time.perf_counter()
    source=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization'/f'{args.case}.vmec'
    inp=replace(vm.VmecInput.from_file(source),ns_array=[args.ns],niter_array=[20000],ftol_array=[args.force_tolerance])
    report=dict(case=args.case,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),force_tolerance=args.force_tolerance,
        vmex_reference_commit='345f940c56264dc2a66c95b0201330283cc5241d',
        versions={name:importlib.metadata.version(name) for name in ['vmex','virtual-casing-jax','jax','scipy','numpy']},
        coordinate_convention='fixed_cartesian_positions',response_semantics='VMEX implicit residual with frozen solver DOFs; requires independent re-solve qualification',boundary_model='fixed_boundary',response_tolerance=args.response_tolerance,
        equilibrium_coil_consistency_qualified=False,
        limitation='Boundary-control diagnostic with fixed external field; no free-boundary coil-response or design-acceptance claim')
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    runtime_root=Path(vm.__file__).resolve().parent
    report['vmex_runtime_sources']={str(path.relative_to(runtime_root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(runtime_root.rglob('*.py'))}
    shutil.copyfile(__file__,args.output/'qualification_source.py')
    import essos.vmex_field as adapter_module
    shutil.copyfile(adapter_module.__file__,args.output/'adapter_source.py')
    def save():
        report['elapsed_seconds']=time.perf_counter()-start
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        print('Solving VMEX equilibrium',flush=True)
        eq=opt.solve_equilibrium(inp,raise_on_max_iterations=True)
        force=[float(getattr(eq.result,name)) for name in ('fsqr','fsqz','fsql')]
        report.update(primal_converged=bool(eq.result.converged),force_residuals=force,volume_average_beta=float(eq.wout.betatotal))
        save()
        if not eq.result.converged or not np.all(np.isfinite(force)) or max(force)>args.force_tolerance:
            raise ValueError('Equilibrium failed convergence gate')
        data=vc.surface_field_data_from_state(inp,eq.state,runtime=eq.runtime,nphi=32,ntheta=32)
        gamma=np.asarray(data.gamma).reshape(3,-1).T
        # Entire query box is radially beyond the maximum boundary radius.
        # This conservative bound avoids an approximate LCFS classifier here.
        maximum_radius=float(np.max(np.hypot(gamma[:,0],gamma[:,1])))
        # Sampled maximum is diagnostic, not a continuous-surface bound. Add a
        # Fourier l1 bound, which is rigorous for this boundary representation.
        radius_bound=float(np.sum(np.abs(np.asarray(inp.rbc)))+np.sum(np.abs(np.asarray(inp.rbs))))
        radius=radius_bound+.6
        points=np.array([[radius,0.,.15],[radius*np.cos(.3),radius*np.sin(.3),-.1]])
        np.save(args.output/'points.npy',points)
        if args.case=='li383':
            from essos.coil_inputs import load_simsopt_xyz_coils
            from essos.fields import BiotSavart
            coil_source=source.parent/'ncsx_coils.json'
            coils=load_simsopt_xyz_coils(coil_source,n_segments=128)
            external=jax.jit(jax.vmap(BiotSavart(coils).B))
            report['external_field']='packaged NCSX modular coils, fixed; matching equilibrium not established'
            report['coil_sha256']=hashlib.sha256(coil_source.read_bytes()).hexdigest()
        else:
            def external(x):
                return jnp.stack((-x[:,1],x[:,0],jnp.zeros(len(x))),axis=1)/jnp.sum(x[:,:2]**2,axis=1)[:,None]
            report['external_field']='declared analytic toroidal field Bphi=1/R T, fixed; matching equilibrium not established'
        definition=EquilibriumDefinition('VMEX '+args.case,'fixed_boundary',('boundary_diagnostic',),
            ('profiles','current','toroidal_flux','external_coils'),f'R > {radius_bound} m')
        def adapted(backend,definition=definition):
            backend.accuracy_check='off'
            return VmexExteriorField(backend,definition=definition,
                domain_check=lambda x:np.hypot(x[:,0],x[:,1])>radius_bound,
                equilibrium_valid=True,maximum_error=1e-5)
        print('Evaluating exterior field and quadrature refinement',flush=True)
        values=[];rows=[]
        for n in (24,32,48):
            backend=VmecExtender.from_equilibrium(eq,external_field=external,nphi=n,ntheta=n,
                digits=5,levels=((4*n,2*n),(8*n,4*n)))
            value,validation=adapted(backend).sample(points)
            values.append(value);rows.append(dict(source_resolution=n,error_estimate=validation.maximum_error_estimate))
        refinement=float(np.linalg.norm(values[-1]-values[-2])/max(np.linalg.norm(values[-1]),1e-30))
        report.update(surface_radius_sampled=maximum_radius,surface_radius_bound=radius_bound,
            field_refinement=rows,relative_field_refinement=refinement,
            plasma_field_norms=np.linalg.norm(values[-1]-np.asarray(external(points)),axis=1).tolist(),
            field_qualified_at_sample_points=refinement<1e-4)
        np.savez(args.output/'fields.npz',points=points,external=np.asarray(external(points)),
            total_24=values[0],total_32=values[1],total_48=values[2])
        save()
        if not args.response:return 0 if report['field_qualified_at_sample_points'] else 2
        print('Constructing implicit equilibrium response',flush=True)
        problem=opt.make_problem(inp,loss=lambda state,rt:opt.mean_iota(state,rt),max_mode=1,
            use_ess=False,restart_from=eq,hot_restart=False,refine_tol=args.force_tolerance,
            adjoint_tol=1e-9,adjoint_maxiter=500)
        names=tuple(problem.dof_names);report['control_names']=names;save()
        index=next(i for i,name in enumerate(names) if name.lower().replace(' ','') == 'rbc(0,1)')
        direction=np.zeros_like(problem.x0);direction[index]=1.
        report['tested_control']=names[index]
        backend=problem.exterior_field(problem.x0,external_field=external,nphi=32,ntheta=32,
            digits=5,levels=((128,64),(256,128)))
        definition=replace(definition,controls=names)
        field=adapted(backend,definition)
        value=field.B(points)
        report['base_field_relative_difference']=float(np.linalg.norm(value-values[1])/max(np.linalg.norm(values[1]),1e-30))
        if report['base_field_relative_difference']>1e-4:raise ValueError('Response base field differs from independently solved field')
        response=field.jvp(points,direction)
        cotangent=np.arange(value.size).reshape(value.shape)/value.size
        report['transpose_error']=float(abs(np.vdot(cotangent,response)-np.vdot(direction,field.vjp(points,cotangent))))
        print('Adjoint response evaluated',flush=True);save()
        rows=[]
        for h in (1e-3,3e-4,1e-4):
            print(f'Independent exterior finite difference {h}',flush=True)
            pair=[];forces=[]
            for sign in (1,-1):
                deck=problem.input_from_x(problem.x0+sign*h*direction)
                independent=opt.solve_equilibrium(deck,raise_on_max_iterations=True)
                force=[float(getattr(independent.result,key)) for key in ('fsqr','fsqz','fsql')]
                forces.append(force)
                if not independent.result.converged or not np.all(np.isfinite(force)) or max(force)>args.force_tolerance:raise ValueError('Independent equilibrium failed')
                fresh=VmecExtender.from_equilibrium(independent,external_field=external,nphi=32,ntheta=32,
                    digits=5,levels=((128,64),(256,128)))
                # Recompute the conservative domain for the changed boundary.
                bound=float(np.sum(np.abs(np.asarray(deck.rbc)))+np.sum(np.abs(np.asarray(deck.rbs))))
                if np.any(np.hypot(points[:,0],points[:,1])<=bound):raise ValueError('FD query leaves exterior domain')
                pair.append(adapted(fresh).B(points))
            fd=(pair[0]-pair[1])/(2*h)
            rows.append(dict(step=h,relative_error=float(np.linalg.norm(response-fd)/max(np.linalg.norm(fd),1e-30)),force_residuals=forces))
            np.savez(args.output/f'fd-{h:.0e}.npz',plus=pair[0],minus=pair[1],response=response,finite_difference=fd)
            report['finite_differences']=rows;save()
        passed=[row['relative_error']<args.response_tolerance for row in rows]
        report['response_qualified']=bool(report['field_qualified_at_sample_points'] and any(a and b for a,b in zip(passed,passed[1:])) and report['transpose_error']<1e-8)
        save()
        if args.frozen_diagnostic:
            print('Checking VMEX frozen-solver-path scalar response',flush=True)
            from vmex.core import implicit as im
            cfg=problem.metadata['config']
            params=im.params_from_input(problem.input_from_x(problem.x0))
            plus=im.params_from_input(problem.input_from_x(problem.x0+direction))
            minus=im.params_from_input(problem.input_from_x(problem.x0-direction))
            tangent=jax.tree.map(lambda a,b:(a-b)/2,plus,minus)
            def metric(state,runtime):
                data=vc.surface_field_data_from_state(inp,state,runtime=runtime,nphi=32,ntheta=32)
                backend=VmecExtender.from_surface_data(data,external_field=external,
                    digits=5,levels=((128,64),(256,128)),accuracy_check='off')
                return jnp.vdot(backend.B(jnp.asarray(points)),jnp.asarray(cotangent))
            frozen,diagnostics=im.frozen_path_directional_fd(params,cfg,metric,tangent,
                h=1e-4,newton_rtol=1e-13,newton_steps=20)
            adjoint=float(np.vdot(response,cotangent))
            report['frozen_path_diagnostic']=dict(adjoint=adjoint,finite_difference=frozen,
                relative_error=abs(adjoint-frozen)/max(abs(frozen),1e-30),**diagnostics,
                qualification_scope='internal frozen-residual consistency only; does not replace ordinary re-solves')
            save()
        return 0 if report['response_qualified'] else 2
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';save();raise


if __name__=='__main__':raise SystemExit(main())
