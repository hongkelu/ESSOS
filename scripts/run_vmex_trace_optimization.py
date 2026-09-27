#!/usr/bin/env python3
"""Re-solved finite-beta current-control baseline with native trace acceptance.

Recover a nearby known common-current factor from a fixed-launch, fixed-phase
endpoint target. Finite-difference gradients are the baseline comparator. Bounds
and targets are numerical; this is not a wall/heat-load engineering design.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equilibrium-run',type=Path,required=True)
    parser.add_argument('--qualified-trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    shutil.copyfile(__file__,args.output/'optimization_source.py')
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vmex as vm
    from vmex.core import virtual_casing as vc
    from vmex.core.extender import VmecExtender
    from essos.fields import BiotSavart
    from essos.equilibrium_response import EquilibriumDefinition
    from essos.vmex_field import VmexExteriorField
    from essos.topology_optimizer import DesignProblem,Evaluation,optimize_topology
    from pyna.toroidal.flt import trace_orbit_along_phi_field
    import vmex_accepted_case as loader
    import vmex_radial_continuation as radial
    shutil.copyfile(loader.__file__,args.output/'loader_source.py')
    shutil.copyfile(radial.__file__,args.output/'continuation_source.py')
    if not jax.config.x64_enabled:raise ValueError('Float64 required')
    case=loader.load_accepted_case(args.equilibrium_run)
    qualified=json.loads((args.qualified_trace/'report.json').read_text())
    if not qualified['native_trace_passed'] or qualified['checkpoint_sha256']!=case.checkpoint_sha256:
        raise ValueError('Matching independently qualified native trace required')
    seed=np.asarray(qualified['seed_RZ_m']);phi_start,phi_end=qualified['phi_interval']
    grids={}
    for key,name in [('coarse','grid-9-64.npz'),('fine','grid-13-128.npz')]:
        with np.load(args.qualified_trace/name,allow_pickle=False) as d:
            grids[key]=tuple(d[a].copy() for a in ('R','Z','Phi'))
    g=case.report['field_grid']
    print('Preparing external coil field and radial ladder',flush=True)
    external=vm.MgridField.from_coils(case.coils,rmin=g['R'][0],rmax=g['R'][1],zmin=g['Z'][0],zmax=g['Z'][1],
        ir=g['R'][2],jz=g['Z'][2],kp=g['nphi'],nfp=g['nfp'])
    inp=replace(case.inp,niter_array=[3000],ftol_array=[1e-12])
    continuation=radial.RadialContinuation(inp,lambda c:replace(external,extcur=jnp.asarray(c)),
        [16,24,32],1e-12,3000,args.output)
    direct=jax.jit(jax.vmap(BiotSavart(case.coils).B))
    definition=EquilibriumDefinition('LI383 re-solved common-current endpoint control','free_boundary',
        ('common_NCSX_current_factor',),('pressure_profile','plasma_current','toroidal_flux','launch','terminal_phi'),
        'grid box wholly above candidate LCFS height bound')
    scale=1e-5;desired=np.array([1.0003]);evaluations=[];start=time.perf_counter()
    report=dict(case='LI383 finite-beta exterior endpoint recovery',method='re_solved_finite_difference',
        field_mode='self_consistent_equilibrium',response_policy='User accepted existing approximate implicit field response; this baseline uses independent re-solves.',
        controls=['common_NCSX_current_factor'],bounds=[.999,1.001],bounds_kind='numerical',target_current_factor=float(desired[0]),
        launch_RZ_m=seed.tolist(),phi_interval=[phi_start,phi_end],residual_scale_m=scale,
        qualified_trace_sha256=hashlib.sha256((args.qualified_trace/'report.json').read_bytes()).hexdigest(),
        physical_design_qualified=False,evaluations=evaluations)
    def save():
        report['elapsed_seconds']=time.perf_counter()-start
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def endpoint(c,*,refined=False,purpose='evaluation'):
        factor=float(np.asarray(c)[0])
        print(f'{purpose}: independently solving current factor {factor:.10f}',flush=True)
        stage=continuation.solve([factor],fresh=True);result=stage.result
        out=vm.wout_from_state(inp=inp,state=result.state,converged=True,
            fsqr=result.fsqr,fsqz=result.fsqz,fsql=result.fsql,niter=result.iterations,vacuum_output=result.vacuum)
        zbound=float(np.sum(np.abs(np.asarray(out.zmns)[-1])))
        R,Z,Phi=grids['fine' if refined else 'coarse']
        if Z[0]<=zbound:raise ValueError('Candidate grid intersects conservative LCFS bound')
        data=vc.surface_field_data_from_state(inp,result.state,nphi=96,ntheta=96)
        backend=VmecExtender.from_surface_data(data,external_field=lambda x:factor*direct(x),
            digits=5,levels=((192,96),(384,192)),accuracy_check='off')
        field=VmexExteriorField(backend,definition=definition,domain_check=lambda x:x[:,2]>zbound,
            equilibrium_valid=True,maximum_error=1e-3)
        grid,validation=field.to_pyna_grid(R,Z,Phi,nfp=3,batch_size=64)
        dt=(phi_end-phi_start)/(128 if refined else 64)
        rr,zz,pp,_,alive=trace_orbit_along_phi_field(grid,*seed,phi_start,phi_end,dt,dphi_out=dt)
        rr,zz,pp=np.asarray(rr),np.asarray(zz),np.asarray(pp)
        if not np.all(alive) or abs(pp[-1]-phi_end)>1e-10 or np.min(zz)<=zbound:
            raise ValueError('Candidate native trajectory failed acceptance')
        answer=np.array([rr[-1],zz[-1]])
        record=dict(current_factor=factor,refined=refined,purpose=purpose,endpoint_RZ_m=answer.tolist(),
            beta=float(out.betatotal),maximum_force_residual=max(float(result.fsqr),float(result.fsqz),float(result.fsql),float(result.fedge)),
            maximum_field_error_estimate=validation.maximum_error_estimate,
            minimum_Z_clearance_bound_m=float(np.min(zz)-zbound),equilibrium_and_trace_valid=True)
        evaluations.append(record)
        np.savez(args.output/f'trajectory-{len(evaluations):02d}.npz',R=rr,Z=zz,phi=pp)
        save()
        # The backend owns potentially large compiled quadrature closures. They
        # need not persist across candidate snapshots in this reduced baseline.
        return answer,record
    try:
        target,_=endpoint(desired,purpose='independent numerical target')
        report['target_RZ_m']=target.tolist();save()
        def refresh(c,snapshot):
            try:
                value,record=endpoint(c,purpose='candidate refresh')
                return Evaluation((value-target)/scale,snapshot={'current_factor':float(c[0]),'endpoint':value.tolist()},diagnostics=record)
            except ValueError as exc:
                return Evaluation(np.ones(2),valid=False,status=str(exc))
        def model(c,snapshot):
            value=refresh(c,snapshot)
            if not value.valid:return value
            h=1e-4
            plus,_=endpoint(c+h,purpose='finite-difference plus')
            minus,_=endpoint(c-h,purpose='finite-difference minus')
            value.jacobian=((plus-minus)/(2*h*scale))[:,None]
            return value
        identity=hashlib.sha256(json.dumps({'equilibrium':case.checkpoint_sha256,
            'qualified_trace':report['qualified_trace_sha256'],'target':target.tolist(),
            'method':report['method']},sort_keys=True).encode()).hexdigest()
        problem=DesignProblem('LI383 finite-beta endpoint recovery v1 '+identity,np.ones(1),np.array([.001]),np.array([.999]),np.array([1.001]))
        initial=refresh(np.ones(1),{})
        if not initial.valid:raise ValueError(initial.status)
        result=optimize_topology(problem,np.ones(1),model,refresh,max_iterations=4,initial_radius=.5,
            numerical_floor=1e-8,checkpoint=args.output/'optimizer-checkpoint.json')
        c=problem.physical(result.state.q)
        final,final_record=endpoint(c,refined=True,purpose='independent final grid refinement')
        target_fine,_=endpoint(desired,refined=True,purpose='independent target grid refinement')
        error=float(np.linalg.norm(final-target_fine))
        report.update(status=result.status,final_current_factor=float(c[0]),initial_objective=initial.objective,
            final_objective=result.state.objective,accepted_iterations=result.state.iteration,trials=list(result.trials),
            refined_recovery_error_m=error,refined_valid=True,
            numerical_recovery_passed=bool(error<1e-7 and result.state.objective<initial.objective and result.state.iteration>0),
            limitation='Known-control recovery on a short exterior trajectory. It is not a wall-target design, physical position accuracy, or a qualification of implicit trace gradients.')
        save();return 0 if report['numerical_recovery_passed'] else 2
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';save();raise

if __name__=='__main__':raise SystemExit(main())
