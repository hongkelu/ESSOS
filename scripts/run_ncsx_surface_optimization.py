#!/usr/bin/env python3
"""Constrained NCSX current-ratio control of a fixed-area invariant circle.

Bounds are numerical benchmark limits, not NCSX engineering specifications.
The surface target uses the map rotation per physical toroidal span. The final
state is independently corrected with more Fourier modes, map steps, and coil
quadrature points. A finite-difference comparator uses re-corrected surfaces.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualified-circle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=['implicit','finite_difference','direct'],default='implicit')
    args=parser.parse_args()
    if os.environ.get('NERSC_HOST')=='perlmutter':
        from run_manifold_baseline import require_compute_allocation
        require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    start=time.perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coil_inputs import load_simsopt_xyz_coils
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.surface_optimization import CircleObjective
    from essos.topology_optimizer import DesignProblem,Evaluation,optimize_topology
    from pyna.topo.torus_solver import InvariantCircleProblem,interpolate
    from pyna.toroidal.flt.jax_poincare import poincare_map
    report=json.loads((args.qualified_circle/'report.json').read_text())
    if not report['valid']:raise ValueError('Input circle must be qualified')
    data=np.load(args.qualified_circle/'circle.npz');reference=data['embedding'];span=report['map_span']
    source=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization/ncsx_coils.json'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=report['input_sha256']:raise ValueError('Coil source differs from qualified circle')
    def make_problem(nodes,steps,segments,area=None):
        coils=load_simsopt_xyz_coils(source,n_segments=segments);currents=jnp.asarray(coils.currents);curves=coils.curves
        def field(x,c):return BiotSavart(Coils(curves,currents*jnp.tile(c,6),currents_scale=1e6)).B(x)
        def mapping(x,c):return poincare_map(field,c,x,phi_span=span,n_steps=steps,bphi_floor=1e-5)
        ref=np.asarray(interpolate(reference,np.arange(nodes)*2*np.pi/nodes))
        return InvariantCircleProblem(mapping,ref,area=report['area'] if area is None else area,tolerance=1e-9,dense_tolerance=2e-6)
    circle=make_problem(report['nodes'],report['steps'],report['coil_quadrature'])
    target=report['signed_iota']+.005
    objective=CircleObjective(circle,lambda z,c:jnp.array([z[-2]/span]),(target,),(.005,),report['omega'],'ncsx-fixed-section-area-transform')
    solve_count={'model':0,'refresh':0,'optimization_corrector_solves':0,'direct_residual_calls':0}
    def constrained(e,c):
        e.equalities=np.array([np.sum(c)-3.]);e.equality_jacobian=np.ones((1,3));return e
    def refresh(c,s):
        solve_count['refresh']+=1;solve_count['optimization_corrector_solves']+=1;return constrained(objective.refresh(c,s),c)
    def model(c,s):
        solve_count['model']+=1
        if args.method=='implicit':
            solve_count['optimization_corrector_solves']+=1
            return constrained(objective.model(c,s),c)
        solve_count['optimization_corrector_solves']+=7
        result=objective.refresh(c,s);columns=[];eps=1e-4
        for direction in np.eye(3):
            plus=objective.refresh(c+eps*direction,s);minus=objective.refresh(c-eps*direction,s)
            if not plus.valid or not minus.valid:result.valid=False;result.status='finite_difference_surface_failed';return result
            columns.append((plus.residual-minus.residual)/(2*eps))
        result.jacobian=np.column_stack(columns)
        return constrained(result,c)
    initial=refresh(np.ones(3),{})
    # Independent re-solved directional differences at several amplitudes.
    local=objective.model(np.ones(3),initial.snapshot);direction=np.array([.3,-.2,-.1]);ad=float((local.jacobian@direction)[0])
    derivative_checks=[]
    for eps in (1e-3,3e-4,1e-4):
        plus=objective.refresh(np.ones(3)+eps*direction,initial.snapshot);minus=objective.refresh(np.ones(3)-eps*direction,initial.snapshot)
        fd=float(((plus.residual-minus.residual)/(2*eps))[0]);derivative_checks.append(dict(step=eps,implicit=ad,finite_difference=fd,absolute_error=abs(ad-fd),valid=plus.valid and minus.valid))
    problem=DesignProblem(objective.fingerprint,np.ones(3),np.full(3,.05),np.full(3,.95),np.full(3,1.05))
    solve_count.update(model=0,refresh=0,optimization_corrector_solves=0)
    optimization_start=time.perf_counter()
    if args.method=='direct':
        from scipy.optimize import minimize
        from types import SimpleNamespace
        z0=np.asarray(initial.snapshot['unknowns']);nz=len(z0)
        def equalities(v):
            return jnp.concatenate((circle.residual(v[:nz],v[nz:]),jnp.array([jnp.sum(v[nz:])-3.])))
        eq_jit=jax.jit(equalities);eq_jac=jax.jit(jax.jacfwd(equalities))
        def eq(v):
            solve_count['direct_residual_calls']+=1;return np.asarray(eq_jit(v))
        def merit(v):return .5*((v[nz-2]/span-target)/.005)**2
        def gradient(v):
            g=np.zeros_like(v);g[nz-2]=(v[nz-2]/span-target)/(.005**2*span);return g
        direct=minimize(merit,np.concatenate((z0,np.ones(3))),jac=gradient,
            constraints=[dict(type='eq',fun=eq,jac=lambda v:np.asarray(eq_jac(v)))],
            bounds=[(None,None)]*nz+[(.95,1.05)]*3,method='SLSQP',
            options={'ftol':1e-12,'maxiter':200})
        if not direct.success or np.max(np.abs(eq(direct.x)))>1e-8:raise ValueError('Direct surface comparator failed: '+direct.message)
        parameters=direct.x[nz:]
        final_coarse=refresh(parameters,dict(initial.snapshot,unknowns=direct.x[:nz].tolist(),omega=float(direct.x[nz-2])))
        if not final_coarse.valid:raise ValueError('Direct comparator failed independent correction')
        result=SimpleNamespace(status='direct_surface_converged',trials=(),
            state=SimpleNamespace(objective=final_coarse.objective,snapshot=final_coarse.snapshot))
    else:
        result=optimize_topology(problem,np.ones(3),model,refresh,snapshot=initial.snapshot,max_iterations=12,
                                 initial_radius=.25,numerical_floor=1e-9,checkpoint=args.output/'checkpoint.json')
        parameters=problem.physical(result.state.q)
    optimization_seconds=time.perf_counter()-optimization_start
    refined=make_problem(2*report['nodes']-1,2*report['steps'],2*report['coil_quadrature'])
    final=refined.solve(parameters,omega=result.state.snapshot['omega'])
    from pyna.topo.circle_validation import validate_circle_nesting
    neighbors=[]
    for area_factor in (.9,1.1):
        neighbor=make_problem(2*report['nodes']-1,2*report['steps'],2*report['coil_quadrature'],area=area_factor*report['area'])
        neighbors.append(neighbor.solve(parameters,omega=final.omega))
    nesting=validate_circle_nesting([neighbors[0],final,neighbors[1]])
    np.savez(args.output/'final-circle.npz',embedding=final.embedding,parameters=parameters)
    output=dict(case='NCSX modular-coil vacuum surface control',method=args.method,status=result.status,
                input_sha256=report['input_sha256'],field_mode='vacuum_coils',label='signed_section_area',area=report['area'],
                initial_objective=initial.objective,final_objective=result.state.objective,target_iota=target,
                final_iota=final.omega/span,refined_valid=final.valid,refined_status=final.status,
                refined_invariance_residual=final.dense_residual,parameters=parameters.tolist(),
                nesting_valid=nesting.valid,nesting_status=nesting.status,neighbor_section_areas=nesting.section_areas,minimum_section_separation_m=nesting.minimum_separation,
                current_bounds=[.95,1.05],sum_current_factors=3.,bounds_kind='numerical benchmark bounds',
                derivative_checks=derivative_checks,trials=list(result.trials),calls=solve_count,
                optimization_seconds=optimization_seconds,diagnostic_corrector_solves=8,refinement_corrector_solves=3,
                compilation_policy="cold process; diagnostic solves compile shared kernels before the optimization timer; total includes setup, diagnostics, optimization and refinement",
                elapsed_seconds=time.perf_counter()-start,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    (args.output/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2),flush=True)
    return 0 if final.valid and nesting.valid and abs(final.omega/span-target)<1e-6 and result.state.objective<initial.objective else 2


if __name__=='__main__':raise SystemExit(main())
