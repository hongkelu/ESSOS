#!/usr/bin/env python3
"""Compare VMEC material-grid field response with re-solved equilibria."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver-root',type=Path,required=True)
    parser.add_argument('--case',choices=['tokamak','li383'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=['discrete','implicit_residual'],default='discrete')
    parser.add_argument('--force-tolerance',type=float,default=1e-13)
    parser.add_argument('--response-tolerance',type=float,default=1e-3)
    args=parser.parse_args()
    if os.environ.get('NERSC_HOST')=='perlmutter':
        from run_manifold_baseline import require_compute_allocation
        require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output,external_repositories=(args.solver_root,))
    os.environ['JAX_ENABLE_X64']='1'
    sys.path.insert(0,str(args.solver_root.resolve()))
    import numpy as np
    from essos.equilibrium import VmecJaxMaterialField
    source=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization'/f'{args.case}.vmec'
    start=time.perf_counter();print('Building equilibrium response adapter',flush=True)
    adapter=VmecJaxMaterialField(source,args.output/'equilibria',controls=(('rc',1,0),),force_tolerance=args.force_tolerance,response_method=args.method)
    print(f'Solving primal and {args.method} response',flush=True)
    try:
        value=adapter.response(np.zeros(1));response=adapter.jvp(value,np.ones(1))
    except Exception as exc:
        failure=dict(case=args.case,response_method=args.method,response_qualified=False,
                     error=f'{type(exc).__name__}: {exc}',force_tolerance=args.force_tolerance,
                     elapsed_seconds=time.perf_counter()-start,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        (args.output/'report.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise
    rows=[]
    for step in [1e-3,3e-4,1e-4,3e-5,1e-5]:
        print(f'Re-solving finite difference {step}',flush=True)
        plus=adapter.solve_primal(np.array([step]));minus=adapter.solve_primal(np.array([-step]))
        fd=(plus.field-minus.field)/(2*step)
        error=float(np.linalg.norm(fd-response)/max(np.linalg.norm(fd),1e-30))
        magnitude=np.linalg.norm(value.field,axis=-1)
        mean_strength_response=float(np.mean(np.sum(value.field*response,axis=-1)/magnitude))
        mean_strength_fd=float((np.mean(np.linalg.norm(plus.field,axis=-1))-np.mean(np.linalg.norm(minus.field,axis=-1)))/(2*step))
        np.savez(args.output/f'finite-difference-{step:.0e}.npz',plus=plus.field,minus=minus.field,finite_difference=fd)
        rows.append(dict(step=step,relative_error=error,primal_converged=plus.converged and minus.converged,
                    mean_field_strength_response=mean_strength_response,mean_field_strength_finite_difference=mean_strength_fd,
                    mean_field_strength_relative_error=abs(mean_strength_response-mean_strength_fd)/max(abs(mean_strength_fd),1e-30)))
    cotangent=np.ones_like(value.field)
    transpose_error=float(abs(np.sum(cotangent*response)-adapter.vjp(value,cotangent)[0]))
    valid=all(row['primal_converged'] for row in rows) and min(row['relative_error'] for row in rows)<args.response_tolerance
    output=dict(case=args.case,response_method=args.method,coordinate_convention=value.coordinate_convention,parameter_names=adapter.parameter_names,
                response_qualified=valid,response_tolerance=args.response_tolerance,force_tolerance=args.force_tolerance,finite_differences=rows,transpose_error=transpose_error,
                force_residuals=value.force_residuals,elapsed_seconds=time.perf_counter()-start,
                limitation='fixed-boundary material-grid field response; no Eulerian/open-region or coil-response claim',
                slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    (args.output/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    np.savez(args.output/'response.npz',field=value.field,response=response)
    print(json.dumps(output,indent=2),flush=True)
    return 0 if valid else 2


if __name__=='__main__':raise SystemExit(main())
