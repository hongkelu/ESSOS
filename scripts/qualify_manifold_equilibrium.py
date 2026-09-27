#!/usr/bin/env python3
"""Qualify accessible tokamak/LI383 inputs with an actual fixed-boundary solve.

This command measures primal convergence; it does not claim coil response or
self-consistent external/open-region fields from a fixed-boundary equilibrium.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver-root',type=Path,required=True)
    parser.add_argument('--case',choices=['tokamak','li383'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-iterations',type=int,default=500)
    args=parser.parse_args()
    if os.environ.get('NERSC_HOST')=='perlmutter':
        from run_manifold_baseline import require_compute_allocation
        require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output,external_repositories=(args.solver_root,))
    sys.path.insert(0,str(args.solver_root.resolve()))
    # This backend recognizes the literal 1, rather than JAX's additional true spelling.
    os.environ["JAX_ENABLE_X64"]="1"
    import numpy as np
    import vmec_jax
    import jax
    if not jax.config.x64_enabled:raise RuntimeError("Equilibrium qualification requires float64")
    from vmec_jax.driver import run_fixed_boundary,write_wout_from_fixed_boundary_run
    source=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization'/f'{args.case}.vmec'
    (args.output/'input.vmec').write_bytes(source.read_bytes())
    revision=subprocess.check_output(['git','-C',str(args.solver_root),'rev-parse','HEAD'],text=True).strip()
    start=time.perf_counter()
    run=run_fixed_boundary(source,max_iter=args.max_iterations,ns_override=16,verbose=False,multigrid=False)
    histories={key:np.asarray(getattr(run.result,key)).tolist() for key in ('fsqr2_history','fsqz2_history','fsql2_history')}
    converged=bool(run.result.diagnostics.get('converged',False))
    output=dict(schema_version=1,case=args.case,field_mode='self_consistent_fixed_boundary_equilibrium',
                controls='input profiles and boundary fixed; no coil controls in this qualification',
                solver_revision=revision,solver_file=str(Path(vmec_jax.__file__).resolve()),
                input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),converged=converged,
                iterations=int(run.result.n_iter),total_history_samples=len(histories["fsqr2_history"]),
                jax_enable_x64=bool(jax.config.x64_enabled),state_dtype=str(run.state.Rcos.dtype),elapsed_seconds=time.perf_counter()-start,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),histories=histories,
                resolution=dict(ns=int(run.cfg.ns),mpol=int(run.cfg.mpol),ntor=int(run.cfg.ntor),nfp=int(run.cfg.nfp)))
    (args.output/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    write_wout_from_fixed_boundary_run(args.output/'wout.nc',run)
    print(json.dumps({key:value for key,value in output.items() if key!='histories'},indent=2),flush=True)
    return 0 if converged else 2


if __name__=='__main__':raise SystemExit(main())
