#!/usr/bin/env python3
"""Optimize a labelled tokamak-regression manifold sample with Cyna acceptance."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('NERSC_HOST')=='perlmutter':
        from run_manifold_baseline import require_compute_allocation
        require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/manifold_optimization'))
    import numpy as np
    from tokamak_fixture import physical_field_controls,_tokamak_production_field,_tokamak_production_branch
    from essos.manifold_sample_objective import ManifoldSampleObjective
    from essos.topology_optimizer import DesignProblem,optimize_topology
    start=time.perf_counter()
    branch=_tokamak_production_branch(physical_field_controls(np.array([1.,-1.2])))
    label=branch.sample_label(branch.n_generations,0)
    desired=_tokamak_production_branch(physical_field_controls(np.array([1.0003,-1.199])))
    target=desired.point(label)
    objective=ManifoldSampleObjective(physical_field_controls,
                        lambda c:_tokamak_production_field(physical_field_controls(c)),branch,label,
                        tuple(target),(.001,.001),'tokamak-prescribed-background-sample')
    problem=DesignProblem(objective.fingerprint,[1.,-1.2],[.001,.005],[.998,-1.21],[1.002,-1.19])
    initial=objective.refresh(np.array([1.,-1.2]),{})
    result=optimize_topology(problem,[1.,-1.2],objective.model,objective.refresh,max_iterations=10,
                             initial_radius=.2,numerical_floor=1e-10,checkpoint=args.output/'checkpoint.json')
    report=dict(case='tokamak prescribed-background manifold sample',status=result.status,
                initial_objective=initial.objective,final_objective=result.state.objective,
                parameters=problem.physical(result.state.q).tolist(),trials=list(result.trials),
                target_RZ_m=target.tolist(),bounds_kind='numerical regression bounds',
                elapsed_seconds=time.perf_counter()-start,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if result.state.objective<initial.objective and result.state.iteration>0 else 2


if __name__=='__main__':raise SystemExit(main())
