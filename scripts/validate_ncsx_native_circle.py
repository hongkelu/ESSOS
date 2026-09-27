#!/usr/bin/env python3
"""Independent Cyna grid and step refinement for an optimized NCSX circle."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimization',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    import numpy as np
    from essos.coils import load_simsopt_xyz_coils
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.topology import essos_field_to_pyna_cylindrical_grid
    from pyna.toroidal.flt import trace_map_batch_span_field
    from pyna.topo.torus_solver import interpolate
    report=json.loads((args.optimization/'report.json').read_text())
    if not report['refined_valid']:raise ValueError('A refined valid circle is required')
    data=np.load(args.optimization/'final-circle.npz');embedding=data['embedding'];parameters=data['parameters']
    source=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization/ncsx_coils.json'
    coils=load_simsopt_xyz_coils(source,n_segments=256)
    field=BiotSavart(Coils(coils.curves,np.asarray(coils.currents)*np.tile(parameters,6),currents_scale=1e6))
    span=2*np.pi/3;theta=np.arange(2*len(embedding))*2*np.pi/(2*len(embedding))
    seeds=np.asarray(interpolate(embedding,theta));expected=np.asarray(interpolate(embedding,theta+span*report['final_iota']))
    rows=[];start=time.perf_counter()
    for n,np_phi in [(31,32),(61,64),(121,256),(181,384)]:
        print(f'Building native field grid {n} x {n} x {np_phi}',flush=True)
        begin=time.perf_counter()
        grid=essos_field_to_pyna_cylindrical_grid(field,np.linspace(1.15,2.05,n),np.linspace(-.55,.55,n),
                  np.linspace(0,span,np_phi,endpoint=False),nfp=3,batch_size=256)
        build_seconds=time.perf_counter()-begin
        previous_map=None
        for steps in (128,256,512,1024):
            begin=time.perf_counter()
            counts,r,z=trace_map_batch_span_field(grid,seeds[:,0],seeds[:,1],0.,span,1,span/steps,n_threads=1)
            complete=bool(np.all(np.asarray(counts)==1))
            error=float(np.max(np.linalg.norm(np.column_stack((r,z))-expected,axis=1))) if complete else None
            mapped=np.column_stack((r,z)) if complete else None
            step_difference=None if previous_map is None or mapped is None else float(np.max(np.linalg.norm(mapped-previous_map,axis=1)))
            previous_map=mapped
            rows.append(dict(step_refinement_difference_m=step_difference,grid=[n,n,np_phi],steps=steps,complete=complete,maximum_invariance_error_m=error,
                             grid_build_seconds=build_seconds,trace_seconds=time.perf_counter()-begin))
            print(rows[-1],flush=True)
        if rows[-1]['complete'] and rows[-1]['maximum_invariance_error_m']<1e-4 and rows[-1]['step_refinement_difference_m']<1e-5:break
    valid=all(r['complete'] for r in rows) and rows[-1]['maximum_invariance_error_m']<1e-4 and rows[-1]['step_refinement_difference_m']<1e-5
    output=dict(valid=valid,tolerance_m=1e-4,rows=rows,elapsed_seconds=time.perf_counter()-start,
                coil_sha256=report['input_sha256'],field_mode='vacuum_coils',slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    (args.output/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    return 0 if valid else 2


if __name__=='__main__':raise SystemExit(main())
