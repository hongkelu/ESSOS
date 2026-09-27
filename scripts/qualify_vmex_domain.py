#!/usr/bin/env python3
"""Measure exterior-field refinement versus distance from an accepted LCFS.

Pointwise qualification only: this sweep does not certify unsampled volumes,
near-surface extrapolation, equilibrium response, or a production design.
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
    parser.add_argument('--field-tolerance',type=float,default=1e-3)
    args=parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    if not 0<args.field_tolerance<1:parser.error('Invalid field tolerance')
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    shutil.copyfile(__file__,args.output/'qualification_source.py')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from vmex.core import virtual_casing as vc
    from vmex.core.extender import VmecExtender
    from essos.fields import BiotSavart
    from essos.equilibrium import FourierLCFSExteriorDomain
    if not jax.config.x64_enabled:raise ValueError('Float64 required')
    start=time.perf_counter()
    import vmex_accepted_case as loader
    shutil.copyfile(loader.__file__,args.output/'loader_source.py')
    case=loader.load_accepted_case(args.equilibrium_run)
    source=case.report
    state,inp,wout=case.state,case.inp,case.wout
    row={'ns':source['radial_resolution'],'checkpoint_sha256':case.checkpoint_sha256}
    domain=FourierLCFSExteriorDomain.from_wout(wout)
    direct=jax.jit(jax.vmap(BiotSavart(case.coils).B))
    seed_data=vc.surface_field_data_from_state(inp,state,nphi=6,ntheta=12)
    gamma=np.asarray(seed_data.gamma).reshape(3,-1).T
    normal=np.asarray(seed_data.normal).reshape(3,-1).T
    distances=np.array([.005,.01,.02,.05,.1,.2])
    points=(gamma[None]+distances[:,None,None]*normal[None]).reshape(-1,3)
    outside,clearance=domain.classify(points)
    if not np.all(outside):raise ValueError('Normal offsets include interior or unresolved points')
    report=dict(case='LI383 exterior field domain refinement',equilibrium_run=str(args.equilibrium_run.resolve()),
        checkpoint_sha256=row['checkpoint_sha256'],radial_surfaces=row['ns'],field_tolerance=args.field_tolerance,
        normalization='local total field norm, floor 1e-12 T; backend estimate separately normalized by RMS surface field',
        derivative_policy='User accepted measured 0.17–0.23 percent discrepancy for proceeding; original strict reports preserved',
        physical_design_qualified=False,section_chord_error_bound_m=domain.chord_error_bound,levels=[],distances=[])
    def save():
        report['elapsed_seconds']=time.perf_counter()-start
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    values=[];estimates=[]
    for n,levels in ((48,((96,48),(192,96))),(96,((192,96),(384,192))),(144,((384,192),(768,384)))):
        print(f'Exterior field source={n}x{n}, quadrature={levels}',flush=True)
        data=vc.surface_field_data_from_state(inp,state,nphi=n,ntheta=n)
        backend=VmecExtender.from_surface_data(data,external_field=direct,digits=5,levels=levels,accuracy_check='off')
        value=[];error=[]
        for i in range(0,len(points),36):
            batch=jnp.asarray(points[i:i+36])
            value.append(np.asarray(backend.B(batch)));error.append(np.asarray(backend.B_error_estimate(batch)))
        value=np.concatenate(value);error=np.concatenate(error)
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(error)):raise ValueError('Nonfinite field/error')
        values.append(value);estimates.append(error)
        report['levels'].append(dict(source_nphi=n,source_ntheta=n,quadrature=levels,max_error_estimate=float(error.max())))
        np.savez(args.output/f'field-{n}.npz',points=points,B=value,error_estimate=error)
        save()
    scale=np.maximum(np.linalg.norm(values[-1],axis=1),1e-12)
    change1=np.linalg.norm(values[1]-values[0],axis=1)/scale
    change2=np.linalg.norm(values[2]-values[1],axis=1)/scale
    for i,distance in enumerate(distances):
        sl=slice(i*len(gamma),(i+1)*len(gamma))
        passed=bool(np.max(change2[sl])<=args.field_tolerance and np.max(estimates[-1][sl])<=args.field_tolerance)
        report['distances'].append(dict(normal_offset_m=float(distance),points=len(gamma),
            minimum_section_clearance_lower_bound_m=float(clearance[sl].min()),
            max_coarse_refined_relative_difference=float(change1[sl].max()),
            max_refined_finest_relative_difference=float(change2[sl].max()),
            max_finest_error_estimate=float(estimates[-1][sl].max()),sampled_ring_passed=passed))
    np.savez(args.output/'sampling.npz',points=points,normal=normal,boundary=gamma,
        distances=distances,section_clearance_lower_bound=clearance,coarse_change=change1,fine_change=change2)
    report['passing_sampled_offsets_m']=[r['normal_offset_m'] for r in report['distances'] if r['sampled_ring_passed']]
    save();print(json.dumps(report['distances'],indent=2),flush=True)
    return 0 if report['passing_sampled_offsets_m'] else 2

if __name__=='__main__':raise SystemExit(main())
