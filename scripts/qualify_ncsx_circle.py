#!/usr/bin/env python3
"""Correct a small invariant circle in the accessible NCSX modular-coil field."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--nodes',type=int,default=15)
    parser.add_argument('--segments',type=int,default=128)
    args=parser.parse_args()
    if os.environ.get('NERSC_HOST')=='perlmutter':
        from run_manifold_baseline import require_compute_allocation
        require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    import faulthandler
    faulthandler.dump_traceback_later(120,repeat=True)
    print("Loading numerical and coil libraries",flush=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import load_simsopt_xyz_coils
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from pyna.topo.jax_periodic import periodic_point_state
    from pyna.topo.torus_solver import InvariantCircleProblem
    from pyna.toroidal.flt.jax_poincare import poincare_map
    source=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization/ncsx_coils.json'
    print("Reading supported coil graph",flush=True)
    coils=load_simsopt_xyz_coils(source,n_segments=args.segments)
    # Three independent current families, repeated in the file's verified order.
    currents=jnp.asarray(coils.currents);curves=coils.curves
    def field(x,c):return BiotSavart(Coils(curves,currents*jnp.tile(c,6),currents_scale=1e6)).B(x)
    parameters=jnp.ones(3);span=2*jnp.pi/3
    start=time.perf_counter()
    print("Solving elliptic axis",flush=True)
    axis=periodic_point_state(field,parameters,jnp.array([1.6,0.]),phi_span=span,
                              n_steps_per_span=args.steps,newton_iterations=8,bphi_floor=1e-5)
    eigenvalues,eigenvectors=np.linalg.eig(np.asarray(axis.monodromy));index=int(np.argmax(eigenvalues.imag))
    eigenvalue=eigenvalues[index];v=eigenvectors[:,index]
    if not bool(axis.converged) or abs(eigenvalue.imag)<1e-5:raise RuntimeError('No qualified elliptic axis near the initial guess')
    basis=np.column_stack((v.real,-v.imag));area=np.sign(np.linalg.det(basis))*np.pi*.03**2
    basis*=np.sqrt(abs(area)/(np.pi*abs(np.linalg.det(basis))))
    theta=np.arange(args.nodes)*2*np.pi/args.nodes
    reference=np.asarray(axis.position)+np.column_stack((np.cos(theta),np.sin(theta)))@basis.T
    def mapping(x,c):return poincare_map(field,c,x,phi_span=span,n_steps=args.steps,bphi_floor=1e-5)
    problem=InvariantCircleProblem(mapping,reference,area=area,tolerance=1e-9,dense_tolerance=2e-6)
    print("Correcting invariant circle",flush=True)
    solution=problem.solve(np.ones(3),omega=float(np.angle(eigenvalue)),max_evaluations=100)
    np.savez(args.output/'circle.npz',embedding=solution.embedding,unknowns=solution.unknowns,axis=np.asarray(axis.position),monodromy=np.asarray(axis.monodromy))
    report=dict(case='NCSX modular-coil vacuum',associated_equilibrium='LI383 association unqualified',
                input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),valid=solution.valid,status=solution.status,
                axis=np.asarray(axis.position).tolist(),omega=solution.omega,map_span=float(span),
                signed_iota=solution.omega/float(span),area=area,condition=solution.condition,
                collocation_residual=solution.residual_norm,dense_residual=solution.dense_residual,
                counterterm=solution.counterterm,minimum_speed=solution.minimum_speed,
                minimum_divisor=solution.minimum_divisor,elapsed_seconds=time.perf_counter()-start,
                steps=args.steps,nodes=args.nodes,coil_quadrature=args.segments,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    faulthandler.cancel_dump_traceback_later()
    return 0 if solution.valid else 2


if __name__=='__main__':raise SystemExit(main())
