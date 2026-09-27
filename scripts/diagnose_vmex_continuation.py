#!/usr/bin/env python3
"""Attribute a failed response check using recorded finite-difference states.

This is an explanatory diagnostic, NOT an independent derivative certificate:
the omitted-direction estimate deliberately uses the same perturbed solves.
Uses private hooks from the source-hashed VMEX qualification snapshot.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    args.output.mkdir(parents=True, exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    shutil.copyfile(__file__, args.output/'diagnostic_source.py')
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    import numpy as np
    import vmex as vm
    if not jax.config.x64_enabled:
        raise ValueError('Float64 required')
    from vmex.core import implicit as im, freeboundary_implicit as fi, virtual_casing as vc
    from vmex.core.extender import VmecExtender
    from vmex.core.solver import SpectralState
    from essos.coil_inputs import load_simsopt_xyz_coils
    from essos.fields import BiotSavart
    report = json.loads((args.run/'report.json').read_text())
    runtime_root = Path(vm.__file__).resolve().parent
    for name, digest in report['vmex_runtime_sources'].items():
        if hashlib.sha256((runtime_root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Runtime differs from qualification: '+name)
    records = json.loads((args.run/'continuation.json').read_text())
    def load(current):
        matches = [r for r in records if r['current_factor'] == current]
        if len(matches) != 1:raise ValueError('Ambiguous or absent current factor')
        row = matches[0]['stages'][-1]
        if not row['accepted'] or row['ns'] != report['radial_resolution']:
            raise ValueError('Checkpoint is not an accepted final rung')
        path = args.run/row['checkpoint']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['checkpoint_sha256']:
            raise ValueError('Checkpoint digest mismatch')
        with np.load(path, allow_pickle=False) as a:
            state = SpectralState(**{k:jnp.asarray(a[k]) for k in
                ('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin')})
            return state, jnp.asarray(a['rcon0']), jnp.asarray(a['zcon0'])
    h = report['finite_differences'][-1]['step']
    state, rcon, zcon = load(1.)
    plus, minus = load(1+h), load(1-h)
    dstate, drcon, dzcon = jax.tree.map(lambda a,b:(a-b)/(2*h), plus, minus)
    data = Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization'
    for name, digest in report['input_sha256'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Physical input changed')
    coils = load_simsopt_xyz_coils(data/'ncsx_coils.json', n_segments=report['coil_quadrature'])
    g = report['field_grid']
    external = vm.MgridField.from_coils(coils,rmin=g['R'][0],rmax=g['R'][1],
        zmin=g['Z'][0],zmax=g['Z'][1],ir=g['R'][2],jz=g['Z'][2],kp=g['nphi'],nfp=g['nfp'])
    inp = replace(vm.VmecInput.from_file(data/'li383.vmec'),lfreeb=True,mgrid_file='NCSX packaged coils, direct tabulation',nzeta=16,
        ns_array=[report['radial_resolution']], ftol_array=[report['force_tolerance']])
    cfg = vm.make_free_boundary_config(inp,external,ns=report['radial_resolution'],
        ftol=report['force_tolerance'],adjoint_tol=1e-8,adjoint_solver='boundary_schur',
        include_edge_in_convergence=True,edge_force_tolerance=report['force_tolerance'],
        field_from_parameters=lambda c:replace(external,extcur=c))
    params = im.params_from_input(inp)
    runtime = im.runtime_from_params(params,cfg.implicit)
    # Reconstruct only the pressure needed for structural-mask discovery.
    rt = replace(runtime,rcon0=rcon,zcon0=zcon,lfreeb=True,jmax=report['radial_resolution'])
    stage = SimpleNamespace(result=SimpleNamespace(state=state),rcon0=rcon,zcon0=zcon,
        vacuum=SimpleNamespace(bsqvac=cfg.vacuum_program.bsq(state,rt,external)))
    _, mask, _, _ = fi._linearization_from_stage(cfg,params,stage,inp=inp)
    project = im._dof_projector(cfg.implicit,mask)
    inactive = jax.tree.map(lambda a,b:a-b,dstate,project(dstate))
    norm = lambda t:float(jnp.linalg.norm(ravel_pytree(t)[0]))
    c = jnp.ones(1)
    residual = fi._projected_residual(cfg,mask,formulation='raw')
    z = project(state)
    nuisance = jax.jvp(lambda s,r,q:residual(z,params,c,s,r,q),
        (state,rcon,zcon),(inactive,drcon,dzcon))[1]
    flat, unravel = ravel_pytree(state)
    direct = jax.jit(jax.vmap(BiotSavart(coils).B))
    points = jnp.array([[3.2,0.,.1]])
    backend = VmecExtender.from_parameterized_surface_data(
        lambda x:vc.surface_field_data_from_state(inp,unravel(x),runtime=runtime,nphi=48,ntheta=48),
        flat,external_field=direct,digits=5,levels=((192,96),(384,192)),accuracy_check='off')
    backend.set_points(points)
    rows = []
    result = dict(independent_qualification=False,source_run=str(args.run.resolve()),step=h,
        explanation='Finite-difference state and constraint drift attribute error; they do not qualify a corrected derivative.',
        state_direction_norm=norm(dstate),inactive_direction_norm=norm(inactive),
        rcon_direction_norm=norm(drcon),zcon_direction_norm=norm(dzcon),components=rows)
    fd_data = np.load(args.run/f'fd-{h:.0e}.npz',allow_pickle=False)
    dot = lambda a,b:float(jnp.vdot(ravel_pytree(a)[0],ravel_pytree(b)[0]))
    for component in range(3):
        print(f'Attributing Cartesian component {component}',flush=True)
        weight = jnp.zeros((1,3)).at[0,component].set(1.)
        state_bar = unravel(backend.B_vjp(weight))
        adjoint = fi._host_boundary_schur_adjoint(cfg,z,params,c,state,rcon,zcon,
            mask,project(state_bar),fail='error')
        explicit = dot(state_bar,inactive)
        implicit = -dot(adjoint,nuisance)
        fd = float(fd_data['finite_difference'][0,component])
        native = float(fd_data['response'][0,component])
        rows.append(dict(component=component,finite_difference=fd,native=native,
            inactive_explicit=explicit,nuisance_implicit=implicit,
            explained_response=native+explicit+implicit,
            remaining_difference=fd-native-explicit-implicit))
        (args.output/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    unexplained = np.array([r['remaining_difference'] for r in rows])
    result['remaining_relative_difference'] = float(np.linalg.norm(unexplained)/np.linalg.norm(fd_data['finite_difference']))
    (args.output/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__ == '__main__':main()
