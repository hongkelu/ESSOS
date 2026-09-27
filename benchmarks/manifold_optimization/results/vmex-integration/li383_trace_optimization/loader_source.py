"""Read a source-verified, edge-accepted LI383/NCSX continuation checkpoint."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def load_accepted_case(run):
    import numpy as np
    import jax.numpy as jnp
    import vmex as vm
    from vmex.core.solver import SpectralState
    from essos.coil_inputs import load_simsopt_xyz_coils
    run=Path(run)
    report=json.loads((run/'report.json').read_text())
    runtime_root=Path(vm.__file__).resolve().parent
    for name,digest in report['vmex_runtime_sources'].items():
        if hashlib.sha256((runtime_root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('VMEX runtime changed since equilibrium acceptance')
    records=json.loads((run/'continuation.json').read_text())
    bases=[r for r in records if r['current_factor']==1.]
    if len(bases)!=1:raise ValueError('Expected one base equilibrium')
    row=bases[0]['stages'][-1]
    residuals=np.asarray(row['residuals'],float)
    if (not row['accepted'] or residuals.shape!=(4,) or not np.all(np.isfinite(residuals))
            or np.any(residuals<0) or np.max(residuals)>report['force_tolerance']):
        raise ValueError('Equilibrium/edge acceptance failed')
    checkpoint=run/row['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=row['checkpoint_sha256']:
        raise ValueError('Equilibrium checkpoint digest mismatch')
    with np.load(checkpoint,allow_pickle=False) as a:
        state=SpectralState(**{k:jnp.asarray(a[k]) for k in ('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin')})
    inputs=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization'
    for name,digest in report['input_sha256'].items():
        if hashlib.sha256((inputs/name).read_bytes()).hexdigest()!=digest:raise ValueError('Physical input changed')
    inp=replace(vm.VmecInput.from_file(inputs/'li383.vmec'),lfreeb=True,nzeta=16,
        mgrid_file='NCSX packaged coils, direct tabulation',ns_array=[row['ns']])
    wout=vm.wout_from_state(inp=inp,state=state,converged=True,fsqr=residuals[0],
        fsqz=residuals[1],fsql=residuals[2],niter=row['iterations'])
    coils=load_simsopt_xyz_coils(inputs/'ncsx_coils.json',n_segments=report['coil_quadrature'])
    return SimpleNamespace(inp=inp,state=state,wout=wout,coils=coils,report=report,checkpoint_sha256=row['checkpoint_sha256'])
