#!/usr/bin/env python3
"""Verify installed-wheel imports, resources, JAX and native tracing outside source roots.

Run from a directory outside both repositories with PYTHONPATH set to the
pip --target installation directory. Numerical modules must come from the installed prefix.
"""
import argparse
import hashlib
import importlib
import importlib.resources
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prefix=args.prefix.resolve()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    modules={}
    for name in ('essos','essos.topology_objectives','essos.equilibrium','essos.topology_objectives',
                 'essos.topology_objectives','essos.coils','pyna','pyna.topo.torus_solver','pyna.topo.circle_validation','pyna.topo.open_validation','pyna._cyna'):
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        if not path.is_relative_to(prefix):raise RuntimeError(f'{name} imported outside installed prefix: {path}')
        modules[name]=str(path)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pyna._cyna as cyna
    from pyna.fields import VectorFieldCylind
    from pyna.toroidal.flt import trace_map_batch_span_field
    from pyna.topo.torus_solver import InvariantCircleProblem
    if not cyna.is_available():raise RuntimeError('Native tracing unavailable')
    native=Path(cyna._cyna_ext.__file__).resolve()
    if not native.is_relative_to(prefix):raise RuntimeError('Native extension leaked from source installation')
    r=np.linspace(1.,2.,8);z=np.linspace(-.5,.5,8);phi=np.linspace(0.,2*np.pi,8,endpoint=False)
    rr,zz,pp=np.meshgrid(r,z,phi,indexing='ij')
    field=VectorFieldCylind(R=r,Z=z,Phi=phi,BR=rr*0,BZ=rr*0,BPhi=rr,nfp=1)
    counts,rout,zout=trace_map_batch_span_field(field,np.array([1.5]),np.array([.1]),0.,.5,1,.01,n_threads=1)
    np.testing.assert_array_equal(counts,[1]);np.testing.assert_allclose([rout[0],zout[0]],[1.5,.1],atol=1e-12)
    theta=np.arange(9)*2*np.pi/9;reference=np.column_stack((2+.2*np.cos(theta),.2*np.sin(theta)))
    def mapping(x,c):
        a=.7+c[0];v=x-jnp.array([2.,0.]);return jnp.array([2.,0.])+jnp.array([[jnp.cos(a),-jnp.sin(a)],[jnp.sin(a),jnp.cos(a)]])@v
    problem=InvariantCircleProblem(mapping,reference,area=np.pi*.2**2)
    circle=problem.solve(np.array([0.]),omega=.7)
    if not circle.valid:raise RuntimeError(circle.status)
    np.testing.assert_allclose(problem.jvp(circle,np.ones(1))[-2],1.,atol=1e-10)
    resources={}
    for name in ('tokamak.vmec','li383.vmec','ncsx_coils.json','ncsx_wall.dat','provenance.json'):
        data=importlib.resources.files('essos').joinpath('data/manifold_optimization',name).read_bytes()
        resources[name]=hashlib.sha256(data).hexdigest()
    report=dict(valid=True,modules=modules,native_extension=str(native),native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                resources=resources,jax_version=jax.__version__,x64=jax.config.x64_enabled,circle_response=True,native_trace=True)
    if not report['x64']:raise RuntimeError('float64 required')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
