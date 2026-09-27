#!/usr/bin/env python3
"""Exercise VMEX field/response/native-trace tests using installed project wheels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    if args.output.exists():parser.error('Output must be new')
    prefix=args.prefix.resolve();root=Path(__file__).resolve().parents[1]
    os.environ.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0,str(prefix))
    import essos.vmex_field as adapter
    import essos.vmex_domain as domain
    import pyna._cyna as native
    if any(not Path(path).is_relative_to(prefix) for path in (adapter.__file__,domain.__file__,native._cyna_ext.__file__)):
        raise RuntimeError('Project modules must come from the installed-wheel prefix')
    if not native.is_available():raise RuntimeError('Native Cyna required')
    with tempfile.TemporaryDirectory(prefix='vmex-wheel-test-') as directory:
        directory=Path(directory)
        test=root/'tests/test_vmex_field.py'
        shutil.copy(test,directory/'test_installed_vmex.py')
        domain_test=root/'tests/test_vmex_domain.py'
        shutil.copy(domain_test,directory/'test_installed_domain.py')
        shutil.copy(root/'scripts/manifold_audit.py',directory/'manifold_audit.py')
        environment=os.environ.copy();environment['PYTHONPATH']=str(prefix)+os.pathsep+str(directory)
        environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';environment['PYTEST_ADDOPTS']=''
        audit=directory/'audit.json'
        result=subprocess.run([sys.executable,'-m','pytest','test_installed_vmex.py','test_installed_domain.py','-q',
            '-p','manifold_audit',f'--manifold-audit={audit}'],cwd=directory,env=environment)
        data=json.loads(audit.read_text()) if audit.exists() else {}
        valid=result.returncode==0 and data.get('exit_code')==0 and len(data.get('collected',[]))==6
        report=dict(valid=valid,tests=len(data.get('collected',[])),adapter=adapter.__file__,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),native=native._cyna_ext.__file__,
            adapter_sha256=hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
            native_sha256=hashlib.sha256(Path(native._cyna_ext.__file__).read_bytes()).hexdigest(),
            test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
            domain_sha256=hashlib.sha256(Path(domain.__file__).read_bytes()).hexdigest(),
            domain_test_sha256=hashlib.sha256(domain_test.read_bytes()).hexdigest())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        return 0 if valid else 1


if __name__=='__main__':raise SystemExit(main())
