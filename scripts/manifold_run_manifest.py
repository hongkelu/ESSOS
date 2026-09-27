"""Standard-library provenance capture before a numerical benchmark starts."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime,timezone
from run_manifold_baseline import repository_state


def capture_source_manifest(output,external_repositories=()):
    root=Path(__file__).resolve().parents[1]
    repositories={'essos':root,'pyna':root.parent/'pyna'}
    repositories.update({Path(path).name:Path(path).resolve() for path in external_repositories})
    inputs=root/'essos/data/manifold_optimization'
    manifest=dict(schema_version=1,started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        python=sys.version,platform=platform.platform(),
        repositories={name:repository_state(path) for name,path in repositories.items()},
        packages=dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())),
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()},
        environment={name:os.environ.get(name) for name in ('JAX_PLATFORMS','JAX_ENABLE_X64','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                      'SLURM_JOB_ID','SLURM_STEP_ID','SLURM_CPUS_PER_TASK','SLURM_JOB_NODELIST')},
        compiler=subprocess.check_output([os.environ.get('CXX','c++'),'--version'],text=True).splitlines()[0])
    (Path(output)/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
