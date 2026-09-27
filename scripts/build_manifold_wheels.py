#!/usr/bin/env python3
"""Build the working source pair in temporary copies, preserving live extensions."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pyna-root',type=Path,required=True)
    args=parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    roots={'ESSOS':Path(__file__).resolve().parents[1],'pyna':args.pyna_root.resolve()}
    hashes={};versions={'essos':importlib.metadata.version('essos'),'pyna-chaos':importlib.metadata.version('pyna-chaos')}
    with tempfile.TemporaryDirectory(prefix='manifold-wheel-source-') as temporary:
        copied=[]
        for name,root in roots.items():
            destination=Path(temporary)/name;destination.mkdir()
            names=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
            for relative in sorted(set(names)):
                if not relative:continue
                source=root/relative
                if not source.is_file():continue
                target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target);hashes[name+'/'+relative]=hashlib.sha256(target.read_bytes()).hexdigest()
            copied.append(str(destination))
        environment=os.environ.copy();environment['SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ESSOS']=versions['essos']
        subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',*copied,'-w',str(output)],env=environment,check=True)
    manifest=dict(source_files=hashes,versions=versions,wheels={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.whl')})
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
