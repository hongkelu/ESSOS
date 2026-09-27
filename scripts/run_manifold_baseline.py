#!/usr/bin/env python3
"""Run the required pinned-pair regression inventory, one process per module.

This measures regression execution, not optimizer or kernel performance.
Only standard-library imports occur before the compute-node guard.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "benchmarks/manifold_optimization/baseline.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def repository_state(root):
    return {"path": str(root), "commit": git(root, "rev-parse", "HEAD"),
            "status": git(root, "status", "--porcelain"),
            "untracked_sha256": {name: sha256(root / name) for name in
                git(root, "ls-files", "--others", "--exclude-standard").splitlines()
                if (root / name).is_file()},
            "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(
                ["git", "-C", str(root), "diff", "HEAD", "--binary"]
            )).hexdigest()}


def audit_passed(returncode, audit, expected_count):
    """A successful process and audit must account for the complete inventory."""
    return (returncode == 0 and audit.get("exit_code") == 0
            and len(audit.get("collected", [])) == expected_count
            and expected_count > 0)


def select_lanes(config, lane):
    selected = config["lanes"] if lane == "all" else {lane: config["lanes"][lane]}
    if not selected:
        raise ValueError("Required baseline inventory is empty")
    for groups in selected.values():
        if not groups or any(not selections for selections in groups.values()):
            raise ValueError("Required baseline lane is empty")
        for repo, selections in groups.items():
            for selection in selections:
                count = config.get("expected_counts", {}).get(f"{repo}/{selection}")
                if type(count) is not int or count <= 0:
                    raise ValueError(f"Missing positive expected count for {repo}/{selection}")
    return selected


def require_compute_allocation(parser):
    """Enforce this project's Perlmutter interactive/debug execution policy."""
    if os.environ.get("NERSC_HOST") != "perlmutter":
        return {}
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not os.environ.get("SLURM_STEP_ID"):
        parser.error("On Perlmutter, launch with srun in an interactive/debug allocation.")
    details = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"], text=True)
    allocation = dict(re.findall(r"(?:^|\s)(QOS|NumNodes|NumCPUs)=(\S+)", details))
    if allocation.get("QOS") not in {"interactive", "gpu_interactive", "debug", "gpu_debug"}:
        parser.error("Perlmutter baseline runs require an interactive or debug allocation.")
    return allocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--pyna-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=["analytic", "production", "qa", "tokamak", "all"],
                        default="analytic")
    args = parser.parse_args()
    allocation = require_compute_allocation(parser)

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    try:
        selected = select_lanes(config, args.lane)
    except ValueError as exc:
        parser.error(str(exc))
    roots = {"essos": ROOT, "pyna": args.pyna_root.resolve()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.json").write_bytes(config_path.read_bytes())
    # A fresh output directory prevents stale success reports from surviving a crash.
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="")
    env["PYTHONPATH"] = os.pathsep.join(map(str, [ROOT / "scripts", *roots.values()]))
    os.environ.update({key: env[key] for key in (
        "JAX_PLATFORMS", "JAX_ENABLE_X64", "MPLBACKEND", "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    sys.path[:0] = list(map(str, roots.values()))
    manifest = {
        "schema_version": 1, "allocation": allocation, "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preflight", "lane": args.lane,
        "baseline": config["repositories"],
        "repositories": {name: repository_state(path) for name, path in roots.items()},
        "python": sys.version, "platform": platform.platform(),
        "cpu": platform.processor(), "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "compiler": subprocess.check_output([os.environ.get("CXX", "c++"), "--version"], text=True).splitlines()[0],
        "environment": {k: env[k] for k in (
            "JAX_PLATFORMS", "JAX_ENABLE_X64", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "slurm": {k: os.environ.get(k) for k in (
            "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_PARTITION",
            "SLURM_JOB_QOS", "SLURM_JOB_NUM_NODES", "SLURM_CPUS_PER_TASK")},
        "packages": dict(sorted((d.metadata["Name"], d.version)
                                for d in importlib.metadata.distributions())),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(__file__),
        "audit_plugin_sha256": sha256(ROOT / "scripts/manifold_audit.py"),
        "requirements_sha256": sha256(config_path.parent / config.get("requirements", "requirements-cpu-py312.txt")),
        "command": sys.argv, "results": [],
    }
    path = output / "manifest.json"

    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)

    save()
    try:
        if manifest["repositories"]["pyna"]["commit"] != config["repositories"]["pyna"]["commit"]:
            raise RuntimeError("pyna HEAD differs from the pinned compatibility commit")
        for name, root in roots.items():
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f"{name} imported outside the selected checkout")
        import jax
        import pyna._cyna as cyna
        if not cyna.is_available() or cyna.VectorFieldCylind is None:
            raise RuntimeError("Required native cyna tracing backend is unavailable")
        for symbol in ("trace_map_batch_span", "trace_wall_hits_twall", "trace_orbit_along_phi"):
            if not callable(getattr(cyna, symbol, None)):
                raise RuntimeError(f"Required cyna API unavailable: {symbol}")
        manifest["backend"] = {"jax": jax.__version__, "x64": jax.config.x64_enabled,
                               "devices": [str(d) for d in jax.devices()],
                               "cyna": cyna.get_version(),
                               "cyna_binary_sha256": sha256(cyna._cyna_ext.__file__)}
        manifest["inputs"] = {name: sha256(ROOT / name) for name in config["inputs"]}
    except Exception as exc:
        manifest.update(status="preflight_failed", error=f"{type(exc).__name__}: {exc}")
        save()
        print(manifest["error"], file=sys.stderr)
        return 1

    manifest["status"] = "running"
    save()
    for lane, groups in selected.items():
        for repo, selections in groups.items():
            for index, selection in enumerate(selections):
                label = f"{lane}-{repo}-{index:02d}"
                audit_path = output / f"{label}.json"
                command = [sys.executable, "-m", "pytest", selection, "-q", "-ra",
                           "--durations=10", "-p", "manifold_audit",
                           f"--manifold-audit={audit_path}",
                           f"--junitxml={output / (label + '.xml')}"]
                print(f"[{label}] {selection}", flush=True)
                started = time.perf_counter()
                with (output / f"{label}.log").open("w") as log:
                    result = subprocess.run(command, cwd=roots[repo], env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
                audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
                expected_count = config["expected_counts"][f"{repo}/{selection}"]
                passed = audit_passed(result.returncode, audit, expected_count)
                manifest["results"].append({
                    "lane": lane, "repository": repo, "selection": selection,
                    "test_file_sha256": sha256(roots[repo] / selection.split("::")[0]),
                    "command": command, "elapsed_seconds": time.perf_counter() - started,
                    "returncode": result.returncode, "passed": passed,
                    "collected": len(audit.get("collected", [])),
                    "expected_count": expected_count, "artifact_prefix": label,
                })
                save()
                print(f"  {'PASS' if passed else 'FAIL'} ({len(audit.get('collected', []))} tests)", flush=True)
    manifest["status"] = "passed" if all(r["passed"] for r in manifest["results"]) else "failed"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
