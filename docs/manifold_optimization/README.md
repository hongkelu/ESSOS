# Manifold optimization

The [reference plan](plan.md) is a proposed research and implementation roadmap.
Its checkboxes are not evidence that a release or experiment is complete.
The working implementation includes shared constrained acceptance, moving-seed
response, independent open bundles, 3-D clearance, exact and approximate surface
solvers, and equilibrium-response contracts. Start with the [API guide](api.md),
[implementation status](implementation_status.md), and [measured results](results.md).
The [baseline report](baseline_report.md) records the original regression run.
Finite-beta work will use [VMEX for the exterior field](vmex_integration.md),
as requested by the user; the VMEC-JAX measurements below are historical diagnostics.

## Compatible source pair

| Repository | Baseline commit |
| --- | --- |
| ESSOS | `3c1dc4ffe0b255eb05389d91803db8591255026b` |
| pyna | `4f032d7128cbc5edd62350b42c1015587c0d684c` |

Both use the `manifold-optimization` branch. The machine-readable contract is
[`baseline.json`](../../benchmarks/manifold_optimization/baseline.json).
The runner records the actual ESSOS revision and dirty state, so testing an ESSOS
change is distinguishable from reproducing the original source baseline. It
requires the pinned pyna revision; changing that pin needs a new compatibility run.

## Reproduce the CPU reference

Use Python 3.12 on Linux, xmake 3.0.1, and a C++17 compiler. Install xmake in a
workspace-local prefix if it is not already provided. For a source build, check
out xmake tag `v3.0.1` with its submodules, configure with
`CC=gcc CXX=g++ ./configure --prefix=<workspace-local-prefix>`, then run
`make -j 8` and `make install` on a compute node. Wait for installation to finish
before invoking it. Put its `bin` directory on `PATH`.

From a parent directory containing `ESSOS/` and `pyna/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONPATH=
export CYNA_WITH_CUDA=0 CYNA_SKIP_TOOL_INSTALL=1
export CC=gcc CXX=g++
export XMAKE_GLOBALDIR="$PWD/.tools/xmake-global"
python -m pip install -r ESSOS/benchmarks/manifold_optimization/requirements-cpu-py312.txt
python -m pip install --no-build-isolation --no-deps -e ./pyna -e ./ESSOS
python -m pip check
python -m pytest ESSOS/tests/test_manifold_baseline_runner.py -q
python ESSOS/scripts/run_manifold_baseline.py \
  --pyna-root pyna --lane all --output run_artifacts/baseline-cpu
```

The output directory must be new. Each selected module runs in a fresh process
to bound accumulated JAX compilation caches and avoid cross-repository `tests`
package collisions. The reference uses CPU JAX, float64, and one OpenMP/BLAS
thread; selecting a GPU allocation does not turn it into a GPU benchmark.

On Perlmutter, run the build, tests, and all production workloads using `srun`
within an **interactive or debug allocation**, never on the login node. The
user's production allocation template is:

```bash
salloc -q interactive --exclusive --constraint=gpu -A m4656_g \
  --nodes=4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none -t 04:00:00
```

A reduced single-process baseline can use one node and one GPU task:

```bash
salloc -q interactive --exclusive --constraint=gpu -A m4656_g \
  --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 --gpu-bind=none -t 01:00:00
srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --gpu-bind=none \
  .venv/bin/python ESSOS/scripts/run_manifold_baseline.py \
  --pyna-root pyna --lane all --output run_artifacts/baseline-cpu
```

Launch the environment/build commands through `srun` too. The runner rejects a
Perlmutter invocation without a Slurm step. Release allocations when work ends.

## Fixture inventory and evidence

| Lane | Existing coverage | Interpretation |
| --- | --- | --- |
| `analytic` | Fixed-angle maps, conditioned roots, moving seeds, curved-wall events, flux measure, exact/approximate circles, snapshots, caches and optimizer contracts | Manufactured derivative and state-integrity verification; physical benchmarks have separate gates. |
| `production` | Native Cyna/JAX comparisons, labelled correspondence, refreshed roots, clearance, candidate validation | Requires the actual compiled backend, including wall-aware tracing. |
| `qa` | Landreman–Paul QA coils, period-five X-line, multi-turn wall strike | Frozen regression wall, not a reactor divertor design benchmark. |
| `tokamak` | Prescribed tokamak-like background plus controlled coils, outer validation and stable-leg clearance | Prescribed-background verification, not a coil-only vacuum tokamak equilibrium. |

The exact test paths, required collection counts, and physical-input path are
listed in `baseline.json`. Update counts deliberately when changing the inventory;
a mismatch fails the required lane.
The QA input is SHA-256 hashed for every run. The runner writes `manifest.json`,
per-module logs, JUnit XML, and a JSON audit listing every collected test and
setup/call/teardown outcome. It records source revisions, tracked-diff checksum,
dirty status, test source checksums, installed package versions, JAX precision
and devices, native binary checksum, Slurm settings, and elapsed time.

Missing imports, an absent native backend, empty collection, skipped tests,
expected failures, deselection, or missing audit output cannot pass the required
lane. Ordinary local pytest remains unchanged. The GitHub workflow exercises
the four lanes separately; configuring its checks as merge requirements is a
repository-settings task, not something a workflow file alone enforces.

Recorded elapsed times include pytest startup and compilation. They are not
steady-state kernel timings, numerical error estimates, or comparative optimizer
benchmarks. Installed-wheel checks execute native tracing and a surface derivative outside both repositories; see the measured-results report.

## Reproduce the physical starting cases

The packaged data directory contains tokamak and LI383 VMEC decks, an NCSX
modular-coil graph, and a vessel mesh with SHA-256/source provenance. The NCSX
vacuum-coil demonstration is distinct from the finite-pressure LI383 equilibrium;
matching them as one physical coil/equilibrium/wall configuration remains open.
Numerical optimization bounds are not device engineering limits.

Run the commands below through `srun` in an interactive/debug allocation on
Perlmutter, with `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1`. VMEC-JAX specifically requires the literal `1` for x64.
All output directories must be new.

```bash
.venv/bin/python ESSOS/scripts/qualify_ncsx_circle.py --output run_artifacts/ncsx-circle
.venv/bin/python ESSOS/scripts/run_ncsx_surface_optimization.py \
  --qualified-circle run_artifacts/ncsx-circle --method implicit --output run_artifacts/ncsx-implicit
.venv/bin/python ESSOS/scripts/run_ncsx_surface_optimization.py \
  --qualified-circle run_artifacts/ncsx-circle --method finite_difference --output run_artifacts/ncsx-fd
.venv/bin/python ESSOS/scripts/run_ncsx_surface_optimization.py \
  --qualified-circle run_artifacts/ncsx-circle --method direct --output run_artifacts/ncsx-direct
.venv/bin/python ESSOS/scripts/validate_ncsx_native_circle.py \
  --optimization run_artifacts/ncsx-implicit --output run_artifacts/ncsx-native
.venv/bin/python ESSOS/scripts/run_ncsx_open_optimization.py \
  --method implicit --output run_artifacts/ncsx-open
.venv/bin/python ESSOS/scripts/run_ncsx_open_optimization.py \
  --method finite_difference --output run_artifacts/ncsx-open-fd
.venv/bin/python ESSOS/scripts/run_tokamak_manifold_optimization.py --output run_artifacts/tokamak-control
.venv/bin/python ESSOS/scripts/qualify_vmec_response.py \
  --solver-root /path/to/vmec_jax --case li383 --output run_artifacts/li383-response
```

The VMEC response command exits unsuccessfully when independent response
qualification fails; this is an essential scientific gate, not an optional test.
The accessible solver revision is recorded with the input provenance. Benchmark
source manifests record dirty working-tree contents as well as base commits.

The current changes span both repositories. Before publishing paired commits,
advance the pyna compatibility pin to the committed implementation and rerun the
required lanes. The original base commit alone does not contain the new APIs.
The workflow has been prepared locally; no remote CI or publication is claimed.

## Installed-wheel validation

`build_manifold_wheels.py --pyna-root pyna --output .wheels` builds temporary
source copies so a build cannot overwrite a native library loaded by an ongoing
validation process. Set `CYNA_WITH_CUDA=0 CYNA_SKIP_TOOL_INSTALL=1` and the local
xmake/compiler environment described above. The source manifest hashes every
copied file and the resulting wheels. Install with `pip install --no-deps
--target /tmp/manifold-installed .wheels/*.whl`, then run
`verify_installed_manifold.py --prefix /tmp/manifold-installed --output report.json`
from outside both repositories with `PYTHONPATH=/tmp/manifold-installed`. Execute
these build/install/numerical commands through `srun` on Perlmutter too.


## VMEX exterior-field and tracing checks

The recorded 0.17–0.23% common-current response discrepancy is accepted by the
user for moving to the next stage. See the [VMEX integration report](vmex_integration.md)
for that scoped decision, field-domain refinement, and the direct/native trace
comparison. The original strict derivative reports remain intact.

Use the separate `.venv-vmex` environment and an interactive/debug `srun` step:

```bash
.venv-vmex/bin/python ESSOS/scripts/qualify_vmex_domain.py \
  --equilibrium-run run_artifacts/vmex-20260922/li383-continuation-16-24-32 \
  --output run_artifacts/new-domain-check
.venv-vmex/bin/python ESSOS/scripts/qualify_vmex_trace.py \
  --equilibrium-run run_artifacts/vmex-20260922/li383-continuation-16-24-32 \
  --output run_artifacts/new-trace-check
```

The checkpoint loader verifies equilibrium/edge residuals, input hashes, VMEX
source hashes and spectral checkpoint hashes. Domain classification uses the
actual Fourier boundary; the trace example chooses a complete rectangular
field grid above a conservative global LCFS height bound. It is a short fixed-
equilibrium exterior regression, with no wall-target or design claim.
