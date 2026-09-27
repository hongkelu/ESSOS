# VMEX finite-beta integration decision

The user selected **VMEX** for finite-beta work requiring magnetic field outside
the LCFS. This supersedes the proposed next step of extending the experimental
VMEC-JAX material-grid adapter into the production exterior-field provider.
The original attached plan and historical measured results remain unchanged.

The user subsequently accepted the measured **0.17–0.23%** common-current
response discrepancy and requested proceeding to the next plan stage. That
acceptance applies to the recorded LI383/NCSX cases and direction; it does not
change the historical 0.1% reports or establish a universal error tolerance.
The decision is recorded in
[`accepted_response_policy.json`](../../benchmarks/manifold_optimization/results/vmex-integration/accepted_response_policy.json).
Exterior-domain and tracing qualification now take priority over further
reduction of this accepted discrepancy.

## Accessible implementation

Read-only source inspection found
`/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex-main`, with base revision
`345f940c56264dc2a66c95b0201330283cc5241d`. This working tree has local changes;
the revision alone does not identify its current runtime source. Separate
`vmex-finite-beta` and other VMEX worktrees also exist and contain ongoing work.
The integration uses an isolated export of that committed revision, as recorded
below; no other task's working tree was changed.

Verified entry points in `vmex/core/extender.py`:

- `VmecExtender.from_equilibrium` / `from_state` / `from_file` construct the
  Cartesian exterior field from external coils plus the virtual-casing field
  of currents inside the LCFS.
- `from_parameterized_surface_data` exposes parameter VJPs, including separate
  external-field parameters. The surface-data function must carry the actual
  equilibrium response. Spatial differentiability alone does not establish
  the total derivative with respect to coil controls.
- `B_error_estimate` and `accuracy_check="raise"` support direct-quadrature
  qualification. JIT/traced calls bypass eager accuracy checks; acceptance must
  explicitly check error estimates at the relevant points.
- `with_near_surface_continuation` is a first-order near-surface approximation
  with a limited domain. It must be qualified against appropriate refinement
  or independent reference data before tracing close to the LCFS.

The tree includes `examples/data/input.li383_low_res` and the example
`examples/vmex_fieldline_tracing_finite_beta.py`. That example demonstrates
exterior-field use; it is not yet a qualified LI383 coil/equilibrium/wall case
for this project. Production tracing here continues to belong to pyna/Cyna.

## Qualification sequence and remaining scope

1. Capture VMEX and virtual-casing dependency versions/source identities in a
   workspace-local environment. Retain tokamak and LI383 as starting inputs.
2. Establish a consistent equilibrium and external-coil configuration. For coil
   controls, use the corresponding free-boundary equilibrium response; a fixed
   boundary plus arbitrary external coils does not establish that consistency.
3. Evaluate total field at fixed Cartesian positions outside the LCFS. Check
   boundary/interface consistency and convergence versus quadrature resolution
   and distance from the LCFS. Establish the valid near-surface treatment.
4. Check total coil-control JVP/VJP against independently re-solved equilibria
   with rebuilt plasma surface data and exterior fields at the same Cartesian
   positions. Include both direct coil response and changing plasma response.
5. Supply the qualified field to pyna/Cyna, qualify grid/tracing error, and run
   a small finite-beta optimization with fresh equilibrium, exterior-field and
   topology validation at every accepted candidate.

Builds and numerical checks must use a Perlmutter interactive/debug allocation.
The recorded VMEX numerical checks used allocations 58712906 and 58716831; see
`run_artifacts/vmex-20260922/`. The earlier
1.05–4.87% LI383 derivative discrepancy belongs to the separate VMEC-JAX
material-grid adapter; it is not a VMEX test result or evidence of a VMEX defect.


## Implemented integration

`essos.equilibrium.VmexExteriorField` wraps a live VMEX `VmecExtender` and:

- samples Cartesian total fields with an explicit exterior-domain predicate;
- checks virtual-casing error estimates eagerly, including for compiled fields;
- rejects near-surface continuation until a separate error budget is supplied;
- exposes backend parameter VJPs and a reduced reference JVP assembled through
  public adjoint queries (one adjoint per queried field component);
- validates every sample before converting an exterior-only cylindrical grid
  into pyna's production field type for native Cyna tracing.

The quadrature estimate is normalized by RMS surface field, not by local total
field. It does not bound equilibrium discretization, external-coil quadrature,
field-grid interpolation or tracing error. Backend DOF names must match the
explicit equilibrium definition. A fixed wout field is usable for field queries
but does not acquire an equilibrium response by being wrapped in this adapter.

The implementation deliberately rejects a rectangular grid that reaches inside
the LCFS. A future whole-domain provider needs separately qualified interior and
exterior evaluations and interface treatment.

## Reproduction environment and commands

VMEX needs JAX >= 0.9.2 and SciPy >= 1.16. The existing vacuum reference uses
JAX 0.6.2/SciPy 1.15.3, so the VMEX lane has its own environment. The pyna SciPy
upper bound is extended only to `<1.17`; the new lane targets SciPy 1.16.3.
ESSOS exposes an optional `vmex` dependency extra. Reproducibility requires the
recorded VMEX source commit in addition to its package version.

The local backend was exported with `git archive` at commit
`345f940c56264dc2a66c95b0201330283cc5241d` into
`.dependencies/vmex-345f940c`, leaving all other VMEX working trees untouched.
Install that snapshot with its `freeb` extra and the VMEX requirements file,
then install the two working-source wheels. The resolved dependency list and
wheel source manifests live in the run artifacts. All install/build/test/solve
commands run through `srun` inside an interactive/debug allocation.

```bash
.venv-vmex/bin/python ESSOS/scripts/run_manifold_baseline.py \
  --config ESSOS/benchmarks/manifold_optimization/vmex.json \
  --pyna-root pyna --lane production --output run_artifacts/vmex-required
.venv-vmex/bin/python ESSOS/scripts/qualify_vmex_exterior.py \
  --case li383 --response --output run_artifacts/vmex-li383
.venv-vmex/bin/python ESSOS/scripts/qualify_vmex_exterior.py \
  --case tokamak --response --output run_artifacts/vmex-tokamak
```

Set `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1`; during development set `PYTHONPATH` to the ESSOS/pyna
working trees. `run_manifold_baseline.py` sets those paths itself. The dedicated
lane requires all seven tests; ordinary pytest may skip the optional VMEX module
when the backend is absent, but the required audit rejects that skip.

The new filament fixture compares the actual virtual-casing implementation with
independent Biot--Savart fields, moving control responses, and native wall-hit
tracing. It is a prescribed layer-potential reference, not a manufactured MHD
force-balanced equilibrium. The fixed-boundary qualification script solves the actual tokamak/LI383 inputs,
checks source-grid refinement, and compares boundary response at fixed Cartesian
points against independent complete equilibrium re-solves. That experiment
retains fixed external fields and does not qualify free-boundary coil control.
The separate free-boundary pilot below includes changing coil currents.


## Measured results and response convention

The pinned VMEX source solves LI383 with volume-average beta about **4.262%**.
The tokamak deck in this diagnostic has zero pressure and a plasma field; it
is a current-carrying comparator, not a finite-pressure demonstration.
At the two fixed exterior Cartesian sample points, source-grid refinement passes
for both cases. These points are well outside the LCFS; this does not yet
qualify a near-LCFS tracing region or establish equilibrium/coil matching.

At force tolerance `1e-16`, the tokamak exterior boundary-response relative
errors against complete re-solves are `2.07e-5`, `5.04e-6`, and `4.23e-6`
for perturbations `1e-3`, `3e-4`, and `1e-4`. It passes the `1e-3` threshold.
The gate requires two adjacent perturbations to pass, finite converged primals,
field quadrature/refinement checks, and the JVP/VJP transpose check.

LI383 does not pass that independent-response gate. At 16 radial surfaces,
errors are 4.163%, 1.607%, and 1.392%; at 32 surfaces they are 2.321%, 2.195%,
and 2.059%. Tightening force tolerance and refining radius therefore do not
establish agreement with ordinary re-solves.

VMEX's `core/implicit.py` explicitly freezes inactive solver combinations,
including constrained m=1 combinations, at the base state. A separate diagnostic
holds those choices fixed and Newton-solves the same residual at the perturbed
parameters using VMEX's `frozen_path_directional_fd`. For a weighted exterior
field scalar, its finite difference and adjoint agree to relative `9.43e-8`,
with the two Newton residual norms below `1.7e-14`.

This supports a distinction between the frozen-residual derivative and the
ordinary solver's parameter-to-field map. It does not independently prove which
convention gives the desired physical response at finite resolution. The
frozen-path diagnostic is retained **separately** and does not turn the failed
ordinary re-solve qualification into a pass. In particular, the adapter's VJP
inherits the backend's response convention; it must not be described as an
already qualified total coil-response derivative.

The free-boundary pilot below establishes a converged coil-driven starting
equilibrium. Its total response still needs to pass independent re-solves under
an explicit continuation/gauge convention. Near-LCFS error budgets and
whole-domain tracing remain separate requirements.

Curated reports are in
[`results/vmex-integration`](../../benchmarks/manifold_optimization/results/vmex-integration).
The historical VMEC-JAX discrepancy is a separate experiment; the numbers above
come from actual VMEX runs.


## Free-boundary coil-driven pilot

`qualify_vmex_freeboundary.py` now solves LI383 with the packaged NCSX modular
coils supplying the NESTOR external field, retaining the input pressure/current
profiles and toroidal flux while allowing the boundary to move. Both the initial
64 x 64 x 32 coil grid (128-point coil quadrature, force tolerance `1e-9`) and
96 x 96 x 64 refinement (256-point quadrature, tolerance `1e-10`) converge with
edge-force convergence enabled. Their volume-average beta values are about
5.48443% and 5.48450%, respectively. This establishes a converged reduced
coil-driven starting equilibrium, not an engineering-qualified design or proof
that it reproduces the original fixed LI383 boundary.

The coupled response pilot ties VMEX's plasma-response and direct-coil parameter
blocks to one common coil-current factor using `control_jacobian`. It compares
against ordinary free-boundary re-solves at perturbed coil currents. The live
state field representation is held fixed across both derivative and finite-
difference calculations; wout reconstruction is checked separately.

A detected backend behavior matters here: the free-boundary implicit wrapper
hot-restarts on repeated calls even at identical controls. Its next solve can
change the state after the first exterior field is constructed. The reference
script therefore creates a fresh identity-keyed free-boundary configuration for
each surface evaluation. This costs additional cold solves but makes the value
and derivative construction independent of warm-start call history. Production
reuse will need an accepted-state cache keyed by the complete physical and
numerical problem, with repeatability and response checks.


At the fixed Cartesian point `[3.2, 0, 0.1]`, fresh configurations remove the
base-field mismatch: the ordinary and implicit base fields agree, while wout
and live-state evaluations differ by only `4.39e-16` relatively. The remaining
response discrepancy is measured against independently converged free-boundary
solves, including both the direct coil and plasma changes:

| Radial surfaces | Force tolerance | Error at h=1e-3 | Error at h=3e-4 | Error at h=1e-4 | Outcome |
| --- | --- | --- | --- | --- | --- |
| 16 | 1e-10 | 0.08595% | 0.13605% | 0.13746% | Fails two-adjacent-step gate |
| 16 | 1e-12 | 0.09867% | 0.12391% | 0.12198% | Fails two-adjacent-step gate |
| 32 (cold) | 1e-12 | — | — | — | Base equilibrium did not converge |
| 32 (continuation) | 1e-12 | 0.19889% | 0.22750% | 0.23357% | Fails response gate |
| 64 (continuation) | 1e-12 | 0.16459% | 0.17740% | 0.17108% | Fails response gate |

The threshold remains **0.1%**, with two adjacent perturbation amplitudes
required to pass. Tightening the force tolerance alone does not close this
gate. The direct 32-surface cold-start attempt reached its 3000-iteration limit
with force residuals about `17.2`, `20.4`, and `3.39e-5`; no derivative comparison
was accepted from that state. The radial continuations documented below now
converge at 32 and 64 surfaces, but still fail the response gate. These current-
response checks are a different problem from the fixed-boundary response above;
their percentages must not be treated as successive estimates of one error.

The full 262-test baseline passes in the new JAX/SciPy environment, together
with four new VMEX tests (**266 unique tests**). The required VMEX lane passes
all seven tests (three overlap the baseline), and all four VMEX tests also pass
against freshly built installed wheels and the installed native Cyna binary.
Curated manifests retain the original allocation interruption and the successful
replacement QA/tokamak runs. Software test success does not override the failed
physical response gate.


Run the refined coupled pilot through an interactive/debug `srun` step:

```bash
.venv-vmex/bin/python ESSOS/scripts/qualify_vmex_freeboundary.py \
  --grid 96 --planes 64 --coil-quadrature 256 --force-tolerance 1e-12 \
  --response --output run_artifacts/vmex-freeboundary-response
```

The tested physical response direction scales all NCSX coil currents together;
it is not a qualification of every coil-shape/current direction. Field query
points stay fixed in Cartesian space. The pilot does not yet optimize a field-
line objective or certify the region immediately outside the LCFS.

For installed-wheel verification, install the source-pair wheels into an empty
prefix and run `ESSOS/scripts/verify_installed_vmex.py --prefix <prefix>
--output <new-report.json>` through `srun`. It copies the four VMEX integration
tests to a temporary directory and requires successful execution with imports
from that prefix, including the native Cyna binary. The adapter/control-map tests
are also required by the seven-test source lane; repeated installed runs do not
increase the unique test count.

## Radial continuation qualification

The qualification script accepts `--radial-ladder 16 24 32 --ns 32`. Each
current factor starts independently at 16 surfaces, then interpolates the
accepted state onto each finer grid. Every stage must converge with finite,
nonnegative interior **and edge** residuals below the requested tolerance.
The ladder stops at the first rejected stage. Accepted coefficient arrays and
constraint anchors are saved in NPZ checkpoints with hashes; `continuation.json`
records every rung, current factor, residual, and iteration count.

The qualification-only helper `scripts/vmex_radial_continuation.py` uses the
recorded VMEX snapshot's private stage, linearization, and adjoint hooks. This
is needed because that snapshot's public multigrid driver does not expose the
edge-convergence option and its implicit wrapper does not expose a radial seed.
The helper binds the state, structural mask, and constraint anchors from the
same accepted solve to VMEX's native implicit VJP. It does not modify VMEX or
read/write its process-wide hot-start cache. Exact repeated controls reuse the
same accepted root; distinct controls run independent cold-to-fine ladders.
The independent finite-difference reference explicitly forces a fresh ladder.
VMEX's frozen inactive-mode derivative convention is retained and remains
subject to the ordinary re-solve qualification gate.

Run through an interactive/debug `srun` step:

```bash
.venv-vmex/bin/python ESSOS/scripts/qualify_vmex_freeboundary.py \
  --ns 32 --radial-ladder 16 24 32 \
  --grid 96 --planes 64 --coil-quadrature 256 --force-tolerance 1e-12 \
  --response --output run_artifacts/vmex-continuation
```

Two regression tests cover independence from current-evaluation history and
rejection of an unconverged edge even when the interior reports convergence.
They are included in the required VMEX lane, bringing its inventory to nine
checks (three overlap the original 262-test baseline).

The first ladder (interactive allocation `58719990`) converged at 16, 24, and
32 surfaces in 1194, 554, and 536 iterations respectively, with every interior
and edge residual below `1e-12`. The 32-surface beta is 5.482936%. Exact-control
state reuse gives zero base-state and base-field discrepancy. However, the
independent common-current response errors are **0.198891%, 0.227495%, and
0.233568%** at `h = 1e-3, 3e-4, 1e-4`. Thus radial continuation fixes the cold-
start convergence failure but does **not** qualify the response. All six
perturbed-current ladders also passed every stage's equilibrium/edge gate.

The complete report, checkpoint hashes, and coefficient arrays are retained in
[`li383_continuation_ns32`](../../benchmarks/manifold_optimization/results/vmex-integration/li383_continuation_ns32).
The updated required VMEX lane passed **9/9** checks; its manifest is
[`required_continuation_tests.json`](../../benchmarks/manifold_optimization/results/vmex-integration/required_continuation_tests.json).
Together with the earlier unchanged baseline this records 268 unique software
tests. Installed-wheel evidence remains the four exterior-adapter tests; these
new qualification scripts have not been presented as installed-wheel tests.


Continuation through 16→24→32→48→64 also converged, including all six
independent perturbed-current ladders. At 64 surfaces beta is 5.482566%, while
the three response errors are **0.164590%, 0.177401%, and 0.171085%**. Neither
higher-resolution run meets the unchanged 0.1% two-adjacent-step threshold.
See [`li383_continuation_ns64`](../../benchmarks/manifold_optimization/results/vmex-integration/li383_continuation_ns64).

### Attribution of the 32-surface discrepancy

`scripts/diagnose_vmex_continuation.py` reads the hashed checkpoints and checks
that the VMEX runtime and physical inputs match the original run. At `h=1e-4`,
the independently re-solved state changes in directions that the implicit
projector holds fixed. Their norm is 0.01478, compared with 1.43249 for the whole
state response. Both constraint-baseline derivatives are zero in this case.

For each Cartesian field component, the diagnostic computes the explicit
field contribution of that inactive-state drift and its implicit force-balance
contribution using the native coupled adjoint. Adding those contributions to
the original response reduces the unexplained relative difference from
`2.33568e-3` to **`6.29379e-6`** (0.233568% to 0.000629379%). This strongly
attributes this direction's discrepancy to the frozen-state convention,
rather than simply too few radial surfaces. It does not establish that every
physical direction has the same issue.

**This is not an independent derivative pass.** The diagnostic deliberately
uses state drift measured from the same finite-difference solves. It is not
used by `VmexExteriorField`, the qualification gate, or an optimizer. The next
response correction must make the solve and adjoint use a consistent state
convention without borrowing derivative information from the check itself;
then its accuracy should be assessed with ordinary independent re-solves.
The user has accepted the existing measured discrepancy for proceeding, so this
correction is deferred. Exterior-field accuracy and actual finite-beta design
acceptance remain separate requirements.
The diagnostic source and results are in
[`li383_continuation_attribution`](../../benchmarks/manifold_optimization/results/vmex-integration/li383_continuation_attribution).


## Exterior-domain qualification after response acceptance

`FourierLCFSExteriorDomain` classifies Cartesian points using the Fourier LCFS
at their actual toroidal angle. It rejects interior points and points too close
to the section polygon to resolve their side. Its polygon error bound is
`max|d²(R,Z)/dtheta²| * dtheta² / 8`, using a conservative Fourier coefficient
bound. A simple closed RZ section is required. Reported section clearance is
not a shortest three-dimensional wall clearance.

`qualify_vmex_domain.py` samples 72 outward-normal offsets per distance (six
angles per field period and twelve poloidal angles) from the accepted 32-surface
LI383 equilibrium. It compares source grids of 48², 96², and 144² points and
successive virtual-casing quadrature schedules ending at 192×96, 384×192, and
768×384. It uses direct exterior quadrature throughout; no near-surface
extrapolation has been enabled.

| Normal offset | Largest final field change, relative to local total B | Finest backend error estimate, relative to RMS surface B | Sampled points pass 0.1% field-value budget |
| --- | ---: | ---: | --- |
| 5 mm | 111.5% | 206.4% | No |
| 10 mm | 24.90% | 5.780% | No |
| 20 mm | 2.001% | 0.04039% | No |
| 50 mm | 0.006646% | 4.14e-7% | Yes |
| 100 mm | 1.09e-6% | below reported precision | Yes |
| 200 mm | 1.70e-13% | below reported precision | Yes |

These checks qualify the sampled points at the stated resolution, not every
point in a shell or a continuously traced volume. In particular, the optimistic
finest-grid estimate at 20 mm does not override the unresolved difference from
the preceding grid. A finer schedule or separately validated surface treatment
is needed closer to the LCFS. This field-value budget is independent of the
user-accepted equilibrium-response discrepancy.

The [domain report](../../benchmarks/manifold_optimization/results/vmex-integration/li383_exterior_domain/report.json)
retains every query point and refinement field. The two geometry regression
tests cover interior/exterior classification, unresolved boundary proximity,
helical toroidal dependence, clearance guards, and invalid Fourier modes.
The updated required VMEX lane passes **11/11** checks, recorded in
[`required_domain_tests.json`](../../benchmarks/manifold_optimization/results/vmex-integration/required_domain_tests.json).

Run the next checks inside an interactive/debug `srun` step:

```bash
.venv-vmex/bin/python ESSOS/scripts/qualify_vmex_domain.py \
  --equilibrium-run run_artifacts/vmex-20260922/li383-continuation-16-24-32 \
  --output run_artifacts/new-vmex-domain
.venv-vmex/bin/python ESSOS/scripts/qualify_vmex_trace.py \
  --equilibrium-run run_artifacts/vmex-20260922/li383-continuation-16-24-32 \
  --output run_artifacts/new-vmex-trace
```


### Native Cyna tracing of the accepted exterior field

The [native trace report](../../benchmarks/manifold_optimization/results/vmex-integration/li383_exterior_trace/report.json)
uses the accepted 32-surface equilibrium at a fixed launch
`(R,Z) = (1.4599202, 0.9512675) m`, from toroidal angle 0.1 to 0.125 rad.
The whole rectangular grid lies above the conservative global LCFS bound
`Z = 0.8312675 m`, including every toroidal sampling plane. This supplies an
explicit exterior-domain guarantee for this regression's field grid. It does
not qualify a divertor target, a full toroidal circuit, or a general near-LCFS
trajectory. The minimum sampled RZ-section clearance is about 0.372 m.

An independent DOP853 trajectory queries the finest direct VMEX field without
native field interpolation. Cyna uses the separately sampled cylindrical grid:

| Field grid | Maximum trajectory mismatch | Endpoint change when tracing steps double |
| --- | ---: | ---: |
| 5×5×32 | 0.156863 mm | 8.20e-7 mm |
| 9×9×64 | 0.049026 mm | 3.10e-7 mm |
| 13×13×128 | **0.011014 mm** | 2.51e-6 mm |

The final adjacent-grid endpoint change is 0.038012 mm. The final run passes
both the 0.1 mm comparison/grid-change gate and the 0.001 mm tracing-step gate.
Source/quadrature refinement along the reference trajectory differs by
`2.69e-16` relative to local field magnitude. Those tiny field-refinement
numbers do not replace the measured native interpolation error.

The first fine-grid attempt exposed repeated JAX compilation in VMEX's eager
error estimator and failed with an LLVM allocation error. `VmexExteriorField`
now holds stable compiled value/error sampling functions for each accepted
snapshot. The host still checks every batch's domain, finiteness and error
budget. The repeated run completed all three grids in about 88 s. The original
failure report is retained as
[`sampling_compilation_failure.json`](../../benchmarks/manifold_optimization/results/vmex-integration/sampling_compilation_failure.json),
and the updated adapter passes the complete **11-test** required lane in
[`required_cached_sampling_tests.json`](../../benchmarks/manifold_optimization/results/vmex-integration/required_cached_sampling_tests.json).
All numerical work used interactive allocation `58726960`.


Fresh isolated source-pair wheels also pass all **six installed tests** (four
field/response/native-trace checks plus two domain checks). Both installed
module hashes match the working source. See
[`installed_domain_wheels.json`](../../benchmarks/manifold_optimization/results/vmex-integration/installed_domain_wheels.json)
and [`domain_wheel_sources.json`](../../benchmarks/manifold_optimization/results/vmex-integration/domain_wheel_sources.json).
The qualification scripts themselves are source-run benchmarks, not installed
package entry points.


## First re-solved finite-beta optimization baseline

`run_vmex_trace_optimization.py` now couples the actual free-boundary
LI383/NCSX solve, coil-plus-plasma exterior field, native Cyna trajectory, and
transactional topology optimizer. The control is a common coil-current factor
bounded numerically to `[0.999, 1.001]`. Pressure/current profiles, plasma
current, toroidal flux, launch point and terminal toroidal angle stay fixed.
The nearby target is an independently solved current factor of `1.0003`.

This first endpoint-control baseline uses **central finite-difference
gradients**, with step `1e-4`, from complete independent 16→24→32 equilibrium
ladders and rebuilt exterior grids. It is the comparator for future implicit
trace gradients. Every candidate refresh likewise re-solves the equilibrium,
checks all interior/edge residuals against `1e-12`, checks every field-grid
point, and checks completion and exterior-domain retention of the Cyna trace.
The accepted optimizer state is checkpointed. The production script's problem
identity includes the equilibrium, trace qualification and numerical target.

Measured [results](../../benchmarks/manifold_optimization/results/vmex-integration/li383_trace_optimization/report.json):

- Two accepted optimization steps recover current factor **1.0003000555**.
- The scaled least-squares objective drops from **1.74999e-3 to 5.96279e-11**.
- Independent final and target re-solves on the 13×13×128 field grid, with 128
  tracing steps, differ by **1.094e-10 m** in endpoint position.
- All **13** complete equilibrium/field/trace evaluations pass acceptance;
  all **39** radial stages pass their force and edge checks.
- The measured run takes about **352 s** in interactive allocation `58726960`.

The extremely small endpoint difference measures recovery of a known
*numerical* control setting. Absolute trace error against the direct field is
about 0.011 mm in the preceding refinement study. This is a short exterior
endpoint demonstration, with no physical wall target, heat-load limit or
engineering design claim. Implicit endpoint-gradient integration, larger
control spaces and near-LCFS/wall-target cases remain future work. The prior
user-accepted implicit **field** response discrepancy has not been relabeled
as an implicit **trajectory** response test.

```bash
.venv-vmex/bin/python ESSOS/scripts/run_vmex_trace_optimization.py \
  --equilibrium-run run_artifacts/vmex-20260922/li383-continuation-16-24-32 \
  --qualified-trace run_artifacts/vmex-20260922/li383-exterior-trace-cached \
  --output run_artifacts/new-vmex-trace-optimization
```

The command must run through an interactive/debug `srun` step. The qualified
trace supplies the fixed grid axes and launch definition. Fresh output
directories preserve earlier success/failure evidence.
