# Implementation evidence and remaining gates

The user requested the full plan with accessible tokamak and LI383 starting
inputs. This ledger distinguishes implemented software from qualified physical
results. The complete research/release plan is **not yet achieved**.

## Implemented software

- Immutable branch geometry and nested metadata, non-executable serialization,
  independent accepted snapshots, fixed problem identities and cache keys.
- Periodic-root conditioning and response validity, moving oriented eigenpair
  JVP/VJP, moving-linear-seed propagation through ESSOS manifold/strike/heat/leg
  APIs, and seed-size representation-error checks.
- Explicit finite-turn/long-time transform semantics and full-metric covariant
  field-to-field-line-velocity response; historical disputed helpers remain
  clearly limited rather than silently changing their mathematical meaning.
- General open launches, moving walls/launches/weights, connection-length
  response, an ESSOS objective adapter, and native Cyna first-hit/whole-leg
  acceptance with Cartesian 3-D clearance certificates.
- Gauged invariant-circle correction, fixed signed section-area labels, dense
  residual and regularity checks, implicit JVP/VJP and predictor, separate
  approximate-surface KKT response, and sampled multi-circle nesting checks.
- Scaled constrained Gauss–Newton optimization, refreshed physical merit and
  feasibility, transactional acceptance, atomic checkpoints, rollback, restart,
  and constrained-stationarity diagnostics.
- Actual production-refreshed manifold and general-open objective adapters,
  closed-surface objectives, reduced equilibrium response, and joint
  equilibrium/topology acceptance.
- Fixed-current geometry/grid caching, strict NCSX coil import, fixed scalar
  current normalization (fixing a JAX cache deadlock), packaged physical inputs,
  required-backend audit runner, and isolated installed-wheel verification.

The final required inventory contains 262 tests across analytic, production,
QA-coil and tokamak lanes. All 262 passed in
`run_artifacts/verified-20260921/`; three targeted rechecks also pass after the
final physical-distance and interior-launch guards. Individual new integration checks include actual
Cyna open-line optimization and joint manufactured equilibrium/surface control.

## Physical results and limitations

See [measured results](results.md) for artifact links and numerical values.
The tokamak prescribed-background example controls PF current and vertical
position. NCSX uses three vacuum-coil current factors with a fixed signed section
area, a transform target, numerical current bounds and a current-sum equality.
Implicit, re-solved finite-difference, and direct constrained surface methods
have all produced valid refined results. Neighboring surfaces and a separate
native-grid refinement provide additional checks.

The supplied LI383 input retains its pressure and current profiles. Reduced
fixed-boundary primal solves converge. Tokamak material-grid field response has
passed an independent finite-difference check. LI383 response fails the 0.1%
relative-error gate even at a force tolerance of 1e-18. The experimental residual
response also fails its independent base-field check. These failures are kept
visible; LI383 response is not qualified for design acceptance.

## Selected finite-beta backend

VMEX is integrated through an exterior-field adapter with domain/quadrature
acceptance, parameter VJP/JVP, tied plasma/direct-coil controls, and validated
pyna/Cyna grid sampling. The newer isolated environment passes the full 262-test
baseline and eight additional VMEX-specific tests (270 unique tests across the
recorded runs). The eleven-test VMEX lane includes three native regressions already
counted in the baseline; its current inventory passes without skips.

Actual tokamak and LI383 exterior-field solves and response checks are recorded
in the [VMEX report](vmex_integration.md). The coil-driven LI383 free-boundary
pilot converges with edge-force checks and beta about 5.48%. Boundary-response
and coupled-coil-response qualifications are distinguished, including VMEX's
frozen solver-DOF convention and warm-start repeatability. Qualification of a
complete finite-beta optimization remains separate from a converged primal.
At 16 radial surfaces and force tolerance 1e-12, the finer two common-current
perturbations disagree by 0.124% and 0.122%, above the 0.1% response gate. The
32-surface cold-start attempt did not converge. A new 16→24→32 radial
continuation converges with edge checks below 1e-12, but its three ordinary
re-solve comparisons still fail (0.199%, 0.227%, 0.234%). Continuation to 64
surfaces also converges but fails (0.165%, 0.177%, 0.171%). A separate
32-surface diagnostic attributes most of the discrepancy to state directions
held fixed by the adjoint; it uses finite-difference state drift and does not
constitute independent qualification. The user accepts the measured 0.17–0.23%
response discrepancy for proceeding; the strict historical results remain
unchanged. Exterior-domain refinement now passes at the sampled 5, 10 and 20 cm
normal offsets; 2 cm and closer remain unresolved at the tested quadrature.
A short real-VMEX exterior trace passes native Cyna versus direct-DOP853
validation at 0.011 mm, with separate grid and tracing-step refinement.
Sampling now reuses stable compiled kernels after an initial repeated-
compilation memory failure; the 11-test lane passes after the fix.
Fresh installed wheels pass all six field/response/native-trace and domain
tests, with installed module hashes matching the working source.

The first actual finite-beta endpoint-control baseline now succeeds in two
accepted steps, recovering a nearby common-current factor using independently
re-solved finite-difference gradients. All 13 equilibrium/field/trace evaluations
and 39 radial stages pass. Final and target cases are independently re-solved
and traced on a finer grid. This is numerical endpoint recovery; implicit trace
gradients and a physical wall-target optimization remain separate work.

## Open release gates

- **G4 physical open-field demonstration:** the tokamak example is a prescribed
  background regression. A small NCSX vacuum first-hit recovery benchmark now
  uses actual modular coils and the symmetry-expanded vessel with whole-leg
  acceptance; its target is generated from a nearby known control setting.
  Independent divertor targets, validated physical error budgets and engineering
  bounds are still needed. The coil/vessel association with finite-beta LI383
  is not established by their names alone.
- **G5 scientific comparison:** reduced NCSX surface control and a strong direct
  comparator work. Larger control spaces, additional surfaces/perturbations,
  repeated timing trials and uncertainty budgets are needed for broader claims.
  No universal speedup is asserted from one reduced benchmark.
- **G6 release integration:** installed wheels are exercised locally. Paired
  committed revisions and an updated pyna pin are required before publishing;
  remote CI and a published release have not been performed.
- **G7 physical finite-beta design:** complete VMEX total exterior-field response
  checks as the control space expands beyond the user-accepted common-current
  pilot; extend exterior-domain sampling, integrate implicit trace gradients,
  and progress from the successful re-solved endpoint recovery to a physical
  wall-target design.
  The manufactured joint adapter test does not close this physical gate.

No engineering current, stress, clearance or heat-load limits were available in
the selected equilibrium decks. All demonstration limits are explicitly numerical.
No production run was executed on a login node. Builds and numerical checks use
Perlmutter interactive allocations 58705967, 58712906, 58716831, 58719990 and 58726960 (the earlier
baseline used its own recorded allocation).
