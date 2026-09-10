# Manifold-aware Stage-2 coil optimization

ESSOS owns the Stage-2 coil optimization problem.  PyNA owns the magnetic
topology evaluated by that problem.  The dependency direction is one-way:
ESSOS may optionally import PyNA, while PyNA never imports ESSOS.

The bridge in `essos.manifold` passes either an ESSOS magnetic-field object or
an ESSOS `Coils` PyTree to PyNA's differentiable fixed-toroidal-angle map.  JAX
then differentiates the map with respect to the same coil curves and currents
used by the existing ESSOS engineering and normal-field objectives.

The intended Stage-2 sequence is:

1. Optimize normal field and engineering metrics with the existing ESSOS
   objectives.
2. Initialize and identify the desired edge topology with PyNA.
3. Ramp smooth PyNA return-map, X-line, and fixed-label manifold objectives
   inside the ESSOS loss.
4. Refresh the tracked topology in an outer loop and validate accepted designs
   using the production PyNA/Cyna tracing path.

The adapter is intentionally limited to the field and topology boundary.  The
first Stage-2 objective is `return_map_surface_loss`, which maps
fixed Stage-1 edge seeds with PyNA and penalizes their squared signed distance
from an ESSOS-supplied target surface on the destination section.  It composes
directly with `custom_loss`, so it can be ramped alongside normal-field and
engineering terms.

`periodic_xline_position` and `periodic_xline_state` now pass an ESSOS field to
PyNA's implicitly differentiable periodic-point solve.  On a Poincare section,
the returned point represents the toroidally continued X-line.  ESSOS supplies
three initial smooth objective terms:

- `periodic_xline_location_loss` places the tracked section point relative to a
  Stage-1 target with independent R and Z scales.
- `periodic_xline_hyperbolicity_loss` keeps the monodromy trace in a requested
  direct- or inverse-hyperbolic margin window.  The branch sign is fixed by the
  outer topology tracker, preventing an inner optimization from silently
  switching X-line identity.
- `periodic_xline_clearance_loss` samples the complete implicitly tracked
  periodic orbit and applies a normalized squared hinge to an ESSOS-supplied
  wall signed-distance function.  Distance is positive on the allowed side,
  and the duplicate closing point is omitted from the average.

The inner loss never decides whether Newton found the intended orbit.  Before
accepting an optimizer step, the outer loop must inspect the residual and
`converged` flag, update the initial guess, and compare against the production
PyNA/Cyna map.

The clearance term is likewise an inner sequential model, not a collision
certificate.  PyNA's JAX trajectory supplies derivatives on CPU or CUDA,
depending on the installed JAX backend.  Cyna retains global wall-intersection
authority during the outer accepted-design refresh.

## Parallel JAX and Cyna manifold paths

PyNA now exposes a differentiable JAX manifold tracer in parallel with its
production Cyna tracer.  Parallel here means two implementations of the same
labelled return-map problem, not that they must execute concurrently.  JAX can
itself run on CUDA when a CUDA-enabled JAX runtime is installed; Cyna retains
its CPU/CUDA production role for high-throughput and wall-aware tracing.

`essos_field_to_pyna_cylindrical_grid` closes the production side of this
split.  At an outer validation point it evaluates the current ESSOS field in
bounded JAX batches, converts Cartesian components to PyNA's canonical
`(BR, BZ, BPhi)` order, and materializes a host-side `VectorFieldCylind`
snapshot for Cyna.  This conversion is intentionally non-differentiable.  The
live ESSOS field continues through the JAX path for gradients; the grid
snapshot is used only for topology discovery, wall-aware tracing, and parity
checks.

The caller must choose an `(R, Z, Phi)` grid that encloses every candidate
orbit and manifold segment, covers one endpoint-free field period for the
declared `nfp`, and is converged in spatial resolution.  Leaving the grid is a
failed outer validation, not permission to extrapolate a topology label.

ESSOS does not reproduce PyNA's eigensystem or seed-spacing theory.  An outer
PyNA topology refresh selects the orbit, stability, branch side, consistently
oriented eigendirection, geometric seed distances, and sample correspondence.
ESSOS passes those frozen labels to `trace_manifold_branch`, which:

1. solves the current coil field's section anchor with PyNA's implicit root;
2. uses a backward-map anchor for stable branches;
3. asks PyNA to build the seed segment and trace all requested generations.

`manifold_sample_location_loss` then penalizes the normalized `(R,Z)` error of
one fixed `(generation, seed)` sample.  It composes with `custom_loss`, so a
continuation controller can ramp it beside normal-field and engineering terms.
Gradients include both X-line motion and the field dependence accumulated along
the manifold maps, while direction, spacing, side, and correspondence remain
stop-gradient outer state.

This sample-location loss remains a useful smooth lobe/divertor-leg placement
proxy.  For a wall-resolved target, `ManifoldStrikeStage2Target` stores PyNA's
exact `(branch, trace direction, sparse seed order)` strike label and the local
wall tangent plane frozen at the accepted Cyna hit.  Its ESSOS custom loss asks
PyNA's JAX backend to move the periodic X-line, rebuild that exact seed, trace
to the implicit local wall event, and differentiate the resulting hit point
through the live coil field.

The strike metric is explicit.  Axisymmetric tokamaks use `(R,Z)`, because the
toroidal hit phase is symmetry-degenerate.  QA and other non-axisymmetric
configurations use full Cartesian `(X,Y,Z)`.  The local plane is only an inner
sequential model: Cyna still owns global first-wall selection, branch identity,
connection length, and the refreshed plane after every accepted outer step.
The JAX loss never searches for a new hit or silently changes labels.

For heat loading, `ManifoldHeatStage2Target` extends the same frozen-label
contract to an ordered bundle of strikes.  PyNA supplies the exact
`(branch, direction, seed order)` labels and quantitative powers with explicit
provenance.  Its JAX backend solves the moving periodic X-line once and traces
all labelled local wall events; ESSOS then deposits those fixed powers onto a
fixed set of wall monitor cells with an area-normalized Gaussian kernel.  The
discrete inner model conserves power exactly,
`sum(heat_flux * cell_area) == sum(strike_power)`, and provides an
area-weighted squared-hinge penalty above a physical heat-flux limit.

This Gaussian footprint is a differentiable sequential approximation based on
local Cartesian chord distance.  It must not be interpreted as the production
heat map.  After a candidate coil step, PyNA/Cyna must retrace global first
hits, connection lengths, unresolved power, and wall deposition on the full
wall mesh; the fixed labels, powers, tangent planes, and monitor-cell model are
refreshed only after that outer calculation accepts the candidate.

`validate_manifold_strike_candidate` performs that outer strike transaction
after the candidate manifold branch has passed its production refresh.  It
rebuilds the branch's ordered strike-seed bundle, asks Cyna for global first
wall hits, refreshes the exact sparse seed-order label, projects the accepted
hit to a new local wall plane, and compares the live JAX event with Cyna using
the selected `rz` or `xyz` metric.  A missing hit, excessive motion, failed
periodic root or wall event, event-window escape, or excessive JAX/Cyna error
returns a rejected report with no refreshed target.

## Immutable target state and accepted refreshes

`ManifoldStage2Target` is the boundary between one outer topology refresh and
one differentiable inner solve.  It stores the accepted PyNA production branch,
one exact sample label, the physical target and normalization scales, and the
JAX tracing resolution.  `make_manifold_stage2_loss` captures that state in an
ESSOS `custom_loss`; it never searches for a closer sample or changes branch
identity while the optimizer is differentiating.

After a trial coil update, PyNA owns both acceptance gates:

1. `compare_jax_manifold_branch` verifies that the differentiable JAX samples
   still correspond to the production Cyna branch within the chosen tolerance.
2. `refresh_manifold_sample_match` carries the exact
   `(orbit, point, stability, initial side, generation, seed order)` label to
   the candidate field and enforces its displacement trust limit.  It rejects
   a missing sample instead of replacing it with a nearby one.

Only when both reports accept does `refresh_manifold_stage2_target` create the
next immutable snapshot.  The physical divertor target stays fixed; the branch
reference and the position associated with its exact label advance to the
accepted coil field.  A continuation driver should reject or shorten a trial
step when either report fails, then begin the next inner solve from the last
accepted snapshot.

## Fixed-weight continuation stages

`ManifoldContinuationSchedule` separates the ordinary Stage-2 coil objective
from the topology ramp.  Every named stage holds fixed weights for the
return-map, combined X-line, periodic-X-line wall-clearance, exact-label
manifold-sample, exact-label wall-strike, and quantitative heat-load terms.  The
caller chooses the dimensional normalization inside each objective and then
chooses these dimensionless continuation weights; no universal numerical ramp
is assumed.

`compose_manifold_stage2_loss` keeps the supplied normal-field and engineering
loss active with unit weight and adds only topology terms whose current weights
are positive.  In particular, the initial all-zero topology stage is exactly an
ordinary Stage-2 solve and does not compile or evaluate PyNA tracing.  A
combined X-line loss may include both location and hyperbolicity components.
The independently weighted `xline_clearance_loss` lets later continuation
stages introduce the full-orbit wall margin without changing those terms.
The manifold component is constructed internally from the active immutable
target so it cannot accidentally use a different sample label.  A positive
`strike_weight` likewise requires an immutable `strike_target_state` and builds
the local-wall loss internally; no wall tracing is compiled when that weight is
zero.  A positive `heat_weight` similarly requires an immutable
`heat_target_state`; an all-zero or pre-heat stage neither traces the strike
bundle nor compiles the deposition model.

After an inner solve, `accept_manifold_continuation_stage` calls the branch and
sample target refresh and advances to the next stage only if it succeeds.  If
the continuation carries a strike target, an independently accepted strike
snapshot is mandatory and must preserve its label, metric, and physical target.
If the continuation carries a heat target, advancement also requires an
accepted refreshed heat snapshot with the same ordered labels, absolute powers
and provenance, monitor cells, deposition width, and heat-flux limit.  Only the
production branch, strikes, and local tangent planes may move.
The function returns a new frozen state; an exception leaves the previous state
unchanged.  Coil degrees of freedom and optimizer rollback remain with the
calling ESSOS driver, while production retracing and acceptance decisions
remain with PyNA.

## Pre-strike leg clearance

`ManifoldLegStage2Target` stores a complete ordered bundle of exact strike
labels, local wall planes, integration controls, a static JAX wall-distance
callback, and the physical clearance requirements. It has no heat-power or
wall-deposition dependencies. Optional non-negative `strike_weights` affect
only the smooth objective; even a zero-weight leg must pass production
validation. The target copies those weights into an immutable tuple.

The signed-distance callback accepts Cartesian `(X,Y,Z)` and returns metres,
positive inside the allowed vessel. It and its captured wall geometry must
remain unchanged during an inner solve. Use the same physical wall for this
smooth model and the production validator.

For `N = wall_n_steps`, the inner loss retains fixed sample indices satisfying
`index / N < 1 - terminal_exclusion_fraction`, subsampled by `sample_stride`.
The excluded fraction is a fraction of the **signed toroidal-angle span**, not
arc length. The hit endpoint is always excluded. The loss is half the weighted
mean of each leg's mean squared normalized clearance violation. Fixed indices
keep the objective differentiable even as the hit angle moves.

```python
from essos.manifold_leg_optimization import ManifoldLegStage2Target
from essos.manifold_leg_validation import ManifoldLegValidationConfig

leg_target = ManifoldLegStage2Target(
    branch_reference=accepted_branch,
    strike_matches=accepted_matches,  # complete branch seed-order identity
    wall_planes=accepted_planes,
    wall_signed_distance=wall_signed_distance,
    minimum_clearance_m=1e-3,
    clearance_scale_m=1e-3,
    terminal_exclusion_fraction=0.2,
    maximum_phi_shift=0.1,
)
leg_validation = ManifoldLegValidationConfig(
    wall=production_wall,
    maximum_hit_displacement_m=2e-3,
    jax_cyna_tolerance_m=5e-4,
    maximum_endpoint_error_m=1e-5,
    production_DPhi=0.005,
)
```

These numbers illustrate configuration, not universal tolerances. Carry the
target as `ManifoldContinuationState.leg_target_state`, using the same branch
object as the sample target, and set `leg_clearance_weight` on the desired
stages. `compose_manifold_stage2_loss` constructs the loss from that snapshot.
A zero weight skips inner leg tracing completely. Pass `leg_validation_config`
to `validate_manifold_continuation_candidate`; carrying a leg target requires
its acceptance gate even during a zero-weight stage, as for strike/heat targets.

PyNA's `validate_manifold_leg_clearance` independently:

1. traces global first-wall hits for every exact labelled seed;
2. rejects missing hits, excessive strike motion, and unwrapped phase changes
   outside the trust window (including same-position hits on another turn);
3. samples each full Cyna seed-to-hit trajectory, checking signed phase order,
   launch identity, domain survival, span coverage, and endpoint agreement;
4. projects retained interior points onto the continuous wall, determines the
   sign by polygon containment in the periodic interpolated section, and checks
   every leg's minimum clearance;
5. rebuilds wall planes only for valid strikes.

ESSOS then checks the refreshed JAX local events against those production
strikes and independently enforces the smooth inner clearance margin. The
transaction returns `leg_validation` with production per-label reports, JAX
state, correspondence, and an accepted target only if all gates pass. Rejection
reasons are prefixed `leg_validation:` and use the existing backtracking path.
Stage acceptance preserves ordered labels, distance modes, wall callback,
weights, clearance requirements, exclusion fraction, and integration settings.

Clearance is a **sampled** certificate. Continuous wall projection does not
prove clearance between trajectory samples. Converge Cyna `production_DPhi`,
JAX `wall_n_steps`/`sample_stride`, the field grid, and wall resolution before
interpreting a margin physically. The terminal exclusion permits the intended
wall approach; it never disables the independent global first-hit check.
The production distance uses the closest poloidal wall segment at each query
angle, as in the existing X-line gate. It is not a global nearest-surface
distance in three dimensions. Polygon containment determines its sign, avoiding
normal-dot-product ambiguity at wall vertices.

## Candidate validation transaction

`validate_manifold_continuation_candidate` connects the pieces at an outer
step boundary.  Given a trial ESSOS field and an explicit production grid, it:

1. samples the trial field into a PyNA cylindrical snapshot;
2. asks PyNA/Cyna to continue the X-point and retrace the same sparse seed
   orders under anchor and tangent trust limits;
3. when X-line clearance is active, asks Cyna to sample the complete production
   orbit and PyNA to reject domain loss, closure drift, or insufficient signed
   clearance from the continuously interpolated wall;
4. when a heat target is carried, asks Cyna to retrace every labelled global
   first-wall hit and PyNA to reject unresolved power, label loss, wall-
   projection failure, excessive strike motion, or a production heat-flux
   limit violation, then traces the refreshed local events with JAX and applies
   the per-label bundle correspondence gate;
5. traces the candidate branch directly through the live ESSOS field with JAX;
6. asks PyNA to compare all available JAX/Cyna labels and refresh the one exact
   sample label under its displacement limit;
7. when a strike target is active, runs the global Cyna wall trace, exact strike
   refresh, local-plane rebuild, and metric-specific JAX/Cyna strike gate; and
8. when a leg target is carried, runs the independent production first-hit and
   leg-clearance gate, rebuilds its local events, checks JAX/Cyna strike parity,
   and enforces the smooth inner clearance margin; and
9. returns a new continuation state only when every required gate accepts.

The returned `ManifoldContinuationValidationReport` retains the individual
PyNA reports and a stable rejection reason.  A rejected report contains no
accepted state, so an ESSOS driver can keep the preceding coil field and reduce
its step or topology weight.  By default, Cyna branches may be incomplete
because wall termination is physical; the selected label must still exist.
Setting `require_complete_correspondence=True` is useful for domain-resolution
tests where every requested generation is expected to remain inside the grid.

## First tokamak coil checkpoint

The integration suite includes an eight-period tokamak-like field assembled
from discrete toroidal-field coils, an axisymmetric plasma-current loop, and a
PF coil.  The direct ESSOS/JAX map finds a hyperbolic X-line and differentiates
its unstable manifold with respect to the PF current.  The production path
samples the same Biot--Savart field on a `49 x 49 x 12` one-period cylindrical
grid, refines the grid field's own X-point with Cyna, and traces two manifold
generations with sparse seed orders.

Refining the production X-point is essential: seeding Cyna directly from the
live-field JAX root can introduce millimetre-scale drift because the
interpolated grid defines a slightly different discrete map.  With production
refinement, the test uses a conservative `2e-4 m` JAX/Cyna correspondence gate.
A small PF-current candidate remains within `2e-3 m` anchor and exact-sample
trust limits and advances the continuation state.

The next checkpoint solves a bounded one-degree-of-freedom target problem.  A
frozen production label identifies one unstable-manifold endpoint, JAX
differentiates its normalized `(R,Z)` residual with respect to the PF current,
and a scalar Gauss--Newton step proposes a current within explicit bounds.  The
proposal is accepted only after the full Cyna refresh, label, and JAX/Cyna
correspondence transaction.  The regression target is generated by a known
small PF-current perturbation, so the test can verify that optimization
recovers the control and reduces the differentiable target loss without
confounding the result with target-selection physics.

The wall-resolved checkpoint uses a larger `1e-2 m` sparse seed so the first
connection to a circular tokamak vessel is short enough for the live
Biot--Savart and interpolated production fields to be meaningfully compared.
Cyna selects seed order 11 and its first hit; ESSOS/JAX differentiates the same
label against a local wall plane using the axisymmetric `(R,Z)` metric.  The
measured JAX/Cyna discrepancy is about `2.6e-4 m`, inside an explicit
`5e-4 m` gate.

A known PF-current control of `1.0002` generates the strike target.  One scalar
Gauss--Newton step proposes approximately `1.00019998`, the full step passes
branch, sample, strike-label, wall-event, and JAX/Cyna validation, and the
normalized strike loss drops by roughly eight orders of magnitude.  This is a
solved physical tokamak coil-to-first-wall optimization loop.

The same checkpoint also varies a true coil-shape DOF: the vertical Fourier
offset of the PF loop.  A target generated by a `1e-3 m` height change is
recovered to about `2e-7 m` in one bounded Gauss--Newton proposal, while the
production X-line, manifold sample, exact Cyna wall hit, and JAX/Cyna strike
gate all remain valid.  This verifies that the strike derivative reaches ESSOS
curve geometry rather than only coil currents.

## First QA coil checkpoint

The QA integration checkpoint starts from the repository's physical
Landreman--Paul QA modular-coil data and adds sixteen explicit vertical
toroidal-field trim loops.  A `102.993 A` current in each weak trim loop places
a genuine period-five hyperbolic orbit at approximately
`(R,Z) = (1.09046,-0.14000) m`; its one-field-period residual remains large, so
the test cannot accidentally pass on the elliptic magnetic axis or a period-one
root.  The full configuration retains the QA field's `nfp=2` periodicity.

The production field uses a converged `257 x 313 x 160` one-period cylindrical
grid.  Cyna refines its own period-five X-line to within `0.1 mm` of the live
ESSOS/JAX root, and the labelled unstable branch agrees with JAX to about
`0.11 mm`.  The test wall is an explicitly non-axisymmetric, frozen regression
geometry rather than a proposed reactor vessel.  It protects all five points
of the closed X-line with a clearance envelope and adds a localized divertor
notch, then verifies that the X-line itself does not collide with the wall.

Cyna selects seed order 30 on the positive unstable branch.  Its first strike
occurs at approximately `phi = 6.486 rad`, after more than one full toroidal
turn, with a connection length near `7.13 m`.  Retaining this signed, unwrapped
flight angle is essential: folding it into one field period would send the JAX
event solver down the wrong trajectory segment.  With the corrected contract,
the local event is strongly transverse and the full Cartesian JAX/Cyna strike
discrepancy is about `0.77 mm`, inside the explicit `1.5 mm` gate.  The JAX
derivative of `(X,Y,Z)` with respect to a symmetry-preserving modular-coil
Fourier deformation also agrees with a centered finite difference to `3e-5`
relative tolerance.

The checkpoint then turns that derivative into a bounded shape solve.  A known
`10 micrometre` change to the first base coil's vertical sine coefficient
defines the Cartesian strike target.  One scalar Gauss--Newton proposal
recovers the coefficient within `5 nanometres`, reduces the normalized strike
loss by more than six orders of magnitude, and changes the affected physical
coil lengths by only about `31 micrometres`.  The full proposal passes the
production X-line, direction, manifold-sample, exact strike-label, local wall,
and JAX/Cyna correspondence gates without backtracking.

The same solve now carries a periodic-X-line clearance stage.  A frozen JAX
radial interpolant of the regression wall has zero hinge violation at the
`0.5 mm` requested margin before and after the shape step.  The independent
Cyna/PyNA gate traces all five field periods against the full wall: the
accepted candidate has about `1.32 mm` minimum continuous-section clearance
at the converged output spacing and `0.54 micrometres` orbit-closure error.
Native projection signs use polygon containment in the periodic interpolated
section, preserving inside/outside classification across `nfp=2` seams and at
notch vertices. Local wall-plane points and normals are still rotated back to
the signed, unwrapped strike phase.

It also carries the first physical-coil heat transaction with a deliberately
prescribed `1 W` regression load on seed order 30.  This absolute input is
labelled in provenance as a regression value, not a transport or reactor-power
prediction.  The ESSOS inner model deposits it on a five-cell local tangent
patch and conserves the watt before and after the accepted shape step.  The
independent PyNA/Cyna gate retraces the global hit, deposits exactly `1 W` on a
`64 x 128` full-wall grid with zero unresolved power, stays below the explicit
`1e5 W/m^2` regression limit, and keeps the bundled local-event discrepancy
below `1 mm`.

The step now also passes pre-strike leg clearance: the production minimum is
about `30.91 mm` at a requested `0.5 mm` margin, excluding the final 20% of the
toroidal-angle span. Halving production integration/output spacing changes
the nominal minimum by about `1.1 nm`. An excessive `40 mm` requirement is
explicitly rejected. See the
[clearance milestone report](manifold_leg_clearance_achievement_report.md)
for test evidence and the distinct inner/production distance definitions.

This closes the first physical QA coil/wall and three-dimensional strike
checkpoint, including a true QA modular-coil shape step and periodic-X-line
clearance plus an end-to-end prescribed-power heat gate.  A transport-derived
QA power allocation and multi-DOF
engineering-constrained optimization remain subsequent milestones.

## Optimizer proposal rollback

`validated_manifold_backtracking_step` is the narrow adapter from an arbitrary
optimizer to the outer validation transaction.  It accepts the last approved
flat ESSOS DOFs, an optimizer proposal, a field reconstruction callback, and a
candidate-validation callback.  Trial fractions start at one and contract
toward the accepted DOFs until topology validation succeeds or the attempt
budget is exhausted.

Every contraction is tested against the same preceding topology snapshot.  An
accepted attempt returns its trial DOFs, field, and PyNA-approved next
continuation state.  If every attempt is rejected, the result has zero accepted
step fraction and reconstructs the unchanged input field and state.  Backend or
configuration exceptions propagate instead of being treated as reasons to
shrink a physically meaningful coil step.  This keeps proposal generation with
SciPy, JAXopt, or Optax while keeping topology-based acceptance with PyNA.
