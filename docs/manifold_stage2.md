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
manifold-sample, and exact-label wall-strike terms.  The caller chooses the
dimensional normalization inside each objective and then chooses these
dimensionless continuation weights; no universal numerical ramp is assumed.

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
zero.

After an inner solve, `accept_manifold_continuation_stage` calls the branch and
sample target refresh and advances to the next stage only if it succeeds.  If
the continuation carries a strike target, an independently accepted strike
snapshot is mandatory and must preserve its label, metric, and physical target.
The function returns a new frozen state; an exception leaves the previous state
unchanged.  Coil degrees of freedom and optimizer rollback remain with the
calling ESSOS driver, while production retracing and acceptance decisions
remain with PyNA.

## Candidate validation transaction

`validate_manifold_continuation_candidate` connects the pieces at an outer
step boundary.  Given a trial ESSOS field and an explicit production grid, it:

1. samples the trial field into a PyNA cylindrical snapshot;
2. asks PyNA/Cyna to continue the X-point and retrace the same sparse seed
   orders under anchor and tangent trust limits;
3. when X-line clearance is active, asks Cyna to sample the complete production
   orbit and PyNA to reject domain loss, closure drift, or insufficient signed
   clearance from the continuously interpolated wall;
4. traces the candidate branch directly through the live ESSOS field with JAX;
5. asks PyNA to compare all available JAX/Cyna labels and refresh the one exact
   sample label under its displacement limit;
6. when a strike target is active, runs the global Cyna wall trace, exact strike
   refresh, local-plane rebuild, and metric-specific JAX/Cyna strike gate; and
7. returns a new continuation state only when every required gate accepts.

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
Wall projections are rotated back
to the signed, unwrapped orbit phase before their inside/outside sign is used,
which is essential after crossing an `nfp=2` field-period seam.

This closes the first physical QA coil/wall and three-dimensional strike
checkpoint, including a true QA modular-coil shape step and periodic-X-line
clearance.  Wall heat loading, pre-strike manifold-leg clearance, and multi-DOF
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
