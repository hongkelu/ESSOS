# Manifold-aware coil optimization: achievement report

Update, 2026-09-10: the pre-strike leg-clearance milestone described below as
uncommitted work is now implemented and verified. See the
[leg-clearance achievement report](manifold_leg_clearance_achievement_report.md)
for current APIs, production gates, physical evidence, and the wall-vertex
sign correction discovered during integration. The remainder of this document
preserves the earlier snapshot and its original branch/test counts.

Snapshot date: 2026-09-10 (Asia/Shanghai)

## Executive summary

We now have a working, tested architecture for adding invariant-manifold and
X-point divertor objectives to coil optimization during ESSOS Stage 2, with a
strict division of responsibility:

- **ESSOS owns the coil-design problem:** coil curves and currents, magnetic
  field evaluation, engineering and normal-field losses, continuation weights,
  optimizer proposals, and rollback.
- **PyNA owns magnetic topology:** fixed-angle field-line maps, periodic
  X-lines, monodromy and hyperbolicity, stable/unstable manifold labels,
  first-wall strikes, connection lengths, and topology validation.
- **JAX is the differentiable inner backend:** it supplies fixed-shape smooth
  objectives and derivatives with respect to live ESSOS coil degrees of
  freedom. It can run on CPU or on CUDA when the installed JAX backend supports
  CUDA.
- **Cyna is the production outer backend:** its CPU/CUDA implementation remains
  authoritative for global tracing, first-wall selection, wall-aware heat
  maps, and accepted-step validation.

The two implementations are parallel descriptions of the same labelled
topology; “parallel” does not require simultaneous execution. PyNA does not
import ESSOS. ESSOS optionally consumes PyNA through a Cartesian magnetic-field
callable and materializes a non-differentiable cylindrical grid only for the
production validation path.

At this snapshot:

| Repository | Dedicated worktree | Local branch | Implementation progress | Status |
| --- | --- | --- | ---: | --- |
| PyNA | `/home/lhk/uwplasma/pyna-manifold` | `manifold-optimization` | 24 commits above `origin/manifold-optimization` | Clean |
| ESSOS | `/home/lhk/uwplasma/ESSOS-manifold` | `manifold-optimization` | 27 commits above `origin/manifold-optimization` | Pre-strike clearance work in progress |

Excluding this report-only commit and the uncommitted work in progress, the
implementation comprises 51 commits, 52 changed files, 16,249 insertions, and
26 deletions relative to the two remote branch bases. It adds 68 PyNA and 60
ESSOS test functions. Nothing has been pushed.

## Scientific and numerical formulation

### Periodic X-line

For coil parameters `p`, PyNA solves the period-`m` return-map equation

```text
F(x, p) = P_p^m(x) - x = 0.
```

The JAX periodic solver uses a fixed-count Newton solve in the primal path and
an implicit custom derivative,

```text
dx*/dp = -(D_x F)^(-1) D_p F,
```

so optimization derivatives do not depend on differentiating through the
Newton iteration history. The returned state includes the residual,
monodromy, Greene residue, eigenpairs, and hyperbolicity diagnostics. The outer
loop rejects non-converged or near-parabolic candidates.

### Stable and unstable invariant manifolds

At a hyperbolic periodic point, the monodromy eigenvectors define the stable
and unstable tangent directions. PyNA freezes the accepted branch orientation,
side, sparse seed orders, and geometric seed distances during one inner solve.
JAX then maps the fundamental segment forward for an unstable branch or
backward for a stable branch, retaining fixed-shape generations

```text
(generation, sparse seed order, R, Z).
```

This brings invariant-manifold theory into coil optimization without allowing
a nearest-point search, eigenvector sign flip, seed reordering, or branch swap
inside a gradient evaluation. Those discrete choices belong to the production
outer refresh.

### Moving wall intersection

For a frozen local wall model `g`, a labelled manifold seed is traced to the
event

```text
g(R(phi, p), Z(phi, p), phi) = 0.
```

The JAX event solve also uses an implicit derivative. Consequently, a strike
gradient contains all of the following:

1. motion of the periodic X-line;
2. motion of the invariant-manifold launch point;
3. deformation accumulated along the field-line trajectory; and
4. motion of the terminal wall-hit angle.

PyNA/Cyna still decides which seed hits, whether the hit is the global first
hit, and whether the local event remains a valid representation of it.

### Smooth Stage-2 objectives

The implemented inner objectives cover:

- return-map surface alignment;
- periodic-X-line location;
- direct- or inverse-hyperbolicity margin;
- clearance of the complete periodic X-line from the wall;
- placement of one exact labelled invariant-manifold sample;
- placement of one exact labelled first-wall strike; and
- a power-conserving heat-flux-limit surrogate for an ordered strike bundle.

For strike powers `P_i`, fixed wall-cell areas `A_c`, and a Gaussian footprint,
the cell fractions are normalized before flux is computed. Therefore the
inner heat model satisfies

```text
sum_c(q_c A_c) = sum_i(P_i)
```

to floating-point precision, even when all unnormalized Gaussian values would
underflow. This is a differentiable sequential model, not the production heat
map.

The current in-progress pre-strike objective evaluates a wall signed distance
on the retained interior samples of every exact labelled seed-to-strike leg.
Its proposed loss is

```text
L_leg = 1/2 * weighted_mean_i(
    mean_j(max((d_min - d(x_ij)) / d_scale, 0)^2)
).
```

A fixed terminal path fraction is excluded because the desired divertor leg
must approach and intersect the wall. Fixed sample indices keep this selection
differentiable; no live-field collision mask enters the inner solve.

## End-to-end optimization and acceptance loop

```text
ordinary ESSOS Stage 2
        |
        v
PyNA/Cyna identifies and labels the accepted topology
        |
        v
ESSOS freezes an immutable target snapshot
        |
        v
JAX differentiates smooth topology losses through live coil DOFs
        |
        v
ESSOS proposes a coil-current or coil-shape step
        |
        v
PyNA/Cyna globally retraces and validates the candidate
        |
        +-- reject --> ESSOS backtracks; old labels/state remain authoritative
        |
        +-- accept --> refresh immutable labels/planes and advance continuation
```

Topology weights can be zero in the initial Stage-2 stage, exactly recovering
the ordinary ESSOS problem without compiling PyNA traces. Later stages ramp
return-map, X-line, clearance, manifold-sample, strike, and heat terms while
normal-field and engineering objectives stay active. A topology objective may
also be run after an ordinary Stage-2 solve, but the implemented continuation
machinery supports using it during Stage 2 as requested.

## Completed PyNA work

### 1. Differentiable fixed-angle field-line map

- Added a JAX fixed-toroidal-angle RK4 trajectory and Poincare map.
- Added map Jacobians through automatic differentiation.
- Made the signed map span explicit, supporting forward and backward maps.
- Added Cyna/JAX parity tests.
- Retained the current assumption that geometric toroidal angle is monotone
  and `B_phi` does not vanish on the trace.

Primary module: `pyna/toroidal/flt/jax_poincare.py`.

### 2. Implicit periodic points and periodic trajectories

- Added an implicitly differentiated periodic-point root solve.
- Added residual, convergence, monodromy, Greene-residue, eigenvalue, and
  hyperbolicity state.
- Kept eigenvector ordering/sign as stop-gradient outer diagnostic data while
  preserving derivatives of the root and monodromy invariants.
- Added full differentiable trajectories of period-`m` X-lines.

Primary module: `pyna/topo/jax_periodic.py`.

### 3. Differentiable invariant-manifold branches

- Added stable and unstable monodromy directions.
- Added geometrically spaced fundamental segments.
- Added fixed-label forward/backward manifold generations.
- Added Cyna/JAX generation parity reporting.
- Preserved sparse production seed orders instead of renumbering retained
  seeds as dense indices.

Primary module: `pyna/topo/jax_manifold.py`.

### 4. Persistent topology identity and production refresh

- Added immutable sample labels carrying orbit identity, stability, initial
  side, generation, and sparse seed order.
- Distinguished physical point side from initial seed side for
  orientation-reversing saddles.
- Added exact-label sample selection, refresh, displacement trust regions, and
  JAX/Cyna correspondence reports.
- Added a production refresh that continues the X-point, recomputes Cyna
  monodromy, enforces hyperbolicity and anchor trust, retraces the same sparse
  seed orders, and locks the eigendirection to its preceding orientation.
- Returned structured rejection reports for topology loss rather than silently
  selecting a replacement branch or point.

Primary modules: `pyna/topo/manifold_correspondence.py` and
`pyna/topo/manifold_refresh.py`.

### 5. Exact first-wall strike identity

- Added strike identities `(branch label, trace direction, sparse seed order)`.
- Added production selection and exact-label refresh of Cyna first-wall hits.
- Added explicit strike metrics: `(R,Z)` for axisymmetric tokamaks and full
  Cartesian `(X,Y,Z)` for QA and other three-dimensional configurations.
- Added local tangent-plane construction from continuously projected wall
  geometry.
- Preserved signed, unwrapped hit phase, including strikes after more than one
  toroidal turn.

Primary modules: `pyna/topo/manifold_strike.py` and
`pyna/topo/manifold_strike_contracts.py`.

### 6. Differentiable single and bundled wall strikes

- Added implicit local wall-event solves with convergence, residual, event
  window, phase-shift, and transversality diagnostics.
- Added a full chain from the moving X-line through the exact manifold seed to
  the moving strike.
- Added a vectorized ordered strike bundle that solves the periodic root once.
- Added per-label and aggregate JAX/Cyna strike correspondence gates.
- Added full fixed-step seed-to-event trajectories whose final samples agree
  with the implicit intersections to backend precision.

Primary modules: `pyna/topo/jax_strike.py` and
`pyna/topo/manifold_strike_correspondence.py`.

### 7. Production clearance and heat gates

- Added Cyna sampling of the complete periodic X-line and continuous wall
  projection checks.
- Fixed inside/outside clearance sign across field-period seams by rotating
  projections back to the signed, unwrapped orbit phase.
- Added global first-wall heat validation with exact labels, absolute powers,
  explicit provenance, unresolved-power accounting, full-wall binning,
  discrete power conservation, wall-projection checks, and a physical peak
  heat-flux limit.

Primary modules: `pyna/topo/xline_clearance.py` and
`pyna/topo/manifold_heat.py`.

## Completed ESSOS work

### 1. One-way field bridge

- Added direct adaptation of an ESSOS field object or `Coils` PyTree to PyNA's
  Cartesian JAX field callable.
- Added fixed-angle maps and periodic/manifold/strike adapters without adding
  an ESSOS dependency to PyNA.
- Added bounded-batch sampling of a live ESSOS field into PyNA's canonical
  `(BR, BZ, BPhi)` cylindrical grid for non-differentiable Cyna validation.

Primary module: `essos/manifold.py`.

### 2. Differentiable Stage-2 losses

- Added return-map surface residual and loss.
- Added periodic-X-line location and hyperbolicity losses.
- Added full-period X-line wall-clearance loss.
- Added exact-label manifold-sample location loss.
- Added exact-label local-wall strike loss.
- Added area-weighted heat-flux-limit loss with exactly power-conserving
  Gaussian deposition on fixed monitor cells.
- Wrapped each objective as an ESSOS `custom_loss`, differentiating through the
  same field/coils PyTree used by existing Stage-2 objectives.

Primary modules: `essos/manifold.py`, `essos/manifold_optimization.py`,
`essos/manifold_strike_optimization.py`, and
`essos/manifold_heat_optimization.py`.

### 3. Immutable continuation state

- Added named continuation stages with independent weights for the return map,
  X-line, X-line clearance, manifold sample, strike, and heat terms.
- Kept the base Stage-2 loss active at unit weight.
- Avoided topology tracing and compilation for zero-weight terms.
- Required accepted refreshed target snapshots before advancing a stage.
- Preserved physical targets, distance metrics, ordered labels, powers,
  provenance, wall cells, deposition width, and heat limit across refreshes.

Primary module: `essos/manifold_optimization.py`.

### 4. Candidate validation transaction and rollback

- Added a single outer transaction joining field-grid sampling, production
  X-point continuation, sparse manifold retracing, X-line clearance, heat,
  live JAX branch comparison, exact sample refresh, and exact strike refresh.
- Added stable, structured rejection reasons and no accepted state on failure.
- Added optimizer-independent backtracking that tests contracted proposals
  against the same preceding topology snapshot and returns the unchanged
  design if every attempt is rejected.

Primary modules: `essos/manifold_validation.py`,
`essos/manifold_strike_validation.py`, and `essos/manifold_driver.py`.

## Physical integration checkpoints

### Axisymmetric tokamak

The tokamak regression uses discrete toroidal-field coils, an axisymmetric
plasma-current loop, and a PF coil.

Completed checks include:

- a live ESSOS/JAX hyperbolic X-line and unstable manifold derivative;
- a `49 x 49 x 12` one-period production grid and Cyna-refined grid X-point;
- sparse labelled manifold correspondence under a `2e-4 m` JAX/Cyna gate;
- a bounded one-DOF PF-current manifold target solve;
- an exact seed-order-11 first-wall strike on a circular vessel;
- an axisymmetric `(R,Z)` strike metric and approximately `2.6e-4 m` JAX/Cyna
  strike discrepancy under a `5e-4 m` gate;
- recovery of a known PF-current target near `1.0002`, with approximately
  eight orders of magnitude reduction in normalized strike loss; and
- recovery of a true PF-loop vertical Fourier displacement to about `2e-7 m`
  while all production topology gates remain valid.

This establishes a physical coil-to-X-line-to-manifold-to-first-wall
optimization path for a tokamak, including both current and curve-shape DOFs.

### Quasi-axisymmetric stellarator

The QA regression starts from the Landreman--Paul modular coils and adds 16
weak vertical toroidal-field trim loops.

Completed checks include:

- a genuine period-five hyperbolic orbit near
  `(R,Z) = (1.09046,-0.14000) m` while retaining `nfp=2` geometry;
- a converged `257 x 313 x 160` one-period production grid;
- Cyna/JAX X-line agreement within `0.1 mm` and labelled manifold agreement of
  about `0.11 mm`;
- a non-axisymmetric regression wall that clears all five X-line points and
  contains a localized divertor notch;
- exact unstable seed order 30, a first strike near unwrapped
  `phi = 6.486 rad`, and connection length near `7.13 m`;
- a full Cartesian strike discrepancy of about `0.77 mm` under a `1.5 mm`
  gate;
- agreement of the strike derivative with centered finite differences to
  `3e-5` relative tolerance;
- recovery within `5 nm` of a known `10 micrometre` modular-coil Fourier
  perturbation, more than six orders of magnitude loss reduction, and only
  about `31 micrometres` change in affected physical coil lengths;
- complete period-five production X-line clearance of about `1.32 mm`, orbit
  closure error of about `0.54 micrometres`, and a satisfied `0.5 mm` inner
  clearance margin; and
- an end-to-end prescribed `1 W` heat transaction with zero unresolved power,
  exact inner and outer power conservation, production peak below the
  `1e5 W/m^2` regression limit, and bundled local-event discrepancy below
  `1 mm`.

This establishes the first physical QA coil/wall checkpoint with a true
three-dimensional strike and a symmetry-preserving coil-shape step.

## Verification evidence

The committed branches add 128 test functions relative to their remote bases:

| Area | Added test functions | Main coverage |
| --- | ---: | --- |
| PyNA | 68 | JAX maps, implicit roots, manifold generations, Cyna parity, labels, production refresh, strikes, clearance, heat |
| ESSOS | 60 | adapters, losses, continuation, rollback, outer transactions, tokamak coils, QA coils, strikes, clearance, heat |

Recorded milestone runs during implementation include:

- 27 related PyNA strike/manifold tests passing in `167.49 s` after full
  labelled strike trajectories were added;
- the full physical QA heat transaction passing in `216.98 s`; and
- endpoint agreement for retained strike-leg trajectories to approximately
  `2e-13`, with an interior directional derivative matching finite differences
  at `1e-6` relative tolerance.

The two development environments currently split optional dependencies. The
combined regression command used during integration is:

```bash
env PYTHONPATH=/home/lhk/uwplasma/ESSOS-manifold:/home/lhk/uwplasma/pyna-manifold:/home/lhk/.conda/envs/pyna-env/lib/python3.10/site-packages \
  /home/lhk/.conda/envs/uwplasma-env/bin/python -m pytest ...
```

This does not mutate either environment. A CUDA-enabled JAX installation is
not required for correctness tests; it changes the JAX execution device, not
the ownership contract.

## Current uncommitted milestone

The next milestone, differentiable pre-strike invariant-manifold-leg
clearance, is partially implemented in the ESSOS worktree and is deliberately
not counted among the 27 completed ESSOS commits.

Current files:

```text
M  essos/manifold.py
M  tests/test_manifold_heat_optimization.py
?? essos/manifold_leg_optimization.py
```

Work already present in this uncommitted change:

- an ESSOS adapter to PyNA's committed full labelled strike trajectories;
- a signed-distance state and squared-hinge loss over fixed pre-strike samples;
- a fixed terminal exclusion fraction;
- optional per-strike weighting, including use of absolute strike powers;
- an ESSOS `custom_loss` builder with the wall callback captured as static
  configuration; and
- focused tests for shape, endpoint identity, terminal exclusion, invalid
  configuration, live field gradients, finite-difference agreement, and
  direct/custom-loss parity.

The focused test run was interrupted before a complete result was recorded.
Therefore this ESSOS milestone must be treated as **in progress**, not
verified or committed.

## Remaining work, in recommended order

1. Finish and verify the ESSOS differentiable pre-strike leg-clearance loss;
   document it and commit it as an isolated milestone.
2. Add PyNA's authoritative production leg-clearance validator using exact
   Cyna-labelled legs, signed/unwrapped phase, continuous seam-aware wall
   projection, the same fixed terminal exclusion convention, and per-label
   reports.
3. Add a dedicated continuation weight, immutable target/config state, outer
   acceptance gate, rollback reason, and refresh rules for leg clearance.
4. Extend the physical QA checkpoint so the accepted coil-shape step passes
   both the differentiable inner leg margin and the independent production
   Cyna clearance certificate.
5. Add corresponding tokamak coverage, including both stable and unstable
   divertor legs where appropriate.
6. Replace the prescribed QA `1 W` regression allocation with a clearly
   defined transport-derived absolute power model. Keep provenance and power
   conservation mandatory.
7. Move from the present bounded one-DOF regression solves to multi-DOF,
   engineering-constrained coil optimization with realistic step sizes and
   continuation schedules.
8. Validate grid, integration, wall-mesh, footprint-width, and label-trust
   convergence before interpreting any optimized design physically.

## Important limitations and interpretation

- The QA wall is a frozen non-axisymmetric regression geometry, not a proposed
  reactor first wall or divertor design.
- The QA `1 W` load is prescribed regression data, not a transport prediction
  or reactor power estimate.
- The Gaussian heat footprint is a local differentiable surrogate. The global
  Cyna wall trace and production wall binning are authoritative.
- Local tangent planes are valid only inside their outer trust regions and are
  rebuilt after accepted production refreshes.
- Fixed-angle integration assumes monotone toroidal angle and non-vanishing
  `B_phi`. More general fields require a Cartesian periodic-curve
  formulation.
- The current regression optimizations demonstrate differentiability,
  correspondence, validation, and rollback. They do not yet demonstrate a
  reactor-relevant engineering optimum.
- A topology label is never allowed to change silently inside differentiation.
  Any relabelling is an explicit outer-loop decision.

## Branch provenance

### PyNA

- Remote: `https://github.com/WenyinWei/pyna.git`
- Remote base: `32d04ed07cf18fee14f4948f1cc9c45f44fcf878`
- Implementation head: `69d0836f36b88d8646ea8047f342aa4fc5707334`
- Committed diff: 33 files, 7,793 insertions, 26 deletions

Commits, newest first:

```text
69d0836 feat(topo): trace full labelled strike legs
f1a8d21 feat(topo): compare labelled strike bundles
3a3ae33 feat(topo): validate manifold heat loading
13032cf feat(topo): trace labelled strike bundles
afdb50f fix(topo): preserve clearance sign across nfp seams
509492f feat(topo): validate production X-line clearance
3520d12 feat(topo): trace differentiable periodic orbits
87f57dc fix(topo): preserve unwrapped wall-hit phase
b6362e9 feat(topo): validate differentiable strike parity
da4c6fc feat(topo): distinguish tokamak and 3D strike metrics
9b2f4ab feat(topo): trace labelled manifold strikes
cff18c7 feat(topo): freeze local wall planes at strikes
adb16b1 feat(topo): add differentiable wall intersections
54156e1 feat(topo): preserve first-wall strike labels
631a491 feat(topo): add production manifold refresh gate
a5b51c1 feat(topo): preserve manifold seed orders on retrace
1640d66 feat(topo): validate manifold label refreshes
69ba3f3 feat(topo): track manifold sample correspondence
a5a1f2e feat(topo): add differentiable manifold branches
8ac3b79 fix(topo): make periodic-root sensitivities explicit
ceab1f5 fix(topo): separate eigenvector diagnostics from gradients
4c882f2 feat(topo): add implicit periodic-point solver
e29e765 test(flt): verify JAX and Cyna map parity
0003c84 feat(flt): add differentiable JAX Poincare map
```

### ESSOS

- Remote: `https://github.com/hongkelu/ESSOS.git`
- Remote base: `754e5953042c0dea0987956f2237d17cb9e4c7e8`
- Implementation head before this report: `7f544bd7203625b3bb9d4c9fa9ea518d5d540290`
- Committed diff: 19 files, 8,456 insertions

Commits, newest first:

```text
7f544bd test(manifold): validate QA heat transaction
39b044b feat(manifold): gate heat strike correspondence
5e9d5c6 feat(manifold): gate Stage-2 heat loading
bfd3086 feat(manifold): ramp heat load in stage 2
7b10f03 feat(manifold): add power-conserving heat surrogate
1a4182b test(manifold): preserve QA X-line clearance
19d80de feat(manifold): gate Stage-2 X-line clearance
8b0e150 feat(manifold): ramp X-line clearance in stage 2
104871c feat(manifold): add periodic X-line clearance loss
95801cd test(manifold): solve QA coil-shape target
d9fe71f test(manifold): validate physical QA strike
9136b68 test(manifold): optimize tokamak coil shape for strike
861a1e7 test(manifold): solve tokamak wall-strike target
325844c feat(manifold): integrate strike continuation gate
3bfd4ff feat(manifold): validate wall-strike refreshes
8d8337d feat(manifold): add differentiable wall-strike target
44e08e7 test(manifold): solve bounded tokamak target
f2ea66c feat(manifold): backtrack topology-rejected coil steps
20e86bd test(manifold): validate tokamak coil continuation
9fb79d4 feat(manifold): validate continuation candidates
e50c1d8 feat(manifold): bridge ESSOS fields to Cyna validation
a194202 feat(manifold): add Stage-2 continuation schedule
a2582f4 feat(manifold): add validated Stage-2 target state
4f0cb39 feat(manifold): add invariant-manifold target loss
4cb1f14 feat(manifold): add periodic X-line losses
f279395 feat: add return-map surface loss
08c139c feat: add PyNA manifold field adapter
```

## Bottom line

The project has progressed beyond a conceptual interface. It now contains a
complete differentiable-and-production transaction for periodic X-lines,
labelled invariant-manifold samples, exact first-wall strikes, X-line
clearance, and prescribed-power heat loading, exercised on both an
axisymmetric tokamak and a physical QA modular-coil configuration. The next
bounded step is to complete the same inner/outer contract for the clearance of
the manifold legs before their intended divertor strikes.
