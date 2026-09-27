# Implemented interfaces and their validity domains

## Problem, tracking snapshot, and evaluation

`essos.topology_optimizer.DesignProblem` fixes the physical parameter reference,
scales, bounds, equality tolerance, and versioned problem identity. Parameters
are optimized as `c = reference + scales * q`. `Evaluation` carries residuals,
physical-parameter Jacobians, inequalities (`g <= 0`), equalities, validity,
diagnostics, and a proposed tracking snapshot. `AcceptedState` owns immutable
parameters and JSON snapshots. Every callback gets an independent copy.

`optimize_topology(problem, c0, model, refresh, checkpoint=...)` solves a scaled
constrained Gauss–Newton subproblem, then requires refreshed feasibility and
positive physical objective decrease with acceptable predicted/actual agreement.
A successful topology check alone cannot accept a step. Checkpoints are written
atomically before commit. Restarts verify problem identity, scales, numerical
controls, and the independently refreshed objective. Callable definitions must
be versioned in the identity; hashing a Python function name cannot establish
that two physical problems are identical.

A minimal callback returns `Evaluation(residual, snapshot, jacobian=...)`.
`refresh` must reconstruct the physical candidate and return `valid=False` for
unusable physics; backend/programming exceptions propagate. Diagnostics may
contain nonfinite conditioning values, which are recorded as strings, while
accepted snapshots must remain finite, non-executable JSON. A projected-gradient
check distinguishes constrained stationarity from an unresolved model decrease.

## Open lines and manifolds

`pyna.topo.open_lines.LaunchBundle` declares fixed physical, moving physical,
flux-labelled, or manifold-generated launches and whether quadrature weights
move. Moving conventions require explicit differentiable launch/weight arrays;
a label does not construct a flux surface. `trace_open_bundle` computes local
wall events and connection lengths without a periodic-orbit dependency.

`essos.open_bundle_optimization.OpenBundleObjective` uses observables of
`(hit_RZPhi, connection_lengths, weights, parameters)`. Its production callback
must return `OpenBundleValidation`. `pyna.topo.open_validation.validate_open_bundle_3d`
requires launches strictly inside the production wall, retraces first hits and
complete trajectories, checks patch and unwrapped
phase identity, and calls the Cartesian triangle-wall clearance certificate.
Targets and scales remain fixed during refresh. The local model's value is
aligned with production observables; its Jacobian still requires independent
production finite-difference/refinement checks.

`essos.manifold_sample_objective.ManifoldSampleObjective` provides the equivalent
adapter for a labelled, production-refreshed manifold sample. Existing ESSOS
manifold, strike, heat and leg targets accept `derivative_mode`:

- `frozen_reference` preserves the historical fixed eigendirection/seed geometry.
- `moving_linear_seed` includes periodic-point and oriented eigenpair response.

Moving a straight linear seed does not differentiate an exact nonlinear
invariant manifold. Seed-size convergence tests separate representation error
from differentiation error. Discrete branch, side, parity and hit identities
remain outer-loop decisions. Near-singular periodic roots expose condition,
minimum singular value and response-amplification diagnostics; invalid JVPs/VJPs
are unusable rather than silently regularized.

## Native tracing correction

Cyna wall-aware tracing now interpolates corresponding polygon vertices between
adjacent toroidal sections, matching Python projection and the local event model.
It no longer creates a discontinuous wall by selecting the nearest section.
Native first-hit localization integrates the trajectory during bisection, and
connection length uses RK4 quadrature ending at the located hit. Plain native
RK4 rejects unusable/nonfinite toroidal fields instead of treating them as zero
motion. Results for a varying wall may therefore change from older revisions;
regenerate accepted tracking snapshots and qualification artifacts.

## Three-dimensional walls

`pyna.topo.clearance3d.TriangleWall` computes Euclidean point-to-triangle distances.
Signed distances require an oriented, watertight mesh and are positive inside.
`certify_path_clearance` uses the distance function's Lipschitz bound, certified
arc-length upper bounds, and explicit tracing/wall error bounds. Chord lengths
are not valid arc-length upper bounds. A failed certificate means clearance is
unresolved; it does not prove a collision. The reduced implementation scans
triangles and is not intended as a high-throughput BVH implementation.

A terminal exclusion is a physical length, authorized only by an independently
verified first hit on the declared target patch. Other wall patches remain
checked. Existing legacy section-distance objectives and fractional exclusions
retain their older semantics and must not be described as this 3-D certificate.

## Closed surfaces

`pyna.topo.torus_solver.InvariantCircleProblem` solves a gauged Fourier section
circle, its unwrapped map rotation, and a normal counterterm. The fixed label is
**signed geometric section area**, not magnetic flux. Acceptance checks the
collocation and dense off-grid residuals, counterterm, conditioning, angular
lift, regularity and sampled self-intersection. Implicit JVP/VJP and a corrected
system predictor use linear solves. A nonzero counterterm cannot be labelled an
exact invariant circle.

`ApproximateCircleProblem` instead declares a constrained least-squares
stationarity problem and differentiates its full KKT system, including residual
curvature. Its result exposes nonzero invariance residual separately.
`validate_circle_nesting` checks ordered labels, sampled containment, edge
intersections and separation; Fourier and dense-sampling refinement remain
necessary. `essos.surface_optimization.CircleObjective` connects exact corrected
circles to the shared optimizer without a manifold target.

Map rotation is an angle per declared physical toroidal span. Divide by that
span for the signed transform. The new `finite_turn_iota_response`,
`axisymmetric_iota_response`, and `fieldline_velocity_response` helpers distinguish
finite observation windows, long-time zero-mode response, and covariant versus
contravariant field components. Legacy disputed spectral helpers retain their
historical behavior with explicit limitations/deprecation warnings.

## Equilibrium response

`essos.equilibrium_response.ImplicitEquilibriumField` wraps an external primal
solver and a square, gauged residual `E(z,c)=0`. It computes total **Eulerian**
field response through primal and adjoint linear solves, rejecting nonconverged
or singular states. The caller declares boundary assumptions, profiles held
fixed, controls and spatial domain. `EquilibriumCircleObjective` composes this
response with a surface map depending on `(controls, equilibrium_unknowns)` and
requires both equilibrium and topology validity at every acceptance.

`essos.vmec_response.VmecJaxMaterialField` is an optional, distinct adapter to the
accessible VMEC-JAX checkout. Its Cartesian field is sampled at **moving VMEC
flux-grid positions**. It cannot be passed to a fixed-position field-line tracer
without coordinate inversion and response. Controls are fixed-boundary Fourier
coefficients, not free-boundary coil controls. Independent full-equilibrium
finite differences qualify each response; LI383 currently fails this gate.
The experimental residual-response path also fails its independent base-state
check and must not be used for an accepted physical design.

## Inputs, caching, and migration

Packaged inputs are available via `importlib.resources.files('essos')` under
`data/manifold_optimization`. The strict SIMSOPT JSON reader supports the supplied
XYZ-Fourier/rotation/current graph without executing serialized objects. Source
hashes and physical-association limitations are in `provenance.json`.

`CurrentGridBasis` represents fields per ampere on fixed geometry and sampling
coordinates. Changing geometry requires rebuilding it. `content_identity`
separately binds currents, geometry, wall, numerics, topology and targets.
`Coils` current normalization is now a finite Python scalar in PyTree metadata;
constructing coils from traced currents requires an explicit fixed
`currents_scale`. This prevents a JAX cache deadlock. Ordinary finite host
currents and all-zero currents retain usable defaults.


## VMEX exterior-field adapter

`essos.vmex_field.VmexExteriorField` consumes a VMEX `VmecExtender`, an explicit
`EquilibriumDefinition`, a strict exterior-domain predicate, and the caller's
primal acceptance. `sample` returns Cartesian B plus an eager quadrature report;
`to_pyna_grid` samples an exterior-only cylindrical grid for native Cyna tracing.
It rejects points outside the declared domain, excessive quadrature error,
nonfinite values, and unqualified near-surface continuation.

`vjp` uses the backend's named parameter response. `jvp` is a reduced reference
implementation requiring one adjoint per field component, useful for validation
rather than large-grid sensitivity assembly. Both check backend/control identity.
Neither method certifies a physical derivative merely because it returns finite
values: VMEX's fixed-boundary adjoint freezes inactive solver DOFs, and LI383
currently fails the independent ordinary-re-solve check. See the
[VMEX integration report](vmex_integration.md) for measured scope and semantics.


For a coupled coil/equilibrium control that appears in both of VMEX's parameter
blocks, pass `control_jacobian=np.vstack((np.eye(n), np.eye(n)))`. The adapter
sums the plasma-mediated and direct-coil pullbacks and maps JVP directions into
both blocks. This map is an owned read-only array; the factory must construct
both backend parameter blocks from the same physical control at each candidate.


### Fourier LCFS domain check

`essos.vmex_domain.FourierLCFSExteriorDomain.from_wout(wout)` constructs a
host-side exterior predicate for `VmexExteriorField`. `domain(points)` returns
one Boolean per Cartesian point. `domain.classify(points)` additionally returns
a conservative signed clearance from the polygon in each constant-phi RZ
section, after subtracting its Fourier chord-error bound. This is not a 3-D
clearance certificate. Simple, closed toroidal sections and integer VMEC wout
mode conventions are required. `minimum_section_clearance` can exclude a band
near the boundary, but a field quadrature budget must still be checked.
