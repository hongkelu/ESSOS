# Measured implementation results — 2026-09-21

These are reduced numerical benchmarks run on a Perlmutter interactive compute
node using CPU JAX with float64. They are not reactor engineering designs or a
completed finite-beta release. Raw logs and full audits remain under the
workspace's `run_artifacts/`; curated reports and source manifests are in
[`results/20260921`](../../benchmarks/manifold_optimization/results/20260921).

## NCSX closed-surface control

Three modular-coil current factors obey bounds `[0.95, 1.05]` and sum to three.
The fixed signed section area is `0.0028274333882308137 m²`; the target transform
is `0.40181466005886707`, an increase of 0.005 from the qualified initial circle.
The corrector uses 15 Fourier nodes, 64 map steps per 120-degree field period and
128 coil quadrature points. Independent validation doubles map/coil resolution
and uses 29 Fourier nodes. Neighboring circles at 90% and 110% of the target area
also pass dense nesting checks, with about 0.505 mm minimum section separation.

| Method | Final objective | Refined transform error | Optimization time | Total time |
| --- | ---: | ---: | ---: | ---: |
| [Implicit response](../../benchmarks/manifold_optimization/results/20260921/ncsx_surface_implicit/report.json) | 1.48e-14 | 3.47e-8 | 42.8 s | 265.9 s |
| [Re-solved finite differences](../../benchmarks/manifold_optimization/results/20260921/ncsx_surface_fd/report.json) | 1.48e-14 | 3.47e-8 | 110.5 s | 328.7 s |
| [Direct constrained surface solve](../../benchmarks/manifold_optimization/results/20260921/ncsx_surface_direct/report.json) | 6.71e-26 | 3.35e-8 | 11.4 s | 223.2 s |

The implicit and finite-difference methods accept three steps and use respectively
7 and 25 corrected-surface solves during optimization. The direct comparator
solves for embedding and controls together, then independently corrects/validates
its result; its optimization uses 8 nonlinear constraint evaluations and one
independent corrector. These counts are different kinds of work and must not be
compared as equivalent function calls.

Optimization timing starts after common derivative-qualification solves have
compiled shared kernels. Total timing includes setup, qualification, optimization
and refinement, with the compilation policy recorded in each report. Runs shared
the compute node; these single-run timings are diagnostic, not a controlled
scaling study. In this reduced case the direct comparator is faster. No general
superiority claim is supported.

The refined implicit circle has maximum direct-coil invariance residual
`3.21e-11 m`. A separate [native Cyna grid-refinement check](../../benchmarks/manifold_optimization/results/20260921/ncsx_native/report.json)
reduces the mismatch from about 5.18 mm on a `31×31×32` field grid to about
0.0759 mm on `121×121×256`. Doubling 512 to 1024 trace steps changes mapped points
by at most `4.38e-7 m`. This distinguishes field-grid error from tracing error.

## NCSX open-line control

The supplied vessel covers one 120-degree field period. Symmetry expansion and
seam joining produce an oriented closed mesh with 18,000 vertices and 36,000
triangles. A fixed launch at `(R,Z)=(1.7,0.65) m` reaches a fixed vessel patch.
The numerical target is independently traced from current factors
`[1.001,0.999,1]`. Optimization uses three factors bounded by `[0.995,1.005]`
and a current-sum equality.

| Method | Initial objective | Final objective | Refined numerical target mismatch |
| --- | ---: | ---: | ---: |
| [Implicit local response](../../benchmarks/manifold_optimization/results/20260921/ncsx_open_implicit/report.json) | 2.66e-3 | 3.19e-11 | 7.59e-9 m |
| [Native finite differences](../../benchmarks/manifold_optimization/results/20260921/ncsx_open_fd/report.json) | 2.66e-3 | 1.43e-11 | 5.77e-9 m |

Both accept two steps. Local directional response agrees with production finite
differences to approximately `6.6e-4` relative error across three amplitudes.
Final validation doubles coil quadrature and field-grid resolution and halves
the native tracing step. Every accepted candidate passes native first-hit and
whole-leg checks. A [conservative recheck of the saved finite-difference candidates](../../benchmarks/manifold_optimization/results/20260921/ncsx_open_fd/final-gates.json)
confirms that they also satisfy the final Cartesian-distance, unwrapped-phase
and strictly interior-launch guards. The specified required clearance is 1 mm, the physical target
exclusion is 0.2 m, and the calculated triangle/interpolated-wall error bound is
3.493 mm. Certificates remain conditional on the stated 10 μm tracing error
budget; a separate physical error-budget qualification remains necessary.

The nanometre-sized target mismatch measures recovery of a known numerical
control setting. It is **not** absolute physical wall-position accuracy: wall
representation uncertainty is much larger. Independent divertor targets,
engineering limits, larger controls and held-out launches remain future
qualification work. Total times (about 231 s and 266 s) include common derivative
checks and refinement; they are not isolated optimization speed measurements.

This case exposed and fixed production issues: nearest-toroidal-section wall
selection created false hits at section midpoints; connection-length quadrature
included an overshooting step; and unusable toroidal field could freeze motion.
Continuous wall interpolation, integrated event localization/arc length, and
explicit unusable-field rejection now have independent analytic regressions.

## Tokamak prescribed-background control

The [two-control tokamak case](../../benchmarks/manifold_optimization/results/20260921/tokamak_control/report.json)
changes PF current and vertical position while holding the prescribed plasma
current-loop background fixed. Three accepted steps reduce the labelled-sample
objective from `0.0259` to `5.87e-13`, recovering current factor
`1.000300006894` and height `-1.199000006786 m`. Cyna refresh is mandatory at
acceptance. This is a prescribed-background portability/recovery test, not a
coil-only tokamak equilibrium or self-consistent plasma response.

## Finite-beta response qualification

The accessible VMEC-JAX solver revision is
`807255b2d29c2c733234004024faa6bcad9df8c1`. The adapter controls the fixed-boundary
`RBC(0,1)` coefficient while holding input profiles/current/flux settings fixed.
It returns Cartesian field at moving VMEC flux-grid positions, not at fixed
Eulerian positions. Both reduced primal starting equilibria converge.

The [tokamak material-grid response](../../benchmarks/manifold_optimization/results/20260921/tokamak_vmec_response/report.json)
passes the independent complete-equilibrium finite-difference gate: relative
vector errors range from approximately `1.7e-5` to `3.2e-5` at force tolerance
`1e-13`.

The [LI383 vector response](../../benchmarks/manifold_optimization/results/20260921/li383_vmec_response/report.json)
**fails** the 0.1% qualification threshold. Tightening force tolerance to `1e-18`
does not remove relative errors of 1.05%–4.87% across the tested amplitudes.
The scalar mean-field-strength response agrees much better (best relative error
`1.3e-5`), which is useful diagnostic evidence but does not qualify the vector
response. Stored finite-difference arrays support investigation of coordinate,
gauge and response consistency. The experimental implicit-residual path also
fails its independent base-state check. No LI383 equilibrium-aware design is
accepted on these responses.

The generic equilibrium adapter and joint equilibrium/surface optimizer pass
manufactured primal, JVP/VJP and combined-acceptance tests. Eulerian VMEC field
response, appropriate limiting cases and a physical finite-beta design remain
open release gates.

## Packaging and required tests

The current required inventory contains 262 tests, with no allowed skips or
expected failures. All 262 passed: 206 analytic, 50 production, 1 QA-coil and 5 tokamak tests.
Final lane manifests live in `run_artifacts/verified-20260921/` and the
[curated regression summary](../../benchmarks/manifold_optimization/results/20260921/regression-summary.json).
Three targeted rechecks cover the final Cartesian hit-distance and interior-launch
guards; these repeat existing tests and are not extra unique test counts. The installed-wheel report verifies
package paths outside both repositories, bundled tokamak/LI383/NCSX resources,
actual native tracing and an invariant-circle derivative. The four native
wall/event and whole-leg validation regressions also pass against the freshly built installation.

[`installed_wheels.json`](../../benchmarks/manifold_optimization/results/20260921/installed_wheels.json)
and [`wheel_sources.json`](../../benchmarks/manifold_optimization/results/20260921/wheel_sources.json)
record import paths, native/wheel checksums and source content. The source pair
remains on `manifold-optimization` with uncommitted changes; publishing requires
paired commits and an updated compatibility pin. No remote CI or published
release is claimed.


## Subsequent VMEX integration

The [VMEX report](vmex_integration.md) records the selected exterior-field backend,
new adapter/native-tracing tests, fixed-boundary response diagnostics, and the
coil-driven LI383 free-boundary pilot. Its isolated JAX 0.9.2/SciPy 1.16.3
environment passes the original 262-test inventory plus four new VMEX tests.
Those results are separate from the historical VMEC-JAX measurements above.


## VMEX finite-beta exterior endpoint recovery

Following the user's acceptance of the measured 0.17–0.23% common-current
field-response discrepancy, the next integration stage now has a successful
[re-solved endpoint-control benchmark](../../benchmarks/manifold_optimization/results/vmex-integration/li383_trace_optimization/report.json).
Two accepted steps recover current factor `1.0003000555` for a target of
`1.0003`, reducing the normalized objective from `1.75e-3` to `5.96e-11`.
Every candidate includes a fresh edge-accepted free-boundary LI383 equilibrium,
rebuilt exterior field, and native Cyna trace. This baseline uses finite-
difference gradients; it does not yet qualify implicit endpoint gradients.

Final/target traces agree to `1.09e-10 m` after grid/step refinement, which is
known-control numerical recovery precision. The independently measured absolute
native trajectory discrepancy is about `0.011 mm`. Sampled exterior-field
refinement passes at 5, 10 and 20 cm normal offsets from the LCFS; closer
points need additional quadrature or a qualified surface treatment. See the
[VMEX integration report](vmex_integration.md) for domains, limits and evidence.


## Connection-length benchmarks and tokamak SOL connection-length control

`essos.dynamics.connection_length` is checked against closed-form solutions in
`tests/test_connection_length.py`: an exactly solvable circular tokamak
(`essos.fields.CircularTokamakField`) with a divertor plate gives lengths to
1e-8, strike points to 1e-7 m, the pi q R limit, and B_poloidal, plate-depth
and seed derivatives in reverse and forward mode (`adjoint=diffrax.ForwardMode()`).
Near a hyperbolic X-line the error grows like tolerance / u0, the distance to
the X-line, and the test states that bound. Native Cyna wall hits converge to
the same closed form under grid refinement. Against these references
`connection_length` under `jit` is 30-400x faster than the former pyna
fixed-step open-line tracer at equal accuracy
([comparison](../../benchmarks/manifold_optimization/results/connection-length-20260926/open_lines_vs_connection_length.json)),
which was removed.

[`examples/manifold_optimization/optimize_tokamak_connection_length.py`](../../examples/manifold_optimization/optimize_tokamak_connection_length.py)
raises the mean connection length of 64 field lines across the outboard
scrape-off layer of the regression tokamak (vacuum coils, frozen plasma
current loop, smooth superellipse vessel) with PF current and height, a
divertor coil and a TF factor limited to +-3 %, while holding the X-point
([report](../../benchmarks/manifold_optimization/results/tokamak-connection-length-20260927/report.json)).
Least squares with forward-mode Jacobians of `connection_length` and of the
X-point (`essos.topology.periodic_xline_state`) converges in 15 evaluations
(28 s) to a stationary compromise:

| | Initial | Optimized |
| --- | ---: | ---: |
| Mean SOL Lc, ESSOS | 5.036 m | 5.440 m (+8.0 %) |
| Mean SOL Lc, Cyna grid | 5.073 m | 5.489 m (+8.2 %) |
| X-point (R, Z) | (1.4702, -0.0901) m | (1.4732, -0.0964) m |

The +30 % target is not reached: the result is the weighted balance with the
X-point constraint (7 mm displacement against a 5 mm scale) and the control
regularisation. Cyna agrees with ESSOS to a median 0.27 % per seed; 5 of 64
seeds differ by more than 1 %, all at jumps of Lc where a line changes the
number of turns before striking. Lc is piecewise smooth in the controls and
automatic differentiation sees only the smooth part; a box wall made these
jumps frequent enough to stall the optimizer (first-order optimality 91 vs 5
with the smooth wall). This is a regression geometry, not a device design.


## Tokamak Super-X by divertor-coil currents

[`examples/manifold_optimization/optimize_tokamak_superx.py`](../../examples/manifold_optimization/optimize_tokamak_superx.py)
moves the outer strike point of a lower-single-null tokamak outward along the
divertor floor at fixed plasma shape
([report](../../benchmarks/manifold_optimization/results/tokamak-superx-20260927/report.json)).
The field is an axisymmetric 1/R toroidal field (no TF ripple), a frozen plasma
current on 13 filaments and 13 poloidal-field coils (4 shaping, 9 divertor). The
poloidal flux psi = R A_phi comes from the same discretised loops as the traced
field. In vacuum psi is linear in the coil currents, so the isoflux conditions
(grad psi = 0 at the X-point, psi = psi_X at four LCFS points) are exact linear
constraints; the optimizer moves only in their null space. It drives the strike
radius R_t (outer root of psi(R, Z_floor) = psi_X) towards 2.2 m, asks for more
near-SOL connection length and keeps the poloidal flux expansion from falling.

| | Base | Super-X |
| --- | ---: | ---: |
| Outer strike radius R_t (psi) | 1.795 m | 2.164 m |
| Strike of a traced line 0.1 mm outside the LCFS | 1.795 m | 2.162 m |
| Total flux expansion R_t / R_u | 0.875 | 1.056 |
| Poloidal flux expansion | 1.136 | 1.133 |
| Near-SOL mean Lc, ESSOS / Cyna grid | 25.33 / 25.36 m | 27.75 / 27.76 m |

The X-point and LCFS conditions hold to 1.4e-6 Wb, the largest coil current is
0.93 MA, and Cyna agrees with ESSOS per seed to 0.11 %. With only the first 8
coils (2 null-space directions) the strike still moved to 2.02 m, but the
poloidal flux expansion fell from 1.39 to 0.56 and Lc did not grow: extra
divertor coils are what keep the poloidal field low along the extended leg.
The plasma current is frozen (no free-boundary response) and coil forces and
vertical stability are not constrained; this is stage 1 before coupling to a
Grad-Shafranov solver. The gradient-free STEP study of Nunn et al., Phys.
Plasmas 32, 072507 (2025), optimises a related connection-length objective.

### Connection length as the primary objective: X-point target

A longer leg alone raises Lc only modestly (Super-X above: +9.5 %), because Lc
grows with B/B_p along the path. With Lc itself as the objective
(`DESIGN=long_lc`, mean log Lc over the band, strike kept on the floor) the 7
free directions gave +17 % and grew an uncontrolled null near the outer wall
([report](../../benchmarks/manifold_optimization/results/tokamak-xpoint-target-20260927/long_lc_report.json)).
`DESIGN=xpt` adds an X-point target: grad psi = 0 at (1.85, -1.30) m on the
outer leg and psi there on the flux surface 3 cm outside the LCFS at the
midplane. Both are linear in the currents and join the exact null-space
constraints (9 conditions, 4 free directions); a weak saddle (det Hessian psi
towards -0.3) widens the low-B_p region, and currents are kept near 2.5 MA
([report](../../benchmarks/manifold_optimization/results/tokamak-xpoint-target-20260927/report.json)).

| | Base | X-point target |
| --- | ---: | ---: |
| Near-SOL mean Lc, ESSOS / Cyna grid | 25.33 / 25.36 m | 31.81 / 31.83 m (+26 %) |
| Lc at 2.5 cm / 3.0 cm outside the LCFS | 24.4 / 22.4 m | 44.1 / 53.8 m |
| Poloidal flux expansion | 1.14 | 1.60 |
| Secondary null det Hessian psi | - | -1.36 |

The secondary-null conditions hold to 1e-15 and Cyna agrees per seed to 1.5 %.
A strong saddle placed 1 cm out (det -9.4) lengthened only the lines inside
its flux label (48 -> 67 m) for a +6 % mean: the gain is logarithmic near a
null, so a wide low-field region matters more than the null itself. The weak-
saddle target was not reached (-1.36 vs -0.3), the largest current is 2.54 MA
(soft limit) and the primary separatrix becomes convoluted near the divertor.
Coil positions, a snowflake-like second-order null and wall-conformity
constraints are the next levers.
