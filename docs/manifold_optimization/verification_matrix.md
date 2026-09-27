# Verification coverage

The required inventory is `benchmarks/manifold_optimization/baseline.json`.
A row identifies evidence, not a claim beyond that evidence's model or resolution.
Native lanes reject missing backends and skipped tests. Artifact manifests record
whether a run actually passed. The final inventory has 262 tests.

| ID | Evidence | Scope |
| --- | --- | --- |
| V01 | Existing ESSOS adapter/native map tests | Cartesian/cylindrical components and physical radii. |
| V02 | JAX Poincaré, periodic and native wall tests | Signed span, field period and unwrapped phase. |
| V03 | `pyna/tests/test_manifold_physics_conventions.py` | Solenoidal manufactured field; flux-weighted preservation converges. |
| V04 | JAX map validity and native wall interpolation tests | Unusable toroidal field rejects instead of producing zero motion. |
| V05 | `pyna/tests/test_jax_periodic.py` | Implicit periodic-root response. |
| V06 | Same module | Small residual with nearly singular root is not derivative-valid. |
| V07 | Periodic/manifold derivative tests | Total monodromy and root motion. |
| V08 | `pyna/tests/test_manifold_contracts.py` | Oriented rotating eigenpair JVP/VJP; moving versus frozen seed. |
| V09 | Manifold correspondence and native refresh tests | Side, stability, generation, parity and labels. |
| V10 | Manifold contract test | Quadratic linear-seed representation error under seed refinement. |
| V11 | JAX strike, physics conventions, native wall interpolation tests | Plane/curved events, moving wall and continuous non-axisymmetric native wall. |
| V12 | Strike/correspondence/native open validation | Grazing, missing hits, label and unwrapped-phase failures. |
| V13 | `pyna/tests/test_clearance3d.py` | Cartesian distance differs from section projection. |
| V14 | Same module and native whole-leg validation | Conservative interval bounds catch unresolved between-sample clearance. |
| V15 | Same modules | First-hit authorization and checks against non-target patches. |
| V16 | Manifold contracts and torus-deformation tests | Finite-window, long-time zero-mode and metric conversion semantics. |
| V17 | Torus solver and surface optimizer tests | Phase gauge and fixed signed section-area label; general magnetic-flux labels remain separate work. |
| V18 | `pyna/tests/test_torus_solver.py` | Corrected-system predictor residual order. |
| V19 | Torus/nesting tests and NCSX refined/native runs | Off-grid, Fourier, tracing, grid and sampled nesting checks. |
| V20 | Torus solver tests and NCSX directional differences | Corrected-solution JVP/VJP. |
| V21 | Approximate-circle KKT test | Full stationarity response with a nonzero invariance residual. |
| V22 | Open-line, open-objective and native open-optimization tests | Moving launches/walls and connection lengths without manifold targets. |
| V23 | Snapshot/optimizer tests | Nested mutation isolation and serialization. |
| V24 | Optimizer tests | Physically worse candidates reject despite valid topology. |
| V25 | Same tests | Local predicted improvement cannot replace refreshed merit. |
| V26 | Optimizer tests | Atomic checkpoint, rollback, unchanged semantics on restart. |
| V27 | Candidate-cache tests and problem fingerprints | Current, geometry, wall, numerics, topology and target identity. |
| V28 | Required lanes and installed-wheel script | Actual Cyna, isolated imports, packaged resources and surface response. |
| V29 | `ESSOS/tests/test_candidate_cache.py` | Cached current combination matches fresh Biot–Savart fields/Jacobians. |
| V30 | Equilibrium and coupled-surface tests; VMEC qualification scripts | Manufactured total response/joint acceptance pass; tokamak material response passes; LI383 field response remains unqualified. |

Native wall interpolation, native arc-length quadrature to the located hit, and
unusable-field rejection are production behavior changes. Historical nearest
wall slices could create first hits at toroidal cell midpoints. The new native
regression checks a non-axisymmetric wall at both ordinary and multi-turn phases
against independent analytic first hits and connection lengths.
