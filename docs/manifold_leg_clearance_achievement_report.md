# Pre-strike manifold-leg clearance milestone

Date: 2026-09-10 (Asia/Shanghai)

The differentiable pre-strike clearance milestone is implemented across ESSOS
and PyNA, including independent production validation, continuation, rejection,
rollback, and physical tokamak/QA coverage. This advances the first five items
of the earlier manifold optimization achievement report. Multi-DOF engineering
optimization and transport-derived power remain separate subsequent work.

## Implementation

- ESSOS exposes `ManifoldLegStage2Target`, independent of heat-power data. It
  freezes ordered strike labels, wall planes, a static wall callback, clearance
  requirements, integration settings, exclusion fraction, and optional weights.
- The inner loss differentiates full labelled seed-to-event trajectories with
  fixed retained sample indices. Exclusion is a fraction of toroidal-angle
  span, not arc length. The standalone heat-target adapter remains compatible
  with the preceding uncommitted implementation.
- PyNA/Cyna globally retraces every first-wall hit, checks label/displacement
  and unwrapped-phase trust, independently samples each seed-to-hit leg, checks
  trajectory completeness and endpoint agreement, and measures per-label
  minimum clearance. Missing strikes and any failing required leg reject the
  candidate, including legs with zero objective weight.
- ESSOS adds `leg_clearance_weight`, `leg_target_state`, and
  `ManifoldLegValidationConfig`. The outer transaction requires production
  clearance, refreshed JAX/Cyna strike correspondence, and the inner clearance
  margin before accepting a new target snapshot.
- Rejection uses the existing backtracking transaction. Tests verify unchanged
  DOFs/state after all proposals fail and require explicit accepted refreshes
  before continuation advances.

API usage and acceptance semantics are documented in
[manifold_stage2.md](manifold_stage2.md#pre-strike-leg-clearance).

## Geometry correction discovered by QA

The initial QA run exposed a false clearance sign near a notch vertex. The
old projection-normal dot product could be near zero while the actual section
clearance was approximately 31 mm. Floating-point sign changes then classified
interior points as exterior. A focused concave-wall example reproduced false
negatives at 15 of 32 interior points.

Native wall projections now determine signed clearance using polygon
containment in the continuously interpolated periodic section. The existing
X-line gate consumes the same corrected signed distance. Tests cover both
polygon orientations, concave vertices, outside points, and unwrapped phases.
The QA wall, coil target, and 0.5 mm leg-clearance requirement were preserved.

## Physical checkpoints

The tokamak current and coil-height target-recovery regressions now require
both inner and production clearance. A separate physical stable-leg test
checks backward phase order, accepted clearance, halved production spacing,
and rejection of an excessive margin. The required tokamak margin is 1 mm.

The QA period-five regression retains its symmetry-preserving 10 micrometre
coil Fourier perturbation and all preceding strike, X-line, and heat gates.
The accepted coil-shape step also passes:

- production pre-strike minimum clearance: **30.9121418 mm**;
- inner radial-proxy minimum: **213.658697 mm**;
- required inner/production margin: **0.5 mm**;
- terminal exclusion: **20% of toroidal-angle span**;
- nominal production minimum change under halved integration/output spacing:
  **1.10e-9 m**; and
- explicit rejection of a **40 mm** production clearance requirement.

The inner radial wall proxy and production closest-section-segment distance
are different measures; their reported minima are not a correspondence test.
Both must independently satisfy the configured margin. The QA regression
passed in 234.24 seconds on the available CPU backend.

## Verification

The final focused PyNA batch passed **51 tests in 86.39 seconds**:

```text
tests/test_manifold_leg_clearance.py
tests/test_xline_clearance.py
tests/test_manifold_heat.py
tests/test_jax_strike.py
tests/test_manifold_strike_correspondence.py
tests/test_strike_heat_control.py
```

The final ESSOS integration batch passed **47 tests in 114.02 seconds**:

```text
tests/test_manifold_adapter.py
tests/test_manifold_heat_optimization.py
tests/test_manifold_leg_optimization.py
tests/test_manifold_optimization.py
tests/test_manifold_validation.py
tests/test_manifold_strike_validation.py
```

The final physical runs passed **3 tokamak tests in 138.34 seconds** and
**1 QA test in 234.24 seconds**, for **102 passing targeted tests** across both
repositories. Physical coverage runs:

```text
tests/test_manifold_coils.py -k 'first_wall_strike or stable_leg'
tests/test_qa_manifold_coils.py
```

The development environments still split optional dependencies. Runs use:

```bash
env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/essos-manifold-review-mpl \
  PYTHONPATH=/home/lhk/uwplasma/ESSOS-manifold:/home/lhk/uwplasma/pyna-manifold:/home/lhk/.conda/envs/pyna-env/lib/python3.10/site-packages \
  /home/lhk/.conda/envs/uwplasma-env/bin/python -m pytest -p no:cacheprovider ...
```

These are targeted checks, not a claim that both repositories' entire test
suites have been rerun. PyNA's implementation is recorded in local commit
`4f032d7` on `manifold-optimization`.

## Remaining scope and interpretation

The certificate samples trajectories and measures nearest wall-segment
distance in each poloidal section at the query angle. It is not a continuous
trajectory proof or a global nearest-surface distance in three dimensions.
The new spacing checks do not replace full field-grid, wall-mesh, and JAX
trajectory convergence studies.

The physical cases remain bounded target-recovery regressions. Their proposed
coil steps are driven by strike placement; clearance is now an independently
enforced acceptance constraint. Joint multi-DOF optimization of strike/heat/
clearance and engineering objectives remains the next optimization milestone.
The QA wall remains regression geometry and its 1 W heat allocation remains
prescribed data. A transport-derived absolute power model, realistic step
sizes, and broader convergence studies are still required for physical design
interpretation.
