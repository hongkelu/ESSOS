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
two initial smooth objective terms:

- `periodic_xline_location_loss` places the tracked section point relative to a
  Stage-1 target with independent R and Z scales.
- `periodic_xline_hyperbolicity_loss` keeps the monodromy trace in a requested
  direct- or inverse-hyperbolic margin window.  The branch sign is fixed by the
  outer topology tracker, preventing an inner optimization from silently
  switching X-line identity.

The inner loss never decides whether Newton found the intended orbit.  Before
accepting an optimizer step, the outer loop must inspect the residual and
`converged` flag, update the initial guess, and compare against the production
PyNA/Cyna map.

## Parallel JAX and Cyna manifold paths

PyNA now exposes a differentiable JAX manifold tracer in parallel with its
production Cyna tracer.  Parallel here means two implementations of the same
labelled return-map problem, not that they must execute concurrently.  JAX can
itself run on CUDA when a CUDA-enabled JAX runtime is installed; Cyna retains
its CPU/CUDA production role for high-throughput and wall-aware tracing.

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

This sample-location loss is a smooth lobe/divertor-leg placement proxy.  It is
not yet an exact strike-point objective: wall intersection, branch relabelling,
connection length, and heat-load evaluation remain PyNA/Cyna outer-loop
milestones.  Accepted ESSOS steps must be re-traced there before their topology
is trusted.
