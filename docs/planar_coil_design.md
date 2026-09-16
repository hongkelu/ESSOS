# Fork coil design and planar-coil workflows

For coupled equilibrium or coil-design workflows, keep the geometry and
current parameterization in ESSOS and pass the resulting field to the external
solver. The design vector is ordered as fractional independent-coil current
changes followed by additive Fourier shape coordinates. Run the shared,
artifact-free walkthrough for every public class and function in this API:

```sh
python examples/differentiate_coil_design.py
```

The central construction is:

```python
import jax.numpy as jnp

from essos.coil_design import make_coil_design_field_builder
from essos.coils import Coils_from_json

coils = Coils_from_json(
    "examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"
)
field_from_parameters = make_coil_design_field_builder(
    coils,
    current_groups=(2, 3),
    shape_dofs=((2, 2, 1),),
)
field = field_from_parameters(jnp.zeros((3,)))
br, bphi, bz = field.b_cyl(0.90, 0.05, 0.02)
```

The public coil-design entry points are:

- `FilamentaryBiotSavart`, the differentiable filament-field pytree;
- `CoilDesignFieldBuilder`, the reusable current-and-shape parameter map;
- `PlanarCoilDesignFieldBuilder`, the exact reduced planar parameter map;
- `ShapeDeformationMetrics` and `shape_deformation_metrics`, the corresponding
  local shape diagnostic and its constructor;
- `make_coil_design_field_builder`, for joint current-and-shape charts;
- `make_fractional_current_field_builder`, for current-only charts;
- `make_planar_coil_design_field_builder`, for planar current, local-shape,
  center, and orientation charts; and
- `make_shape_field_builder`, for shape-only charts.

`optimize_planar_residual` is the corresponding JAX-Jacobian/SciPy bridge for
least-squares optimization on a `PlanarCoilDesignFieldBuilder`. It returns both
native `PlanarCoils` and SciPy's complete convergence result.

Each `current_groups` entry is an independent base-coil index and controls its
complete symmetry-expanded physical coil group. A `shape_dofs` entry is
`(base_coil, xyz_component, Fourier_index)` and its additive parameter is in
metres. For a general local chart, `shape_directions` accepts a stack in which
each shape parameter can combine multiple Fourier coefficients across multiple
base coils. The shared walkthrough includes a two-current, two-shape chart,
prints its current-first parameter ordering, evaluates a finite field JVP, and
reports the rebuilt physical-coil length range as a descriptive engineering
diagnostic. Builder construction and `shape_deformation_metrics` perform
host-side validation/reporting; the returned builder and field arrays remain
JAX traceable.

### Planar coils

Here, "planar" means a one-dimensional filament centerline constrained to a
plane embedded in three-dimensional space. It does not imply a finite-width
conductor or a two-dimensional current sheet.

`PlanarXYCurves` stores Cartesian centers, scalar-first local-to-global
quaternions `[w, x, y, z]`, and non-DC local X/Y Fourier coefficients. The
coefficient order is `[sin(1), cos(1), sin(2), cos(2), ...]`; translation is
represented only by the centers. Attach physical currents with `PlanarCoils`:

```python
import jax.numpy as jnp

from essos.coil_design import make_planar_coil_design_field_builder
from essos.planar_coils import PlanarCoils, PlanarXYCurves

curves = PlanarXYCurves(
    centers=jnp.asarray([[1.4, 0.0, 0.0]]),
    quaternions=jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
    xy_dofs=jnp.asarray([[[0.0, 0.28], [0.22, 0.0]]]),
    n_segments=32,
    stellsym=False,
)
planar_coils = PlanarCoils(curves, jnp.asarray([1.1e5]))
builder = make_planar_coil_design_field_builder(
    planar_coils,
    current_groups=(0,),
    shape_dofs=((0, 0, 1),),
    center_dofs=((0, 0),),
    orientation_dofs=((0, 1),),
)
field = builder(jnp.zeros(builder.parameter_shape))
```

The reduced design vector is ordered as fractional current changes, local XY
Fourier changes in metres, Cartesian center changes in metres, and global
rotation-vector increments in radians. Planar shape coordinates use the same
zero-based, non-DC XY ordering shown above. Finite rotations use an exact
exponential map, so optimization steps cannot leave the moving plane.

`PlanarXYCurves.from_polar_radius` converts the radial Fourier form used by
SIMSOPT-style planar curves into the native XY representation, increasing the
XY order by one. `PlanarXYCurves.from_simsopt_planar` reads SIMSOPT's complete
`local_full_x` coordinates, including fixed degrees of freedom. Imports default
to no symmetry because a base SIMSOPT curve does not encode `nfp` or
stellarator symmetry; noncanonical SIMSOPT quadrature requires an explicit
ESSOS `n_segments`. Exact independent-curve export is available through
`to_simsopt_xyz`, leaving symmetry expansion to the caller. Conversion from a
general XY curve back to a radial planar curve would require an explicit fit
and is not done silently. Coil import preserves scalar current values but not
SIMSOPT's shared or scaled current dependency graph.

A trusted SIMSOPT JSON file can initialize either planar or ordinary XYZ
ESSOS coils through one lazy optional-dependency entry point:

```python
from essos.io import load_simsopt_coils_json

coils = load_simsopt_coils_json(
    "simsopt_biot_savart.json",
    nfp=2,
    stellsym=True,
)
```

`nfp` and `stellsym` are required because a general saved coil list does not
carry authoritative symmetry metadata. The default expects the common
symmetry-expanded `BiotSavart.coils` layout and validates its wrapper
transforms and current signs before selecting the base coils. Set
`source_is_expanded=False` for JSON containing independent base coils. The
loader returns `PlanarCoils` only when every base curve is planar; exact XYZ
Fourier input returns ordinary `Coils`. SIMSOPT is imported only when this
loader is called, and its decoder should be used only with trusted JSON files.

`PlanarCoils.to_json` writes a versioned `planar_xy_fourier` representation.
The canonical `essos.io.load_coils_json` entry point loads it while continuing
to accept existing untagged XYZ coil JSON. `Coils_from_json` remains available
for legacy XYZ-only workflows. Run the artifact-free differentiation example
with:

```sh
python examples/differentiate_planar_coils.py
```

Run a complete normal-field least-squares optimization, including native
planar reconstruction and a falling-objective check, with:

```sh
python examples/optimize_planar_coils_bdotn.py
```

The optimization pattern is deliberately explicit:

```python
from essos.optimization import optimize_planar_residual

def residual(parameters):
    field = builder(parameters)
    return field.B(probe_points) @ probe_normal - target_BdotN

optimized_coils, result = optimize_planar_residual(residual, builder)
```

Use `builder.rebuild_coils(parameters)` when an older API requires ordinary
XYZ `Coils`, and `builder.rebuild_planar_coils(parameters)` when the optimized
plane coordinates and planar JSON representation must be preserved.

`FilamentaryBiotSavart` stores filament points, tangents, and physical currents
as JAX pytree leaves. This keeps current and shape derivatives visible across
JIT, JVP/VJP, and external implicit-differentiation boundaries. Its filament
singularity is numerically regularized only to keep transformed evaluations
finite; points on a filament do not represent a physical finite field.
It also implements the Cartesian field methods used by ESSOS field-line,
guiding-center, full-orbit, and field-derivative consumers, including `AbsB`,
spatial derivatives, curls, curvature, and `to_xyz`.

Coil JSON written by current ESSOS versions stores physical `base_currents`
alongside normalized `dofs_currents` and `currents_scale`, and the loader rejects
inconsistent redundant values. For legacy files without a scale,
`dofs_currents` is interpreted as physical current; a normalized-only legacy
file cannot recover a scale that was never stored.

