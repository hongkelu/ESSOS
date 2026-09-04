"""ESSOS adapters for PyNA's differentiable topology primitives.

ESSOS owns the Stage-2 design variables and magnetic-field models.  PyNA owns
field-line maps and topology.  The functions here provide the one-way bridge
from ESSOS to the optional JAX backend in PyNA without introducing an ESSOS
dependency in PyNA.

PyNA is imported lazily so ordinary ESSOS installations do not require it.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from essos.fields import BiotSavart


def essos_field_callable(xyz: Any, field: Any) -> Any:
    """Evaluate an ESSOS magnetic-field object for PyNA.

    ``field`` is passed as the active JAX PyTree, so derivatives of a PyNA
    return map propagate through any dynamic leaves registered by the ESSOS
    field implementation.
    """

    return field.B(xyz)


@jax.jit
def _batched_essos_field_callable(xyz_batch: Any, field: Any) -> Any:
    return jax.vmap(lambda xyz: essos_field_callable(xyz, field))(xyz_batch)


def _pyna_cylindrical_field_class():
    try:
        from pyna.fields import VectorFieldCylind
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS production manifold validation requires PyNA"
        ) from exc
    return VectorFieldCylind


def _sampling_axis(
    values: Any,
    name: str,
    *,
    minimum_size: int,
) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < minimum_size:
        raise ValueError(
            f"{name} must be a one-dimensional array with at least "
            f"{minimum_size} point(s)"
        )
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain only finite values")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return np.ascontiguousarray(axis)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def essos_field_to_pyna_cylindrical_grid(
    field: Any,
    R: Any,
    Z: Any,
    Phi: Any,
    *,
    nfp: int = 1,
    batch_size: int = 4096,
    axisymmetric: bool = False,
    name: str = "ESSOS magnetic field",
) -> Any:
    """Sample an ESSOS field into PyNA's production cylindrical field type.

    This is an outer-loop snapshot for PyNA/Cyna tracing, not a differentiable
    inner-loop operation.  ESSOS evaluates its Cartesian field in bounded JAX
    batches, converts ``(Bx, By, Bz)`` to ``(BR, BZ, BPhi)``, and hands the
    resulting host arrays to PyNA.  ``Phi`` should normally be a uniform
    endpoint-free grid covering one field period, ``2*pi/nfp``.
    """

    if not hasattr(field, "B"):
        raise TypeError("field must provide B(xyz) in Cartesian coordinates")
    VectorFieldCylind = _pyna_cylindrical_field_class()
    radial_axis = _sampling_axis(R, "R", minimum_size=2)
    vertical_axis = _sampling_axis(Z, "Z", minimum_size=2)
    toroidal_axis = _sampling_axis(Phi, "Phi", minimum_size=1)
    if np.any(radial_axis <= 0.0):
        raise ValueError("R must contain only positive cylindrical radii")
    periods = _positive_integer(nfp, "nfp")
    chunk_size = _positive_integer(batch_size, "batch_size")

    radius, vertical, phi = np.meshgrid(
        radial_axis,
        vertical_axis,
        toroidal_axis,
        indexing="ij",
    )
    xyz = np.stack(
        (
            radius * np.cos(phi),
            radius * np.sin(phi),
            vertical,
        ),
        axis=-1,
    ).reshape(-1, 3)
    effective_chunk_size = min(chunk_size, xyz.shape[0])
    cartesian = np.empty_like(xyz)
    for start in range(0, xyz.shape[0], effective_chunk_size):
        stop = min(start + effective_chunk_size, xyz.shape[0])
        batch = xyz[start:stop]
        if batch.shape[0] < effective_chunk_size:
            padding = np.repeat(
                batch[-1:],
                effective_chunk_size - batch.shape[0],
                axis=0,
            )
            batch = np.concatenate((batch, padding), axis=0)
        sampled = np.asarray(
            jax.device_get(
                _batched_essos_field_callable(jnp.asarray(batch), field)
            ),
            dtype=float,
        )
        if sampled.shape != (effective_chunk_size, 3):
            raise ValueError("field.B must return one Cartesian 3-vector per point")
        cartesian[start:stop] = sampled[: stop - start]
    if not np.all(np.isfinite(cartesian)):
        raise ValueError("ESSOS field sampling produced non-finite values")

    flat_phi = phi.ravel()
    cos_phi = np.cos(flat_phi)
    sin_phi = np.sin(flat_phi)
    bx = cartesian[:, 0]
    by = cartesian[:, 1]
    shape = radius.shape
    b_r = (bx * cos_phi + by * sin_phi).reshape(shape)
    b_z = cartesian[:, 2].reshape(shape)
    b_phi = (-bx * sin_phi + by * cos_phi).reshape(shape)

    return VectorFieldCylind(
        radial_axis,
        vertical_axis,
        toroidal_axis,
        BR=b_r,
        BZ=b_z,
        BPhi=b_phi,
        nfp=periods,
        name=str(name),
        units="T",
        axisymmetric=bool(axisymmetric),
    )


def biot_savart_field_callable(xyz: Any, coils: Any) -> Any:
    """Evaluate the ESSOS Biot--Savart field with ``coils`` as parameters."""

    return BiotSavart(coils).B(xyz)


def _pyna_poincare_map():
    try:
        from pyna.toroidal.flt.jax_poincare import poincare_map
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS manifold tracing requires the PyNA optional-JAX backend"
        ) from exc
    return poincare_map


def _pyna_periodic_api():
    try:
        from pyna.topo.jax_periodic import (
            periodic_orbit_trajectory_state,
            periodic_point_state,
            solve_periodic_point,
        )
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS X-line objectives require PyNA's optional-JAX topology backend"
        ) from exc
    return (
        solve_periodic_point,
        periodic_point_state,
        periodic_orbit_trajectory_state,
    )


def _pyna_manifold_api():
    try:
        from pyna.topo.jax_manifold import (
            manifold_seed_segment,
            trace_manifold_generations,
        )
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS manifold objectives require PyNA's optional-JAX "
            "topology backend"
        ) from exc
    return manifold_seed_segment, trace_manifold_generations


def _pyna_strike_api():
    try:
        from pyna.topo.jax_strike import trace_manifold_strike
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS strike objectives require PyNA's optional-JAX "
            "topology backend"
        ) from exc
    return trace_manifold_strike


class ManifoldBranchTrace(NamedTuple):
    """Differentiable samples for one outer-identified manifold branch."""

    anchor: Any
    seeds: Any
    generations: Any


def fixed_phi_poincare_map(
    field: Any,
    rz0: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    n_steps: int = 256,
    bphi_floor: float = 0.0,
) -> Any:
    """Trace PyNA's differentiable map using an ESSOS field object."""

    return _pyna_poincare_map()(
        essos_field_callable,
        field,
        rz0,
        phi_span=phi_span,
        phi_start=phi_start,
        n_steps=n_steps,
        bphi_floor=bphi_floor,
    )


def fixed_phi_poincare_map_from_coils(
    coils: Any,
    rz0: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    n_steps: int = 256,
    bphi_floor: float = 0.0,
) -> Any:
    """Trace PyNA's differentiable map with ESSOS coils as the design PyTree."""

    return _pyna_poincare_map()(
        biot_savart_field_callable,
        coils,
        rz0,
        phi_span=phi_span,
        phi_start=phi_start,
        n_steps=n_steps,
        bphi_floor=bphi_floor,
    )


def periodic_xline_position(
    field: Any,
    initial_guess: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
) -> Any:
    """Locate an ESSOS field's periodic X-line on one Poincare section.

    PyNA solves the periodic-point equation and supplies its implicit
    derivative.  ``initial_guess`` and the orbit identity are expected to be
    refreshed by the outer topology-tracking loop rather than optimized as
    ESSOS design variables.
    """

    solve_periodic_point, _, _ = _pyna_periodic_api()
    return solve_periodic_point(
        essos_field_callable,
        field,
        initial_guess,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=map_power,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
    )


def periodic_xline_state(
    field: Any,
    initial_guess: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
    residual_tolerance: float = 1.0e-10,
) -> Any:
    """Return PyNA's position, monodromy, and stability data for an X-line."""

    _, periodic_point_state, _ = _pyna_periodic_api()
    return periodic_point_state(
        essos_field_callable,
        field,
        initial_guess,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=map_power,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
        residual_tolerance=residual_tolerance,
    )


def periodic_xline_trajectory(
    field: Any,
    initial_guess: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
    residual_tolerance: float = 1.0e-10,
) -> Any:
    """Return PyNA's differentiable trajectory for a tracked periodic X-line."""

    _, _, periodic_orbit_trajectory_state = _pyna_periodic_api()
    return periodic_orbit_trajectory_state(
        essos_field_callable,
        field,
        initial_guess,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=map_power,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
        residual_tolerance=residual_tolerance,
    )


def periodic_xline_clearance_residuals(
    field: Any,
    initial_guess: Any,
    wall_signed_distance: Callable[[Any], Any],
    *,
    minimum_clearance_m: float,
    clearance_scale_m: float,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
    residual_tolerance: float = 1.0e-10,
    sample_stride: int = 1,
) -> Any:
    """Return normalized clearance violations along a periodic X-line.

    ``wall_signed_distance(point_xyz)`` must return a scalar physical distance
    in metres that is positive on the allowed side of the wall.  The closing
    endpoint is omitted because it duplicates the launch point.  This smooth
    inner-loop hinge does not replace PyNA/Cyna's global wall-hit validation.
    """

    if not callable(wall_signed_distance):
        raise TypeError("wall_signed_distance must be callable")
    minimum = float(minimum_clearance_m)
    scale = float(clearance_scale_m)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_clearance_m must be non-negative and finite")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("clearance_scale_m must be positive and finite")
    stride = _positive_integer(sample_stride, "sample_stride")

    trajectory = periodic_xline_trajectory(
        field,
        initial_guess,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=map_power,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
        residual_tolerance=residual_tolerance,
    )
    sample_points = trajectory.point_xyz[:-1:stride]

    def distance(point_xyz):
        value = jnp.asarray(wall_signed_distance(point_xyz))
        if value.shape != ():
            raise ValueError("wall_signed_distance must return a scalar")
        return value

    clearances = jax.vmap(distance)(sample_points)
    return jnp.maximum(minimum - clearances, 0.0) / scale


def periodic_xline_clearance_loss(
    field: Any,
    initial_guess: Any,
    wall_signed_distance: Callable[[Any], Any],
    *,
    minimum_clearance_m: float,
    clearance_scale_m: float,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
    residual_tolerance: float = 1.0e-10,
    sample_stride: int = 1,
) -> Any:
    """Mean squared-hinge wall-clearance loss for a tracked periodic X-line."""

    residuals = periodic_xline_clearance_residuals(
        field,
        initial_guess,
        wall_signed_distance,
        minimum_clearance_m=minimum_clearance_m,
        clearance_scale_m=clearance_scale_m,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=map_power,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
        residual_tolerance=residual_tolerance,
        sample_stride=sample_stride,
    )
    return 0.5 * jnp.mean(residuals * residuals)


def periodic_xline_location_loss(
    field: Any,
    initial_guess: Any,
    target_rz: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
    rz_scales: Any | None = None,
) -> Any:
    """Mean-square normalized displacement of a tracked X-line section point."""

    target = jnp.asarray(target_rz)
    if target.shape != (2,):
        raise ValueError("target_rz must have shape (2,)")
    scales = jnp.ones_like(target) if rz_scales is None else jnp.asarray(rz_scales)
    if scales.shape != (2,):
        raise ValueError("rz_scales must have shape (2,)")

    position = periodic_xline_position(
        field,
        initial_guess,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=map_power,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
    )
    normalized_displacement = (position - target) / scales
    return jnp.mean(normalized_displacement * normalized_displacement)


def periodic_xline_hyperbolicity_loss(
    field: Any,
    initial_guess: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
    minimum_margin: float = 0.0,
    maximum_margin: float | None = None,
    branch_sign: float = 1.0,
) -> Any:
    """Squared-hinge penalty for a tracked direct or inverse hyperbolic X-line.

    For a flux-preserving two-dimensional return map, direct hyperbolicity has
    ``trace(M) > 2`` and inverse hyperbolicity has ``trace(M) < -2``.  Fixing
    ``branch_sign`` to ``+1`` or ``-1`` preserves that outer-loop branch choice.
    The signed margin used here is ``branch_sign * trace(M) - 2``.
    """

    sign = float(branch_sign)
    if sign not in (-1.0, 1.0):
        raise ValueError("branch_sign must be +1 or -1")
    lower = float(minimum_margin)
    if lower < 0.0:
        raise ValueError("minimum_margin must be non-negative")
    upper = None if maximum_margin is None else float(maximum_margin)
    if upper is not None and upper < lower:
        raise ValueError("maximum_margin must be at least minimum_margin")

    state = periodic_xline_state(
        field,
        initial_guess,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=map_power,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
    )
    signed_margin = sign * state.trace - 2.0
    penalty = jnp.square(jnp.maximum(lower - signed_margin, 0.0))
    if upper is not None:
        penalty = penalty + jnp.square(jnp.maximum(signed_margin - upper, 0.0))
    return penalty


def trace_manifold_branch(
    field: Any,
    initial_guess: Any,
    branch_direction: Any,
    seed_distances: Any,
    *,
    stability: str,
    side: int,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_generations: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
) -> ManifoldBranchTrace:
    """Trace one fixed-label stable or unstable branch of an ESSOS field.

    PyNA owns both seed construction and map iteration.  ESSOS supplies the
    active field and the branch data selected by an outer topology refresh.
    For a stable branch, the anchor is solved against the discrete backward
    map so the seeds and subsequent generations use the same integrator.
    ``branch_direction`` and ``seed_distances`` are frozen by PyNA.
    """

    branch = str(stability).lower()
    if branch not in {"stable", "unstable"}:
        raise ValueError("stability must be 'stable' or 'unstable'")
    anchor_span = phi_span if branch == "unstable" else -jnp.asarray(phi_span)
    anchor = periodic_xline_position(
        field,
        initial_guess,
        phi_span=anchor_span,
        phi_start=phi_start,
        map_power=map_power,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
    )

    manifold_seed_segment, trace_manifold_generations = _pyna_manifold_api()
    seeds = manifold_seed_segment(
        anchor,
        branch_direction,
        seed_distances,
        side=side,
    )
    generations = trace_manifold_generations(
        essos_field_callable,
        field,
        seeds,
        stability=branch,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=map_power,
        n_generations=n_generations,
        n_steps_per_span=n_steps_per_span,
        bphi_floor=bphi_floor,
    )
    return ManifoldBranchTrace(
        anchor=anchor,
        seeds=seeds,
        generations=generations,
    )


def trace_manifold_wall_strike(
    field: Any,
    branch_reference: Any,
    strike_match: Any,
    wall_plane: Any,
    *,
    n_steps_per_span: int = 256,
    xline_newton_iterations: int = 8,
    xline_newton_damping: float = 1.0,
    wall_n_steps: int = 2048,
    wall_newton_iterations: int = 8,
    wall_newton_damping: float = 1.0,
    maximum_phi_shift: float,
    bphi_floor: float = 0.0,
    xline_residual_tolerance: float = 1.0e-10,
    wall_residual_tolerance: float = 1.0e-10,
    minimum_abs_transversality: float = 1.0e-8,
) -> Any:
    """Trace one PyNA-labelled manifold seed to its local wall plane.

    The ESSOS magnetic field is the active JAX PyTree.  PyNA owns periodic
    X-line continuation, exact sparse seed reconstruction, the implicit wall
    event, and all returned validity diagnostics.
    """

    return _pyna_strike_api()(
        essos_field_callable,
        field,
        branch_reference,
        strike_match,
        wall_plane,
        n_steps_per_span=n_steps_per_span,
        xline_newton_iterations=xline_newton_iterations,
        xline_newton_damping=xline_newton_damping,
        wall_n_steps=wall_n_steps,
        wall_newton_iterations=wall_newton_iterations,
        wall_newton_damping=wall_newton_damping,
        maximum_phi_shift=maximum_phi_shift,
        bphi_floor=bphi_floor,
        xline_residual_tolerance=xline_residual_tolerance,
        wall_residual_tolerance=wall_residual_tolerance,
        minimum_abs_transversality=minimum_abs_transversality,
    )


def _sample_index(value: int, size: int, name: str) -> int:
    try:
        index = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    normalized = index if index >= 0 else size + index
    if normalized < 0 or normalized >= size:
        raise IndexError(f"{name} is outside the traced branch")
    return normalized


def manifold_sample_location_loss(
    field: Any,
    initial_guess: Any,
    branch_direction: Any,
    seed_distances: Any,
    target_rz: Any,
    *,
    stability: str,
    side: int,
    generation_index: int,
    seed_index: int,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_generations: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
    rz_scales: Any | None = None,
) -> Any:
    """Target one fixed outer-labelled manifold sample in ``(R, Z)``.

    This is a smooth inner-loop proxy for lobe or divertor-leg placement.  The
    outer PyNA/Cyna refresh owns branch identity, wall intersection, and any
    change of the selected sample correspondence.
    """

    target = jnp.asarray(target_rz)
    if target.shape != (2,):
        raise ValueError("target_rz must have shape (2,)")
    scales = jnp.ones_like(target) if rz_scales is None else jnp.asarray(rz_scales)
    if scales.shape != (2,):
        raise ValueError("rz_scales must have shape (2,)")

    trace = trace_manifold_branch(
        field,
        initial_guess,
        branch_direction,
        seed_distances,
        stability=stability,
        side=side,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=map_power,
        n_generations=n_generations,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
    )
    generation = _sample_index(
        generation_index,
        trace.generations.shape[0],
        "generation_index",
    )
    seed = _sample_index(seed_index, trace.generations.shape[1], "seed_index")
    residual = (trace.generations[generation, seed] - target) / scales
    return jnp.mean(residual * residual)


def return_map_surface_residuals(
    field: Any,
    rz_seeds: Any,
    target_signed_distance: Callable[[Any, Any], Any],
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    n_steps: int = 256,
    bphi_floor: float = 0.0,
) -> Any:
    """Return signed target-surface distances after one fixed-phi map.

    ``rz_seeds`` are fixed Stage-1 edge points on the launch section.
    ``target_signed_distance(rz, phi)`` supplies the differentiable signed
    distance to the desired surface on the destination section.  Keeping that
    geometry callback in ESSOS preserves the package ownership boundary while
    PyNA remains responsible for every field-line integration.
    """

    seeds = jnp.asarray(rz_seeds)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError("rz_seeds must have shape (n_seeds, 2)")

    def trace_seed(rz):
        return fixed_phi_poincare_map(
            field,
            rz,
            phi_span=phi_span,
            phi_start=phi_start,
            n_steps=n_steps,
            bphi_floor=bphi_floor,
        )

    endpoints = jax.vmap(trace_seed)(seeds)
    phi_end = jnp.asarray(phi_start) + jnp.asarray(phi_span)

    def distance_to_target(rz):
        return jnp.asarray(target_signed_distance(rz, phi_end)).squeeze()

    return jax.vmap(distance_to_target)(endpoints)


def return_map_surface_loss(
    field: Any,
    rz_seeds: Any,
    target_signed_distance: Callable[[Any, Any], Any],
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    n_steps: int = 256,
    bphi_floor: float = 0.0,
    weights: Any | None = None,
) -> Any:
    """Mean-square surface-invariance objective for Stage-2 coil design.

    When ``weights`` are provided, the result is the weighted mean-square
    signed distance.  The weights are normalized internally, so their absolute
    scale does not alter the objective.
    """

    residuals = return_map_surface_residuals(
        field,
        rz_seeds,
        target_signed_distance,
        phi_span=phi_span,
        phi_start=phi_start,
        n_steps=n_steps,
        bphi_floor=bphi_floor,
    )
    squared = residuals * residuals
    if weights is None:
        return jnp.mean(squared)

    weights_array = jnp.asarray(weights)
    if weights_array.shape != squared.shape:
        raise ValueError("weights must have shape (n_seeds,)")
    return jnp.sum(weights_array * squared) / jnp.sum(weights_array)


__all__ = [
    "ManifoldBranchTrace",
    "biot_savart_field_callable",
    "essos_field_to_pyna_cylindrical_grid",
    "essos_field_callable",
    "fixed_phi_poincare_map",
    "fixed_phi_poincare_map_from_coils",
    "manifold_sample_location_loss",
    "periodic_xline_clearance_loss",
    "periodic_xline_clearance_residuals",
    "periodic_xline_hyperbolicity_loss",
    "periodic_xline_location_loss",
    "periodic_xline_position",
    "periodic_xline_state",
    "periodic_xline_trajectory",
    "return_map_surface_loss",
    "return_map_surface_residuals",
    "trace_manifold_branch",
    "trace_manifold_wall_strike",
]
