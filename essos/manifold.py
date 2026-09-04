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

from essos.fields import BiotSavart


def essos_field_callable(xyz: Any, field: Any) -> Any:
    """Evaluate an ESSOS magnetic-field object for PyNA.

    ``field`` is passed as the active JAX PyTree, so derivatives of a PyNA
    return map propagate through any dynamic leaves registered by the ESSOS
    field implementation.
    """

    return field.B(xyz)


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
            periodic_point_state,
            solve_periodic_point,
        )
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS X-line objectives require PyNA's optional-JAX topology backend"
        ) from exc
    return solve_periodic_point, periodic_point_state


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

    solve_periodic_point, _ = _pyna_periodic_api()
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

    _, periodic_point_state = _pyna_periodic_api()
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
    "essos_field_callable",
    "fixed_phi_poincare_map",
    "fixed_phi_poincare_map_from_coils",
    "manifold_sample_location_loss",
    "periodic_xline_hyperbolicity_loss",
    "periodic_xline_location_loss",
    "periodic_xline_position",
    "periodic_xline_state",
    "return_map_surface_loss",
    "return_map_surface_residuals",
    "trace_manifold_branch",
]
