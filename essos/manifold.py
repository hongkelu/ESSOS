"""ESSOS adapters for PyNA's differentiable topology primitives.

ESSOS owns the Stage-2 design variables and magnetic-field models.  PyNA owns
field-line maps and topology.  The functions here provide the one-way bridge
from ESSOS to the optional JAX backend in PyNA without introducing an ESSOS
dependency in PyNA.

PyNA is imported lazily so ordinary ESSOS installations do not require it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
    "biot_savart_field_callable",
    "essos_field_callable",
    "fixed_phi_poincare_map",
    "fixed_phi_poincare_map_from_coils",
    "return_map_surface_loss",
    "return_map_surface_residuals",
]
