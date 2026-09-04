"""ESSOS adapters for PyNA's differentiable topology primitives.

ESSOS owns the Stage-2 design variables and magnetic-field models.  PyNA owns
field-line maps and topology.  The functions here provide the one-way bridge
from ESSOS to the optional JAX backend in PyNA without introducing an ESSOS
dependency in PyNA.

PyNA is imported lazily so ordinary ESSOS installations do not require it.
"""

from __future__ import annotations

from typing import Any

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


__all__ = [
    "biot_savart_field_callable",
    "essos_field_callable",
    "fixed_phi_poincare_map",
    "fixed_phi_poincare_map_from_coils",
]
