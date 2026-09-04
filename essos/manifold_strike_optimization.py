"""Stage-2 coil objectives for PyNA-labelled first-wall strikes.

PyNA/Cyna selects a production manifold hit and freezes its exact branch/seed
label plus a local wall tangent plane.  ESSOS stores that accepted outer state
while JAX differentiates the moving X-line, seed, trajectory, and wall event
with respect to the active coil field.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from essos.losses import custom_loss
from essos.manifold import trace_manifold_wall_strike


def _pyna_strike_contract_api():
    try:
        from pyna.topo.manifold_correspondence import ManifoldBranchReference
        from pyna.topo.manifold_strike_contracts import (
            LocalWallPlane,
            ManifoldStrikeMatch,
        )
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS strike optimization requires PyNA's manifold-strike "
            "contracts"
        ) from exc
    return ManifoldBranchReference, ManifoldStrikeMatch, LocalWallPlane


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_vector(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).ravel()
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-{size} vector")
    return array


def _nonnegative_scalar(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _damping(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return result


@dataclass(frozen=True)
class ManifoldStrikeStage2Target:
    """Immutable production label and local wall model for one inner solve."""

    branch_reference: Any
    strike_match: Any
    wall_plane: Any
    target_position_m: np.ndarray
    position_scales_m: np.ndarray
    maximum_phi_shift: float
    n_steps_per_span: int = 256
    xline_newton_iterations: int = 8
    xline_newton_damping: float = 1.0
    wall_n_steps: int = 2048
    wall_newton_iterations: int = 8
    wall_newton_damping: float = 1.0
    bphi_floor: float = 0.0
    xline_residual_tolerance: float = 1.0e-10
    wall_residual_tolerance: float = 1.0e-10
    minimum_abs_transversality: float = 1.0e-8

    def __post_init__(self) -> None:
        (
            ManifoldBranchReference,
            ManifoldStrikeMatch,
            LocalWallPlane,
        ) = _pyna_strike_contract_api()
        if not isinstance(self.branch_reference, ManifoldBranchReference):
            raise TypeError("branch_reference must be a PyNA ManifoldBranchReference")
        if not isinstance(self.strike_match, ManifoldStrikeMatch):
            raise TypeError("strike_match must be a PyNA ManifoldStrikeMatch")
        if not isinstance(self.wall_plane, LocalWallPlane):
            raise TypeError("wall_plane must be a PyNA LocalWallPlane")
        if self.strike_match.label.branch_label != self.branch_reference.label:
            raise ValueError("strike_match belongs to a different manifold branch")
        if self.wall_plane.strike_label != self.strike_match.label:
            raise ValueError("wall_plane belongs to a different strike label")
        self.branch_reference.seed_index(self.strike_match.label.seed_order)

        coordinate_size = 2 if self.strike_match.distance_mode == "rz" else 3
        target = _finite_vector(
            self.target_position_m,
            coordinate_size,
            "target_position_m",
        )
        scales = _finite_vector(
            self.position_scales_m,
            coordinate_size,
            "position_scales_m",
        )
        if np.any(scales <= 0.0):
            raise ValueError("position_scales_m must be positive")

        maximum_phi_shift = float(self.maximum_phi_shift)
        if not np.isfinite(maximum_phi_shift) or maximum_phi_shift <= 0.0:
            raise ValueError("maximum_phi_shift must be positive and finite")

        object.__setattr__(self, "target_position_m", target.copy())
        object.__setattr__(self, "position_scales_m", scales.copy())
        object.__setattr__(self, "maximum_phi_shift", maximum_phi_shift)
        for attribute in (
            "n_steps_per_span",
            "xline_newton_iterations",
            "wall_n_steps",
            "wall_newton_iterations",
        ):
            object.__setattr__(
                self,
                attribute,
                _positive_integer(getattr(self, attribute), attribute),
            )
        for attribute in (
            "xline_newton_damping",
            "wall_newton_damping",
        ):
            object.__setattr__(
                self,
                attribute,
                _damping(getattr(self, attribute), attribute),
            )
        for attribute in (
            "bphi_floor",
            "xline_residual_tolerance",
            "wall_residual_tolerance",
            "minimum_abs_transversality",
        ):
            object.__setattr__(
                self,
                attribute,
                _nonnegative_scalar(getattr(self, attribute), attribute),
            )

    @property
    def label(self) -> str:
        return self.strike_match.label.key

    @property
    def distance_mode(self) -> str:
        return self.strike_match.distance_mode


def trace_manifold_strike_reference(
    field: Any,
    target_state: ManifoldStrikeStage2Target,
) -> Any:
    """Trace the target's exact labelled seed through the live ESSOS field."""

    if not isinstance(target_state, ManifoldStrikeStage2Target):
        raise TypeError("target_state must be ManifoldStrikeStage2Target")
    return trace_manifold_wall_strike(
        field,
        target_state.branch_reference,
        target_state.strike_match,
        target_state.wall_plane,
        n_steps_per_span=target_state.n_steps_per_span,
        xline_newton_iterations=target_state.xline_newton_iterations,
        xline_newton_damping=target_state.xline_newton_damping,
        wall_n_steps=target_state.wall_n_steps,
        wall_newton_iterations=target_state.wall_newton_iterations,
        wall_newton_damping=target_state.wall_newton_damping,
        maximum_phi_shift=target_state.maximum_phi_shift,
        bphi_floor=target_state.bphi_floor,
        xline_residual_tolerance=target_state.xline_residual_tolerance,
        wall_residual_tolerance=target_state.wall_residual_tolerance,
        minimum_abs_transversality=target_state.minimum_abs_transversality,
    )


def manifold_strike_stage2_target_loss(
    field: Any,
    *,
    target_state: ManifoldStrikeStage2Target,
) -> Any:
    """Normalized position loss for one immutable first-wall strike label."""

    trace = trace_manifold_strike_reference(field, target_state)
    if target_state.distance_mode == "rz":
        position = trace.intersection.point_RZPhi[:2]
    else:
        position = trace.intersection.point_xyz
    residual = (
        position - jnp.asarray(target_state.target_position_m)
    ) / jnp.asarray(target_state.position_scales_m)
    return 0.5 * jnp.sum(residual * residual)


def make_manifold_strike_stage2_loss(
    target_state: ManifoldStrikeStage2Target,
    *,
    field_dependency: str = "field",
) -> custom_loss:
    """Build an ESSOS custom loss for one accepted strike target."""

    if not isinstance(target_state, ManifoldStrikeStage2Target):
        raise TypeError("target_state must be ManifoldStrikeStage2Target")
    dependency = str(field_dependency).strip()
    if not dependency:
        raise ValueError("field_dependency must not be empty")
    return custom_loss(
        manifold_strike_stage2_target_loss,
        dependency,
        target_state=target_state,
    )


__all__ = [
    "ManifoldStrikeStage2Target",
    "make_manifold_strike_stage2_loss",
    "manifold_strike_stage2_target_loss",
    "trace_manifold_strike_reference",
]
