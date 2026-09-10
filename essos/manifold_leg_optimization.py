"""Differentiable pre-strike clearance for exact invariant-manifold legs.

PyNA supplies full trajectories from each accepted sparse manifold seed to its
implicitly moving local wall event.  ESSOS evaluates a smooth signed-distance
wall model on a fixed interior portion of those paths.  A terminal path
fraction is excluded because the intended divertor strike must approach and
intersect the wall; the excluded sample indices remain static during an inner
solve.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from essos.losses import custom_loss
from essos.manifold import trace_manifold_wall_strike_bundle_trajectories
from essos.manifold_heat_optimization import ManifoldHeatStage2Target
from essos.manifold_strike_optimization import ManifoldStrikeStage2Target


@dataclass(frozen=True)
class ManifoldLegStage2Target:
    """Frozen labelled legs and clearance physics, independent of heat data.

    Exclusion is a fraction of signed toroidal-angle span, not arc length.
    The wall callback is static configuration and must not be mutated.
    """

    branch_reference: Any
    strike_matches: tuple[Any, ...]
    wall_planes: tuple[Any, ...]
    wall_signed_distance: Callable[[Any], Any]
    minimum_clearance_m: float
    clearance_scale_m: float
    terminal_exclusion_fraction: float
    maximum_phi_shift: float
    sample_stride: int = 1
    strike_weights: Any | None = None
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
        matches, planes = tuple(self.strike_matches), tuple(self.wall_planes)
        if not matches or len(matches) != len(planes):
            raise ValueError(
                "strike_matches and wall_planes must be non-empty and aligned"
            )
        # Reuse the single-strike contract, including all integration controls.
        controls = {name: getattr(self, name) for name in _TRACE_CONTROLS}
        for match, plane in zip(matches, planes, strict=True):
            validated = ManifoldStrikeStage2Target(
                branch_reference=self.branch_reference,
                strike_match=match,
                wall_plane=plane,
                target_position_m=match.distance_coordinates_m,
                position_scales_m=np.ones_like(match.distance_coordinates_m),
                **controls,
            )
        physical_sign = (
            np.sign(self.branch_reference.map_span)
            * (1 if self.branch_reference.stability == "unstable" else -1)
        )
        expected_direction = "+" if physical_sign > 0 else "-"
        if tuple(match.label.seed_order for match in matches) != tuple(
            self.branch_reference.seed_orders
        ) or any(match.label.direction != expected_direction for match in matches):
            raise ValueError(
                "strike matches must preserve complete ordered branch identity"
            )
        if not callable(self.wall_signed_distance):
            raise TypeError("wall_signed_distance must be callable")
        for name in _TRACE_CONTROLS:
            object.__setattr__(self, name, getattr(validated, name))
        for name in ("minimum_clearance_m", "clearance_scale_m"):
            object.__setattr__(
                self, name, _nonnegative_finite(getattr(self, name), name)
            )
        if self.clearance_scale_m == 0:
            raise ValueError("clearance_scale_m must be positive")
        fraction = float(self.terminal_exclusion_fraction)
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("terminal_exclusion_fraction must lie in (0, 1)")
        object.__setattr__(self, "terminal_exclusion_fraction", fraction)
        object.__setattr__(
            self, "sample_stride",
            _positive_integer(self.sample_stride, "sample_stride"),
        )
        weights = (
            np.ones(len(matches)) if self.strike_weights is None
            else np.asarray(self.strike_weights, dtype=float)
        )
        if (
            weights.shape != (len(matches),)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not 0 < np.sum(weights) < np.inf
        ):
            raise ValueError(
                "strike_weights must be non-negative, finite, aligned, "
                "and have positive finite total"
            )
        # A tuple prevents callers from mutating weights through a frozen target.
        object.__setattr__(self, "strike_weights", tuple(map(float, weights)))
        object.__setattr__(self, "strike_matches", matches)
        object.__setattr__(self, "wall_planes", planes)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(match.label.key for match in self.strike_matches)


_TRACE_CONTROLS = (
    "maximum_phi_shift", "n_steps_per_span", "xline_newton_iterations",
    "xline_newton_damping", "wall_n_steps", "wall_newton_iterations",
    "wall_newton_damping", "bphi_floor", "xline_residual_tolerance",
    "wall_residual_tolerance", "minimum_abs_transversality",
)


def require_same_leg_configuration(
    previous: ManifoldLegStage2Target,
    candidate: ManifoldLegStage2Target,
) -> None:
    """Refresh may move geometry, but cannot change labels or fixed physics."""
    if not isinstance(candidate, ManifoldLegStage2Target):
        raise TypeError("accepted leg target must be ManifoldLegStage2Target")
    previous_modes = tuple(m.distance_mode for m in previous.strike_matches)
    candidate_modes = tuple(m.distance_mode for m in candidate.strike_matches)
    if previous.labels != candidate.labels or previous_modes != candidate_modes:
        raise ValueError("accepted leg refresh changed strike identity")
    for item in fields(previous):
        if item.name in ("branch_reference", "strike_matches", "wall_planes"):
            continue
        old, new = getattr(previous, item.name), getattr(candidate, item.name)
        same = old is new if item.name == "wall_signed_distance" else old == new
        if not same:
            raise ValueError(f"accepted leg refresh changed {item.name}")


def refresh_manifold_leg_stage2_target(
    target_state: ManifoldLegStage2Target,
    candidate_branch: Any,
    report: Any,
) -> ManifoldLegStage2Target:
    """Refresh geometry only after the exact production clearance gate passes."""
    from pyna.topo.manifold_leg_clearance import ManifoldLegClearanceReport

    if not isinstance(target_state, ManifoldLegStage2Target):
        raise TypeError("target_state must be ManifoldLegStage2Target")
    if not isinstance(report, ManifoldLegClearanceReport) or not report.accepted:
        raise ValueError("leg target refresh requires accepted production leg clearance")
    if report.branch_label != candidate_branch.label:
        raise ValueError("leg clearance belongs to a different branch")
    if (
        report.required_minimum_clearance_m != target_state.minimum_clearance_m
        or report.terminal_exclusion_fraction != target_state.terminal_exclusion_fraction
    ):
        raise ValueError("leg clearance report changed the physical requirements")
    refreshed = replace(
        target_state, branch_reference=candidate_branch,
        strike_matches=report.strike_matches, wall_planes=report.wall_planes,
    )
    require_same_leg_configuration(target_state, refreshed)
    return refreshed


def make_manifold_leg_stage2_loss(
    target_state: ManifoldLegStage2Target,
    *,
    field_dependency: str = "field",
) -> custom_loss:
    """Build the loss using only the active target's frozen configuration."""
    if not isinstance(target_state, ManifoldLegStage2Target):
        raise TypeError("target_state must be ManifoldLegStage2Target")
    return make_manifold_leg_clearance_loss(
        target_state, target_state.wall_signed_distance,
        minimum_clearance_m=target_state.minimum_clearance_m,
        clearance_scale_m=target_state.clearance_scale_m,
        terminal_exclusion_fraction=target_state.terminal_exclusion_fraction,
        sample_stride=target_state.sample_stride,
        strike_weights=target_state.strike_weights,
        field_dependency=field_dependency,
    )


class ManifoldLegClearanceState(NamedTuple):
    """Full labelled paths and retained pre-strike clearance diagnostics."""

    trajectories: Any
    signed_clearance_m: Any
    normalized_violation: Any


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


def _nonnegative_finite(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def trace_manifold_leg_reference(
    field: Any,
    target_state: ManifoldLegStage2Target | ManifoldHeatStage2Target,
) -> Any:
    """Trace full exact-label paths through the live ESSOS field."""

    if not isinstance(target_state, (ManifoldLegStage2Target, ManifoldHeatStage2Target)):
        raise TypeError("target_state must be a labelled leg or heat target")
    return trace_manifold_wall_strike_bundle_trajectories(
        field,
        target_state.branch_reference,
        target_state.strike_matches,
        target_state.wall_planes,
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


def manifold_leg_clearance_state(
    field: Any,
    wall_signed_distance: Callable[[Any], Any],
    *,
    target_state: ManifoldLegStage2Target | ManifoldHeatStage2Target,
    minimum_clearance_m: float,
    clearance_scale_m: float,
    terminal_exclusion_fraction: float,
    sample_stride: int = 1,
) -> ManifoldLegClearanceState:
    """Evaluate signed clearance on a fixed pre-strike portion of every leg.

    ``wall_signed_distance(point_xyz)`` returns metres, positive on the allowed
    side.  For ``N`` RK4 intervals, retained indices satisfy
    ``index / N < 1 - terminal_exclusion_fraction``.  The endpoint is therefore
    always excluded and the selection does not depend on live field values.
    """

    if not callable(wall_signed_distance):
        raise TypeError("wall_signed_distance must be callable")
    if not isinstance(target_state, (ManifoldLegStage2Target, ManifoldHeatStage2Target)):
        raise TypeError("target_state must be a labelled leg or heat target")
    minimum = _nonnegative_finite(minimum_clearance_m, "minimum_clearance_m")
    scale = _nonnegative_finite(clearance_scale_m, "clearance_scale_m")
    if scale == 0.0:
        raise ValueError("clearance_scale_m must be positive")
    fraction = float(terminal_exclusion_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("terminal_exclusion_fraction must lie in (0, 1)")
    stride = _positive_integer(sample_stride, "sample_stride")
    retained_stop = max(
        1,
        int(math.ceil((1.0 - fraction) * target_state.wall_n_steps)),
    )

    trajectories = trace_manifold_leg_reference(field, target_state)
    retained_points = trajectories.point_xyz[:, :retained_stop:stride, :]
    flat_points = retained_points.reshape((-1, 3))

    def distance(point_xyz):
        value = jnp.asarray(wall_signed_distance(point_xyz))
        if value.shape != ():
            raise ValueError("wall_signed_distance must return a scalar")
        return value

    signed_clearance = jax.vmap(distance)(flat_points).reshape(
        retained_points.shape[:2]
    )
    violation = jnp.maximum(minimum - signed_clearance, 0.0) / scale
    return ManifoldLegClearanceState(
        trajectories=trajectories,
        signed_clearance_m=signed_clearance,
        normalized_violation=violation,
    )


def manifold_leg_clearance_loss(
    field: Any,
    *,
    wall_signed_distance: Callable[[Any], Any],
    target_state: ManifoldLegStage2Target | ManifoldHeatStage2Target,
    minimum_clearance_m: float,
    clearance_scale_m: float,
    terminal_exclusion_fraction: float,
    sample_stride: int = 1,
    strike_weights: Any | None = None,
) -> Any:
    """Power-optional mean squared-hinge loss over retained leg samples."""

    state = manifold_leg_clearance_state(
        field,
        wall_signed_distance,
        target_state=target_state,
        minimum_clearance_m=minimum_clearance_m,
        clearance_scale_m=clearance_scale_m,
        terminal_exclusion_fraction=terminal_exclusion_fraction,
        sample_stride=sample_stride,
    )
    per_strike = jnp.mean(state.normalized_violation**2, axis=1)
    if strike_weights is None:
        return 0.5 * jnp.mean(per_strike)
    weights = np.asarray(strike_weights, dtype=float).ravel()
    if weights.shape != (len(target_state.strike_matches),):
        raise ValueError("strike_weights must have one value per strike label")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("strike_weights must be non-negative and finite")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("strike_weights must have positive total weight")
    active_weights = jnp.asarray(weights, dtype=per_strike.dtype)
    return 0.5 * jnp.sum(active_weights * per_strike) / jnp.sum(active_weights)


def make_manifold_leg_clearance_loss(
    target_state: ManifoldLegStage2Target | ManifoldHeatStage2Target,
    wall_signed_distance: Callable[[Any], Any],
    *,
    minimum_clearance_m: float,
    clearance_scale_m: float,
    terminal_exclusion_fraction: float,
    sample_stride: int = 1,
    strike_weights: Any | None = None,
    field_dependency: str = "field",
) -> custom_loss:
    """Build an ESSOS custom loss for fixed-label pre-strike clearance."""

    if not isinstance(target_state, (ManifoldLegStage2Target, ManifoldHeatStage2Target)):
        raise TypeError("target_state must be a labelled leg or heat target")
    if not callable(wall_signed_distance):
        raise TypeError("wall_signed_distance must be callable")
    dependency = str(field_dependency).strip()
    if not dependency:
        raise ValueError("field_dependency must not be empty")
    return custom_loss(
        manifold_leg_clearance_loss,
        dependency,
        wall_signed_distance=wall_signed_distance,
        target_state=target_state,
        minimum_clearance_m=minimum_clearance_m,
        clearance_scale_m=clearance_scale_m,
        terminal_exclusion_fraction=terminal_exclusion_fraction,
        sample_stride=sample_stride,
        strike_weights=strike_weights,
    )


__all__ = [
    "ManifoldLegStage2Target",
    "make_manifold_leg_stage2_loss",
    "refresh_manifold_leg_stage2_target",
    "ManifoldLegClearanceState",
    "make_manifold_leg_clearance_loss",
    "manifold_leg_clearance_loss",
    "manifold_leg_clearance_state",
    "trace_manifold_leg_reference",
]
