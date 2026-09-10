"""Production and live-JAX acceptance gates for pre-strike clearance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from essos.manifold_leg_optimization import (
    ManifoldLegStage2Target,
    manifold_leg_clearance_state,
    refresh_manifold_leg_stage2_target,
)


@dataclass(frozen=True)
class ManifoldLegValidationConfig:
    """Numerical trust limits; the active target owns the physical margin."""

    wall: Any
    maximum_hit_displacement_m: float
    jax_cyna_tolerance_m: float
    maximum_endpoint_error_m: float
    maximum_projection_distance_m: float = 1.0e-3
    max_turns: int = 100
    production_DPhi: float = 0.01
    extend_phi: bool = True
    production_strike_trace_function: Callable[..., Any] | None = None
    production_trajectory_trace_function: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "maximum_hit_displacement_m", "jax_cyna_tolerance_m",
            "maximum_endpoint_error_m", "maximum_projection_distance_m",
            "production_DPhi",
        ):
            value = float(getattr(self, name))
            if (
                not np.isfinite(value) or value < 0
                or (name == "production_DPhi" and value == 0)
            ):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.max_turns, (bool, np.bool_))
            or not np.isfinite(self.max_turns)
            or int(self.max_turns) != self.max_turns or self.max_turns <= 0
        ):
            raise ValueError("max_turns must be a positive integer")
        object.__setattr__(self, "max_turns", int(self.max_turns))
        for name in (
            "production_strike_trace_function", "production_trajectory_trace_function",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")


@dataclass(frozen=True)
class ManifoldLegValidationReport:
    """Both independent clearance measurements and the exact-strike parity gate."""

    production: Any
    correspondence: Any | None
    inner_state: Any | None
    accepted_target: ManifoldLegStage2Target | None
    accepted: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.accepted:
            if (
                not self.production.accepted or self.correspondence is None
                or not self.correspondence.accepted or self.inner_state is None
                or self.accepted_target is None or self.rejection_reason is not None
            ):
                raise ValueError("accepted leg validation requires every gate and a target")
            values = np.asarray(self.inner_state.signed_clearance_m)
            if (
                not values.size or not np.all(np.isfinite(values))
                or np.min(values) < self.accepted_target.minimum_clearance_m
            ):
                raise ValueError("accepted leg validation violates the inner margin")
        elif self.accepted_target is not None or self.rejection_reason is None:
            raise ValueError("rejected leg validation requires a reason and no accepted target")


def validate_manifold_leg_candidate(
    candidate_field: Any,
    production_field: Any,
    candidate_branch: Any,
    target_state: ManifoldLegStage2Target,
    config: ManifoldLegValidationConfig,
) -> ManifoldLegValidationReport:
    """Require Cyna clearance, local strike parity, and the smooth inner margin."""
    from pyna.topo.manifold_leg_clearance import validate_manifold_leg_clearance
    from pyna.topo.manifold_strike_correspondence import compare_jax_manifold_strike_bundle

    if not isinstance(target_state, ManifoldLegStage2Target):
        raise TypeError("target_state must be ManifoldLegStage2Target")
    if not isinstance(config, ManifoldLegValidationConfig):
        raise TypeError("config must be ManifoldLegValidationConfig")
    production = validate_manifold_leg_clearance(
        production_field, candidate_branch, config.wall,
        previous_matches=target_state.strike_matches,
        required_minimum_clearance_m=target_state.minimum_clearance_m,
        terminal_exclusion_fraction=target_state.terminal_exclusion_fraction,
        maximum_hit_displacement_m=config.maximum_hit_displacement_m,
        maximum_hit_phase_shift_rad=target_state.maximum_phi_shift,
        maximum_endpoint_error_m=config.maximum_endpoint_error_m,
        maximum_projection_distance_m=config.maximum_projection_distance_m,
        max_turns=config.max_turns, DPhi=config.production_DPhi,
        extend_phi=config.extend_phi,
        strike_trace_function=config.production_strike_trace_function,
        trajectory_trace_function=config.production_trajectory_trace_function,
    )
    if not production.accepted:
        return ManifoldLegValidationReport(
            production, None, None, None, False, production.rejection_reason,
        )
    refreshed = refresh_manifold_leg_stage2_target(
        target_state, candidate_branch, production,
    )
    inner = manifold_leg_clearance_state(
        candidate_field, refreshed.wall_signed_distance, target_state=refreshed,
        minimum_clearance_m=refreshed.minimum_clearance_m,
        clearance_scale_m=refreshed.clearance_scale_m,
        terminal_exclusion_fraction=refreshed.terminal_exclusion_fraction,
        sample_stride=refreshed.sample_stride,
    )
    correspondence = compare_jax_manifold_strike_bundle(
        refreshed.strike_matches, inner.trajectories.strike_bundle,
        absolute_tolerance_m=config.jax_cyna_tolerance_m,
    )
    if not correspondence.accepted:
        reason = f"leg_jax_cyna_correspondence:{correspondence.rejection_reason}"
    elif not np.all(np.isfinite(np.asarray(inner.signed_clearance_m))):
        reason = "inner_leg_clearance_nonfinite"
    elif np.min(np.asarray(inner.signed_clearance_m)) < refreshed.minimum_clearance_m:
        reason = "inner_leg_clearance_below_limit"
    else:
        reason = None
    return ManifoldLegValidationReport(
        production, correspondence, inner, refreshed if reason is None else None,
        reason is None, reason,
    )


__all__ = [
    "ManifoldLegValidationConfig", "ManifoldLegValidationReport",
    "validate_manifold_leg_candidate",
]
