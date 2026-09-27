"""Production validation of topology-optimization candidates.

Independent pyna/Cyna refreshes of strike, leg-clearance and continuation
candidates, and the validated backtracking step that uses them.
"""
from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field, replace
from typing import Any, Callable, Mapping
from essos.topology_objectives import (
    ManifoldStrikeStage2Target,
    refresh_manifold_strike_stage2_target,
    trace_manifold_strike_reference,
)
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from essos.topology_objectives import (
    ManifoldLegStage2Target,
    manifold_leg_clearance_state,
    refresh_manifold_leg_stage2_target,
)
from dataclasses import dataclass, field as dataclass_field
from collections.abc import Callable
from typing import Any, Mapping, Sequence
from essos.topology import (
    essos_field_to_pyna_cylindrical_grid,
)
from essos.topology_objectives import (
    ManifoldContinuationState,
    accept_manifold_continuation_stage,
    trace_manifold_reference,
)
from essos.topology_objectives import (
    refresh_manifold_heat_stage2_target,
    trace_manifold_heat_reference,
)
import operator
from typing import Any
from essos.topology_objectives import ManifoldContinuationState


# ----------------------------------------------------------------------------
# From essos/manifold_strike_validation.py: Production refresh transaction for an ESSOS first-wall strike target.
# ----------------------------------------------------------------------------

def _pyna_strike_validation_api():
    try:
        from pyna.topo.manifold_strike import (
            local_wall_plane_from_strike,
            manifold_branch_strike_seed_bundle,
            refresh_manifold_strike_match,
        )
        from pyna.topo.manifold_strike_correspondence import (
            compare_jax_manifold_strike,
        )
        from pyna.toroidal.control.strike_heat import trace_wall_strikes_field
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS strike candidate validation requires PyNA and Cyna"
        ) from exc
    return (
        manifold_branch_strike_seed_bundle,
        trace_wall_strikes_field,
        refresh_manifold_strike_match,
        local_wall_plane_from_strike,
        compare_jax_manifold_strike,
    )


@dataclass(frozen=True)
class ManifoldStrikeValidationConfig:
    """Outer-loop wall trace and trust limits for one strike target."""

    wall: Any
    maximum_hit_displacement_m: float
    maximum_projection_distance_m: float
    jax_cyna_tolerance_m: float
    max_turns: int
    production_DPhi: float
    production_trace_function: Callable[..., Mapping[str, Any]] | None = None
    extend_phi: bool = True


@dataclass(frozen=True)
class ManifoldStrikeValidationReport:
    """Production, label-refresh, local-event, and parity gates for one hit."""

    production_bundle: Any
    production_strikes: Any
    strike_refresh: Any
    wall_plane: Any | None
    jax_strike_state: Any | None
    correspondence: Any | None
    accepted_target: ManifoldStrikeStage2Target | None
    accepted: bool
    rejection_reason: str | None = None
    diagnostics: Mapping[str, object] = dataclass_field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        accepted = bool(self.accepted)
        reason = None if self.rejection_reason is None else str(
            self.rejection_reason
        )
        if accepted:
            if (
                not bool(getattr(self.strike_refresh, "accepted", False))
                or self.wall_plane is None
                or self.jax_strike_state is None
                or self.correspondence is None
                or not bool(getattr(self.correspondence, "accepted", False))
                or self.accepted_target is None
                or reason is not None
            ):
                raise ValueError("an accepted strike validation requires every gate")
        else:
            if reason is None:
                raise ValueError("a rejected strike validation requires rejection_reason")
            if self.accepted_target is not None:
                raise ValueError("a rejected strike validation cannot accept a target")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejection_reason", reason)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))


def validate_manifold_strike_candidate(
    candidate_field: Any,
    production_field: Any,
    candidate_branch: Any,
    target_state: ManifoldStrikeStage2Target,
    wall: Any,
    *,
    maximum_hit_displacement_m: float,
    maximum_projection_distance_m: float,
    jax_cyna_tolerance_m: float,
    max_turns: int,
    production_DPhi: float,
    production_trace_function: Callable[..., Mapping[str, Any]] | None = None,
    extend_phi: bool = True,
) -> ManifoldStrikeValidationReport:
    """Refresh one exact Cyna strike and accept its local JAX model atomically.

    ``candidate_branch`` must already have passed the production branch refresh.
    The global Cyna wall trace decides whether the exact sparse seed still has a
    first hit.  Only then is a new local wall plane constructed and checked
    against the live ESSOS/JAX field.
    """

    if not isinstance(target_state, ManifoldStrikeStage2Target):
        raise TypeError("target_state must be ManifoldStrikeStage2Target")
    (
        make_bundle,
        trace_wall_strikes,
        refresh_match,
        make_wall_plane,
        compare_strike,
    ) = _pyna_strike_validation_api()

    production_bundle = make_bundle(candidate_branch)
    production_strikes = trace_wall_strikes(
        production_field,
        (production_bundle,),
        wall,
        max_turns=max_turns,
        DPhi=production_DPhi,
        extend_phi=extend_phi,
        trace_function=production_trace_function,
    )[0]
    strike_refresh = refresh_match(
        target_state.strike_match,
        production_bundle,
        production_strikes,
        target_state.target_position_m,
        maximum_hit_displacement_m=maximum_hit_displacement_m,
    )
    if not strike_refresh.accepted or strike_refresh.candidate_match is None:
        return ManifoldStrikeValidationReport(
            production_bundle=production_bundle,
            production_strikes=production_strikes,
            strike_refresh=strike_refresh,
            wall_plane=None,
            jax_strike_state=None,
            correspondence=None,
            accepted_target=None,
            accepted=False,
            rejection_reason=(
                "strike_refresh:"
                f"{strike_refresh.rejection_reason or 'unknown'}"
            ),
            diagnostics={
                "hit_displacement_m": strike_refresh.hit_displacement_m,
                "maximum_hit_displacement_m": (
                    strike_refresh.maximum_hit_displacement_m
                ),
            },
        )

    candidate_match = strike_refresh.candidate_match
    wall_plane = make_wall_plane(
        candidate_match,
        wall,
        maximum_projection_distance_m=maximum_projection_distance_m,
    )
    provisional_target = replace(
        target_state,
        branch_reference=candidate_branch,
        strike_match=candidate_match,
        wall_plane=wall_plane,
    )
    jax_strike_state = trace_manifold_strike_reference(
        candidate_field,
        provisional_target,
    )
    correspondence = compare_strike(
        candidate_match,
        jax_strike_state,
        absolute_tolerance_m=jax_cyna_tolerance_m,
    )
    if not correspondence.accepted:
        return ManifoldStrikeValidationReport(
            production_bundle=production_bundle,
            production_strikes=production_strikes,
            strike_refresh=strike_refresh,
            wall_plane=wall_plane,
            jax_strike_state=jax_strike_state,
            correspondence=correspondence,
            accepted_target=None,
            accepted=False,
            rejection_reason=(
                "jax_cyna_strike:"
                f"{correspondence.rejection_reason or 'unknown'}"
            ),
            diagnostics={
                "jax_cyna_strike_deviation_m": correspondence.deviation_m,
                "jax_cyna_strike_tolerance_m": (
                    correspondence.absolute_tolerance_m
                ),
                "wall_residual": correspondence.wall_residual,
                "transversality": correspondence.transversality,
                "phi_shift_from_guess": correspondence.phi_shift_from_guess,
            },
        )

    accepted_target = refresh_manifold_strike_stage2_target(
        target_state,
        candidate_branch,
        strike_refresh,
        correspondence,
        wall_plane,
    )
    return ManifoldStrikeValidationReport(
        production_bundle=production_bundle,
        production_strikes=production_strikes,
        strike_refresh=strike_refresh,
        wall_plane=wall_plane,
        jax_strike_state=jax_strike_state,
        correspondence=correspondence,
        accepted_target=accepted_target,
        accepted=True,
        diagnostics={
            "hit_displacement_m": strike_refresh.hit_displacement_m,
            "jax_cyna_strike_deviation_m": correspondence.deviation_m,
            "wall_residual": correspondence.wall_residual,
            "transversality": correspondence.transversality,
        },
    )


# ----------------------------------------------------------------------------
# From essos/manifold_leg_validation.py: Production and live-JAX acceptance gates for pre-strike clearance.
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# From essos/manifold_validation.py: Outer validation of manifold-aware ESSOS continuation candidates.
# ----------------------------------------------------------------------------

def _pyna_validation_api():
    try:
        from pyna.topo.manifold_correspondence import (
            compare_jax_manifold_branch,
            refresh_manifold_sample_match,
        )
        from pyna.topo.manifold_refresh import (
            refresh_manifold_branch_reference_field,
        )
        from pyna.topo.xline_clearance import (
            validate_periodic_xline_clearance,
        )
        from pyna.topo.manifold_heat import validate_manifold_heat_load
        from pyna.topo.manifold_strike_correspondence import (
            compare_jax_manifold_strike_bundle,
        )
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS manifold candidate validation requires PyNA and Cyna"
        ) from exc
    return (
        refresh_manifold_branch_reference_field,
        compare_jax_manifold_branch,
        refresh_manifold_sample_match,
        validate_periodic_xline_clearance,
        validate_manifold_heat_load,
        compare_jax_manifold_strike_bundle,
    )


@dataclass(frozen=True)
class XLineClearanceValidationConfig:
    """ESSOS orchestration settings for PyNA's production clearance gate."""

    wall: Any
    required_minimum_clearance_m: float
    maximum_closure_error_m: float
    production_DPhi: float = 0.01
    extend_phi: bool = True
    production_trace_function: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        for attribute in (
            "required_minimum_clearance_m",
            "maximum_closure_error_m",
        ):
            value = float(getattr(self, attribute))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{attribute} must be non-negative and finite")
            object.__setattr__(self, attribute, value)
        step = float(self.production_DPhi)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("production_DPhi must be positive and finite")
        if self.production_trace_function is not None and not callable(
            self.production_trace_function
        ):
            raise TypeError("production_trace_function must be callable or None")
        object.__setattr__(self, "production_DPhi", step)
        object.__setattr__(self, "extend_phi", bool(self.extend_phi))


@dataclass(frozen=True)
class ManifoldHeatValidationConfig:
    """ESSOS orchestration settings for PyNA's production heat-load gate."""

    wall: Any
    phi_edges: Sequence[float]
    s_edges: Sequence[float]
    maximum_hit_displacement_m: float
    jax_cyna_tolerance_m: float
    maximum_unresolved_power_W: float = 0.0
    maximum_projection_distance_m: float = 1.0e-3
    max_turns: int = 100
    production_DPhi: float = 0.01
    field_period: float | None = None
    extend_phi: bool = True
    production_trace_function: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        for attribute in (
            "maximum_hit_displacement_m",
            "maximum_unresolved_power_W",
            "maximum_projection_distance_m",
            "jax_cyna_tolerance_m",
        ):
            value = float(getattr(self, attribute))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{attribute} must be non-negative and finite")
            object.__setattr__(self, attribute, value)
        for attribute in ("phi_edges", "s_edges"):
            values = np.asarray(getattr(self, attribute), dtype=float).ravel()
            if (
                values.size < 2
                or not np.all(np.isfinite(values))
                or np.any(np.diff(values) <= 0.0)
            ):
                raise ValueError(
                    f"{attribute} must contain at least two finite increasing values"
                )
            object.__setattr__(self, attribute, values.copy())
        turns = int(self.max_turns)
        if turns != self.max_turns or turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        step = float(self.production_DPhi)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("production_DPhi must be positive and finite")
        if self.field_period is None:
            period = None
        else:
            period = float(self.field_period)
            if not np.isfinite(period) or period <= 0.0:
                raise ValueError("field_period must be positive and finite")
        if self.production_trace_function is not None and not callable(
            self.production_trace_function
        ):
            raise TypeError("production_trace_function must be callable or None")
        object.__setattr__(self, "max_turns", turns)
        object.__setattr__(self, "production_DPhi", step)
        object.__setattr__(self, "field_period", period)
        object.__setattr__(self, "extend_phi", bool(self.extend_phi))


@dataclass(frozen=True)
class ManifoldContinuationValidationReport:
    """All PyNA reports and optional next state for one trial coil field."""

    production_refresh: Any
    correspondence: Any | None
    sample_refresh: Any | None
    accepted_state: ManifoldContinuationState | None
    accepted: bool
    rejection_reason: str | None = None
    strike_validation: Any | None = None
    diagnostics: Mapping[str, object] = dataclass_field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    xline_clearance_validation: Any | None = None
    heat_validation: Any | None = None
    heat_strike_state: Any | None = None
    heat_correspondence: Any | None = None
    leg_validation: Any | None = None

    def __post_init__(self) -> None:
        accepted = bool(self.accepted)
        reason = None if self.rejection_reason is None else str(
            self.rejection_reason
        )
        if accepted:
            if (
                self.accepted_state is not None
                and self.accepted_state.leg_target_state is not None
            ):
                if (
                    self.leg_validation is None or not self.leg_validation.accepted
                    or self.leg_validation.accepted_target
                    is not self.accepted_state.leg_target_state
                ):
                    raise ValueError("accepted leg-aware validation requires its leg gate")
            if (
                self.production_refresh is None
                or not bool(getattr(self.production_refresh, "accepted", False))
                or self.correspondence is None
                or not bool(getattr(self.correspondence, "accepted", False))
                or self.sample_refresh is None
                or not bool(getattr(self.sample_refresh, "accepted", False))
                or self.accepted_state is None
                or reason is not None
            ):
                raise ValueError(
                    "an accepted validation requires all gates and a next state"
                )
            if (
                self.accepted_state.strike_target_state is not None
                and (
                    self.strike_validation is None
                    or not bool(getattr(self.strike_validation, "accepted", False))
                )
            ):
                raise ValueError(
                    "an accepted strike-aware validation requires its strike gate"
                )
            if (
                self.xline_clearance_validation is not None
                and not bool(
                    getattr(self.xline_clearance_validation, "accepted", False)
                )
            ):
                raise ValueError(
                    "an accepted validation requires its X-line clearance gate"
                )
            if (
                self.accepted_state.heat_target_state is not None
                and (
                    self.heat_validation is None
                    or not bool(getattr(self.heat_validation, "accepted", False))
                    or self.heat_strike_state is None
                    or self.heat_correspondence is None
                    or not bool(
                        getattr(self.heat_correspondence, "accepted", False)
                    )
                )
            ):
                raise ValueError(
                    "an accepted heat-aware validation requires its heat gate"
                )
        else:
            if reason is None:
                raise ValueError("a rejected validation requires rejection_reason")
            if self.accepted_state is not None:
                raise ValueError("a rejected validation cannot contain accepted_state")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejection_reason", reason)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    @property
    def candidate_branch(self):
        return getattr(self.production_refresh, "candidate_branch", None)


def validate_manifold_continuation_candidate(
    candidate_field: Any,
    continuation_state: ManifoldContinuationState,
    R_grid_m: Any,
    Z_grid_m: Any,
    Phi_grid_rad: Any,
    *,
    maximum_anchor_displacement_m: float,
    minimum_direction_alignment: float,
    maximum_sample_displacement_m: float,
    jax_cyna_tolerance_m: float,
    nfp: int = 1,
    sampling_batch_size: int = 4096,
    axisymmetric: bool = False,
    production_DPhi: float = 0.01,
    production_fd_eps: float = 1.0e-4,
    fixed_point_max_iter: int = 20,
    fixed_point_tolerance: float = 1.0e-10,
    fixed_point_residual_tolerance: float | None = None,
    seed_geometry_tolerance_m: float = 1.0e-10,
    require_complete_correspondence: bool = False,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    wall_phi: Sequence[float] | None = None,
    wall_R_all: np.ndarray | None = None,
    wall_Z_all: np.ndarray | None = None,
    RZlimit: tuple[float, float, float, float] | None = None,
    refine_stable_inverse_anchor: bool | None = None,
    extend_phi: bool = True,
    n_threads: int = -1,
    strike_validation_config: ManifoldStrikeValidationConfig | None = None,
    xline_clearance_config: XLineClearanceValidationConfig | None = None,
    heat_validation_config: ManifoldHeatValidationConfig | None = None,
    leg_validation_config: ManifoldLegValidationConfig | None = None,
) -> ManifoldContinuationValidationReport:
    """Validate a trial field and advance only after all PyNA gates pass.

    The input state is never mutated.  A rejected report has no
    ``accepted_state``, so the calling optimizer keeps its preceding field and
    may shorten the trial step.  Grid/configuration errors and unavailable
    backends raise rather than masquerading as a physical topology rejection.
    """

    if not isinstance(continuation_state, ManifoldContinuationState):
        raise TypeError("continuation_state must be ManifoldContinuationState")
    if (
        continuation_state.stage.leg_clearance_weight > 0
        and continuation_state.leg_target_state is None
    ):
        raise ValueError("an active leg clearance term requires leg_target_state")
    if continuation_state.leg_target_state is None:
        if leg_validation_config is not None:
            raise ValueError("leg_validation_config requires an active leg target")
    elif not isinstance(leg_validation_config, ManifoldLegValidationConfig):
        raise TypeError("an active leg target requires ManifoldLegValidationConfig")
    if continuation_state.strike_target_state is None:
        if strike_validation_config is not None:
            raise ValueError("strike_validation_config requires an active strike target")
    elif not isinstance(strike_validation_config, ManifoldStrikeValidationConfig):
        raise TypeError(
            "an active strike target requires ManifoldStrikeValidationConfig"
        )
    if continuation_state.stage.heat_weight > 0.0 and (
        continuation_state.heat_target_state is None
    ):
        raise ValueError("an active heat term requires heat_target_state")
    if continuation_state.heat_target_state is None:
        if heat_validation_config is not None:
            raise ValueError(
                "heat_validation_config requires an active heat target"
            )
    elif not isinstance(heat_validation_config, ManifoldHeatValidationConfig):
        raise TypeError(
            "an active heat target requires ManifoldHeatValidationConfig"
        )
    clearance_is_active = (
        continuation_state.stage.xline_clearance_weight > 0.0
    )
    if clearance_is_active and not isinstance(
        xline_clearance_config,
        XLineClearanceValidationConfig,
    ):
        raise TypeError(
            "an active X-line clearance term requires "
            "XLineClearanceValidationConfig"
        )
    if xline_clearance_config is not None and not isinstance(
        xline_clearance_config,
        XLineClearanceValidationConfig,
    ):
        raise TypeError(
            "xline_clearance_config must be "
            "XLineClearanceValidationConfig or None"
        )
    (
        production_refresh_function,
        compare_jax_manifold_branch,
        refresh_manifold_sample_match,
        validate_periodic_xline_clearance,
        validate_manifold_heat_load,
        compare_jax_manifold_strike_bundle,
    ) = _pyna_validation_api()

    production_field = essos_field_to_pyna_cylindrical_grid(
        candidate_field,
        R_grid_m,
        Z_grid_m,
        Phi_grid_rad,
        nfp=nfp,
        batch_size=sampling_batch_size,
        axisymmetric=axisymmetric,
    )
    target_state = continuation_state.target_state
    production_refresh = production_refresh_function(
        production_field,
        target_state.branch_reference,
        maximum_anchor_displacement_m=maximum_anchor_displacement_m,
        minimum_direction_alignment=minimum_direction_alignment,
        DPhi=production_DPhi,
        fd_eps=production_fd_eps,
        fixed_point_max_iter=fixed_point_max_iter,
        fixed_point_tolerance=fixed_point_tolerance,
        fixed_point_residual_tolerance=fixed_point_residual_tolerance,
        seed_geometry_tolerance_m=seed_geometry_tolerance_m,
        wall_R=wall_R,
        wall_Z=wall_Z,
        wall_phi=wall_phi,
        wall_R_all=wall_R_all,
        wall_Z_all=wall_Z_all,
        RZlimit=RZlimit,
        refine_stable_inverse_anchor=refine_stable_inverse_anchor,
        extend_phi=extend_phi,
        n_threads=n_threads,
    )
    if not production_refresh.accepted:
        return ManifoldContinuationValidationReport(
            production_refresh=production_refresh,
            correspondence=None,
            sample_refresh=None,
            accepted_state=None,
            accepted=False,
            rejection_reason=(
                "production_refresh:"
                f"{production_refresh.rejection_reason or 'unknown'}"
            ),
            diagnostics=production_refresh.diagnostics,
        )

    candidate_branch = production_refresh.candidate_branch
    xline_clearance_validation = None
    if xline_clearance_config is not None:
        config = xline_clearance_config
        xline_clearance_validation = validate_periodic_xline_clearance(
            production_field,
            candidate_branch,
            config.wall,
            DPhi=config.production_DPhi,
            required_minimum_clearance_m=(
                config.required_minimum_clearance_m
            ),
            maximum_closure_error_m=config.maximum_closure_error_m,
            extend_phi=config.extend_phi,
            trace_function=config.production_trace_function,
        )
        if not xline_clearance_validation.accepted:
            return ManifoldContinuationValidationReport(
                production_refresh=production_refresh,
                correspondence=None,
                sample_refresh=None,
                accepted_state=None,
                accepted=False,
                rejection_reason=(
                    "xline_clearance_validation:"
                    f"{xline_clearance_validation.rejection_reason or 'unknown'}"
                ),
                diagnostics={
                    "minimum_xline_clearance_m": (
                        xline_clearance_validation.minimum_clearance_m
                    ),
                    "required_minimum_xline_clearance_m": (
                        xline_clearance_validation.required_minimum_clearance_m
                    ),
                    "production_xline_closure_error_m": (
                        xline_clearance_validation.closure_error_m
                    ),
                },
                xline_clearance_validation=xline_clearance_validation,
            )
    heat_validation = None
    accepted_heat_target = None
    heat_strike_state = None
    heat_correspondence = None
    if continuation_state.heat_target_state is not None:
        config = heat_validation_config
        heat_target = continuation_state.heat_target_state
        heat_validation = validate_manifold_heat_load(
            production_field,
            candidate_branch,
            config.wall,
            strike_powers_W=heat_target.strike_powers_W,
            power_provenance=heat_target.power_provenance,
            phi_edges=config.phi_edges,
            s_edges=config.s_edges,
            max_turns=config.max_turns,
            DPhi=config.production_DPhi,
            maximum_heat_flux_W_m2=(
                heat_target.maximum_heat_flux_W_m2
            ),
            maximum_unresolved_power_W=(
                config.maximum_unresolved_power_W
            ),
            maximum_projection_distance_m=(
                config.maximum_projection_distance_m
            ),
            previous_matches=heat_target.strike_matches,
            maximum_hit_displacement_m=(
                config.maximum_hit_displacement_m
            ),
            field_period=config.field_period,
            extend_phi=config.extend_phi,
            trace_function=config.production_trace_function,
        )
        if not heat_validation.accepted:
            return ManifoldContinuationValidationReport(
                production_refresh=production_refresh,
                correspondence=None,
                sample_refresh=None,
                accepted_state=None,
                accepted=False,
                rejection_reason=(
                    "heat_validation:"
                    f"{heat_validation.rejection_reason or 'unknown'}"
                ),
                diagnostics={
                    "production_peak_heat_flux_W_m2": (
                        heat_validation.peak_heat_flux_W_m2
                    ),
                    "maximum_heat_flux_W_m2": (
                        heat_validation.maximum_heat_flux_W_m2
                    ),
                    "production_unresolved_power_W": (
                        heat_validation.unresolved_power_W
                    ),
                },
                xline_clearance_validation=xline_clearance_validation,
                heat_validation=heat_validation,
            )
        accepted_heat_target = refresh_manifold_heat_stage2_target(
            heat_target,
            candidate_branch,
            heat_validation,
        )
        heat_strike_state = trace_manifold_heat_reference(
            candidate_field,
            accepted_heat_target,
        )
        heat_correspondence = compare_jax_manifold_strike_bundle(
            accepted_heat_target.strike_matches,
            heat_strike_state,
            absolute_tolerance_m=config.jax_cyna_tolerance_m,
        )
        if not heat_correspondence.accepted:
            return ManifoldContinuationValidationReport(
                production_refresh=production_refresh,
                correspondence=None,
                sample_refresh=None,
                accepted_state=None,
                accepted=False,
                rejection_reason=(
                    "heat_jax_cyna_correspondence:"
                    f"{heat_correspondence.rejection_reason or 'unknown'}"
                ),
                diagnostics={
                    "production_peak_heat_flux_W_m2": (
                        heat_validation.peak_heat_flux_W_m2
                    ),
                    "production_unresolved_power_W": (
                        heat_validation.unresolved_power_W
                    ),
                    "max_jax_cyna_heat_strike_deviation_m": (
                        heat_correspondence.max_deviation_m
                    ),
                    "jax_cyna_heat_strike_tolerance_m": (
                        heat_correspondence.absolute_tolerance_m
                    ),
                },
                xline_clearance_validation=xline_clearance_validation,
                heat_validation=heat_validation,
                heat_strike_state=heat_strike_state,
                heat_correspondence=heat_correspondence,
            )
    jax_trace = trace_manifold_reference(
        candidate_field,
        candidate_branch,
        n_steps_per_span=target_state.n_steps_per_span,
        newton_iterations=target_state.newton_iterations,
        newton_damping=target_state.newton_damping,
        bphi_floor=target_state.bphi_floor,
    )
    correspondence = compare_jax_manifold_branch(
        candidate_branch,
        jax_trace.generations,
        absolute_tolerance_m=jax_cyna_tolerance_m,
        require_complete=require_complete_correspondence,
    )
    sample_refresh = refresh_manifold_sample_match(
        target_state.sample_match,
        candidate_branch,
        target_state.target_RZ_m,
        maximum_sample_displacement_m=maximum_sample_displacement_m,
    )

    if not correspondence.accepted:
        return ManifoldContinuationValidationReport(
            production_refresh=production_refresh,
            correspondence=correspondence,
            sample_refresh=sample_refresh,
            accepted_state=None,
            accepted=False,
            rejection_reason="jax_cyna_correspondence_failed",
            diagnostics={
                "max_jax_cyna_deviation_m": correspondence.max_deviation_m,
                "jax_cyna_tolerance_m": correspondence.absolute_tolerance_m,
                "production_branch_complete": correspondence.complete,
            },
            xline_clearance_validation=xline_clearance_validation,
            heat_validation=heat_validation,
            heat_strike_state=heat_strike_state,
            heat_correspondence=heat_correspondence,
        )
    if not sample_refresh.accepted:
        return ManifoldContinuationValidationReport(
            production_refresh=production_refresh,
            correspondence=correspondence,
            sample_refresh=sample_refresh,
            accepted_state=None,
            accepted=False,
            rejection_reason=(
                "sample_refresh:"
                f"{sample_refresh.rejection_reason or 'unknown'}"
            ),
            diagnostics={
                "sample_displacement_m": sample_refresh.sample_displacement_m,
                "maximum_sample_displacement_m": (
                    sample_refresh.maximum_sample_displacement_m
                ),
            },
            xline_clearance_validation=xline_clearance_validation,
            heat_validation=heat_validation,
            heat_strike_state=heat_strike_state,
            heat_correspondence=heat_correspondence,
        )

    strike_validation = None
    accepted_strike_target = None
    if continuation_state.strike_target_state is not None:
        config = strike_validation_config
        strike_validation = validate_manifold_strike_candidate(
            candidate_field,
            production_field,
            candidate_branch,
            continuation_state.strike_target_state,
            config.wall,
            maximum_hit_displacement_m=config.maximum_hit_displacement_m,
            maximum_projection_distance_m=(
                config.maximum_projection_distance_m
            ),
            jax_cyna_tolerance_m=config.jax_cyna_tolerance_m,
            max_turns=config.max_turns,
            production_DPhi=config.production_DPhi,
            production_trace_function=config.production_trace_function,
            extend_phi=config.extend_phi,
        )
        if not strike_validation.accepted:
            return ManifoldContinuationValidationReport(
                production_refresh=production_refresh,
                correspondence=correspondence,
                sample_refresh=sample_refresh,
                accepted_state=None,
                accepted=False,
                rejection_reason=(
                    "strike_validation:"
                    f"{strike_validation.rejection_reason or 'unknown'}"
                ),
                strike_validation=strike_validation,
                diagnostics=strike_validation.diagnostics,
                xline_clearance_validation=xline_clearance_validation,
                heat_validation=heat_validation,
                heat_strike_state=heat_strike_state,
                heat_correspondence=heat_correspondence,
            )
        accepted_strike_target = strike_validation.accepted_target

    leg_validation = None
    accepted_leg_target = None
    if continuation_state.leg_target_state is not None:
        leg_validation = validate_manifold_leg_candidate(
            candidate_field, production_field, candidate_branch,
            continuation_state.leg_target_state, leg_validation_config,
        )
        if not leg_validation.accepted:
            return ManifoldContinuationValidationReport(
                production_refresh=production_refresh,
                correspondence=correspondence, sample_refresh=sample_refresh,
                accepted_state=None, accepted=False,
                rejection_reason=f"leg_validation:{leg_validation.rejection_reason}",
                diagnostics={
                    "minimum_leg_clearance_m": (
                        leg_validation.production.minimum_clearance_m
                    ),
                    "required_minimum_leg_clearance_m": (
                        continuation_state.leg_target_state.minimum_clearance_m
                    ),
                },
                strike_validation=strike_validation,
                xline_clearance_validation=xline_clearance_validation,
                heat_validation=heat_validation, heat_strike_state=heat_strike_state,
                heat_correspondence=heat_correspondence, leg_validation=leg_validation,
            )
        accepted_leg_target = leg_validation.accepted_target

    accepted_state = accept_manifold_continuation_stage(
        continuation_state,
        candidate_branch,
        sample_refresh,
        correspondence,
        accepted_strike_target=accepted_strike_target,
        accepted_heat_target=accepted_heat_target,
        accepted_leg_target=accepted_leg_target,
    )
    return ManifoldContinuationValidationReport(
        production_refresh=production_refresh,
        correspondence=correspondence,
        sample_refresh=sample_refresh,
        accepted_state=accepted_state,
        accepted=True,
        strike_validation=strike_validation,
        leg_validation=leg_validation,
        diagnostics={
            "anchor_displacement_m": production_refresh.anchor_displacement_m,
            "sample_displacement_m": sample_refresh.sample_displacement_m,
            "max_jax_cyna_deviation_m": correspondence.max_deviation_m,
            **(
                {} if leg_validation is None else {
                    "minimum_leg_clearance_m": (
                        leg_validation.production.minimum_clearance_m
                    ),
                    "minimum_inner_leg_clearance_m": float(np.min(
                        np.asarray(leg_validation.inner_state.signed_clearance_m)
                    )),
                }
            ),
            **(
                {}
                if xline_clearance_validation is None
                else {
                    "minimum_xline_clearance_m": (
                        xline_clearance_validation.minimum_clearance_m
                    ),
                    "production_xline_closure_error_m": (
                        xline_clearance_validation.closure_error_m
                    ),
                }
            ),
            **(
                {}
                if heat_validation is None
                else {
                    "production_peak_heat_flux_W_m2": (
                        heat_validation.peak_heat_flux_W_m2
                    ),
                    "production_unresolved_power_W": (
                        heat_validation.unresolved_power_W
                    ),
                    "max_jax_cyna_heat_strike_deviation_m": (
                        heat_correspondence.max_deviation_m
                    ),
                }
            ),
            **(
                {}
                if strike_validation is None
                else {
                    "strike_displacement_m": (
                        strike_validation.strike_refresh.hit_displacement_m
                    ),
                    "jax_cyna_strike_deviation_m": (
                        strike_validation.correspondence.deviation_m
                    ),
                }
            ),
        },
        xline_clearance_validation=xline_clearance_validation,
        heat_validation=heat_validation,
        heat_strike_state=heat_strike_state,
        heat_correspondence=heat_correspondence,
    )


# ----------------------------------------------------------------------------
# From essos/manifold_driver.py: Optimizer-facing rollback for manifold-aware Stage-2 trial steps.
# ----------------------------------------------------------------------------

def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _flat_finite_dofs(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return array.copy()


@dataclass(frozen=True)
class ManifoldBacktrackingAttempt:
    """One contracted candidate and the topology report it produced."""

    step_fraction: float
    candidate_dofs: np.ndarray
    validation: ManifoldContinuationValidationReport

    def __post_init__(self) -> None:
        fraction = float(self.step_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("step_fraction must lie in (0, 1]")
        dofs = _flat_finite_dofs(self.candidate_dofs, "candidate_dofs")
        if not isinstance(self.validation, ManifoldContinuationValidationReport):
            raise TypeError("validation must be ManifoldContinuationValidationReport")
        object.__setattr__(self, "step_fraction", fraction)
        object.__setattr__(self, "candidate_dofs", dofs)


@dataclass(frozen=True)
class ManifoldBacktrackingResult:
    """Accepted candidate, or the unchanged field/state after all rejections."""

    accepted: bool
    dofs: np.ndarray
    field: Any
    continuation_state: ManifoldContinuationState
    step_fraction: float
    attempts: tuple[ManifoldBacktrackingAttempt, ...]

    def __post_init__(self) -> None:
        accepted = bool(self.accepted)
        dofs = _flat_finite_dofs(self.dofs, "dofs")
        fraction = float(self.step_fraction)
        attempts = tuple(self.attempts)
        if not isinstance(self.continuation_state, ManifoldContinuationState):
            raise TypeError("continuation_state must be ManifoldContinuationState")
        if not all(isinstance(item, ManifoldBacktrackingAttempt) for item in attempts):
            raise TypeError("attempts must contain ManifoldBacktrackingAttempt objects")
        if accepted:
            if not attempts or not attempts[-1].validation.accepted:
                raise ValueError("accepted result requires an accepted final attempt")
            if fraction != attempts[-1].step_fraction:
                raise ValueError("accepted step fraction must match the final attempt")
            if attempts[-1].validation.accepted_state is not self.continuation_state:
                raise ValueError("accepted result must use the validated continuation state")
        elif fraction != 0.0:
            raise ValueError("a rejected result must have zero step_fraction")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "step_fraction", fraction)
        object.__setattr__(self, "attempts", attempts)


def validated_manifold_backtracking_step(
    current_dofs: object,
    proposed_dofs: object,
    field_from_dofs: Callable[[np.ndarray], Any],
    continuation_state: ManifoldContinuationState,
    validate_candidate: Callable[
        [Any, ManifoldContinuationState],
        ManifoldContinuationValidationReport,
    ],
    *,
    contraction: float = 0.5,
    maximum_attempts: int = 8,
    minimum_step_fraction: float = 0.0,
) -> ManifoldBacktrackingResult:
    """Contract a flat optimizer proposal until topology validation accepts.

    Every attempt starts from ``current_dofs`` and the same accepted topology
    state.  A validation exception propagates because it denotes a backend or
    configuration problem; an ordinary physical rejection triggers the next
    contraction.  If no candidate accepts, both DOFs and continuation state
    remain at their input values.
    """

    current = _flat_finite_dofs(current_dofs, "current_dofs")
    proposed = _flat_finite_dofs(proposed_dofs, "proposed_dofs")
    if current.shape != proposed.shape:
        raise ValueError("current_dofs and proposed_dofs must have matching shapes")
    if not callable(field_from_dofs):
        raise TypeError("field_from_dofs must be callable")
    if not callable(validate_candidate):
        raise TypeError("validate_candidate must be callable")
    if not isinstance(continuation_state, ManifoldContinuationState):
        raise TypeError("continuation_state must be ManifoldContinuationState")
    contraction_value = float(contraction)
    if not np.isfinite(contraction_value) or not 0.0 < contraction_value < 1.0:
        raise ValueError("contraction must lie in (0, 1)")
    attempts_limit = _positive_integer(maximum_attempts, "maximum_attempts")
    minimum_fraction = float(minimum_step_fraction)
    if (
        not np.isfinite(minimum_fraction)
        or not 0.0 <= minimum_fraction <= 1.0
    ):
        raise ValueError("minimum_step_fraction must lie in [0, 1]")

    direction = proposed - current
    fraction = 1.0
    attempts: list[ManifoldBacktrackingAttempt] = []
    for _ in range(attempts_limit):
        if fraction < minimum_fraction:
            break
        candidate_dofs = current + fraction * direction
        candidate_field = field_from_dofs(candidate_dofs.copy())
        validation = validate_candidate(candidate_field, continuation_state)
        if not isinstance(validation, ManifoldContinuationValidationReport):
            raise TypeError(
                "validate_candidate must return "
                "ManifoldContinuationValidationReport"
            )
        attempt = ManifoldBacktrackingAttempt(
            step_fraction=fraction,
            candidate_dofs=candidate_dofs,
            validation=validation,
        )
        attempts.append(attempt)
        if validation.accepted:
            return ManifoldBacktrackingResult(
                accepted=True,
                dofs=candidate_dofs,
                field=candidate_field,
                continuation_state=validation.accepted_state,
                step_fraction=fraction,
                attempts=tuple(attempts),
            )
        fraction *= contraction_value

    return ManifoldBacktrackingResult(
        accepted=False,
        dofs=current,
        field=field_from_dofs(current.copy()),
        continuation_state=continuation_state,
        step_fraction=0.0,
        attempts=tuple(attempts),
    )
