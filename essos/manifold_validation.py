"""Outer validation of manifold-aware ESSOS continuation candidates.

The live ESSOS field is used by the differentiable JAX path.  A host-side
cylindrical snapshot of that same field is sent through PyNA's production Cyna
refresh.  PyNA owns every acceptance decision; this module only orchestrates
the package boundary and advances immutable ESSOS state after all gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Mapping, Sequence

import numpy as np

from essos.manifold import (
    essos_field_to_pyna_cylindrical_grid,
)
from essos.manifold_optimization import (
    ManifoldContinuationState,
    accept_manifold_continuation_stage,
    trace_manifold_reference,
)
from essos.manifold_strike_validation import (
    ManifoldStrikeValidationConfig,
    validate_manifold_strike_candidate,
)


def _pyna_validation_api():
    try:
        from pyna.topo.manifold_correspondence import (
            compare_jax_manifold_branch,
            refresh_manifold_sample_match,
        )
        from pyna.topo.manifold_refresh import (
            refresh_manifold_branch_reference_field,
        )
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS manifold candidate validation requires PyNA and Cyna"
        ) from exc
    return (
        refresh_manifold_branch_reference_field,
        compare_jax_manifold_branch,
        refresh_manifold_sample_match,
    )


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

    def __post_init__(self) -> None:
        accepted = bool(self.accepted)
        reason = None if self.rejection_reason is None else str(
            self.rejection_reason
        )
        if accepted:
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
) -> ManifoldContinuationValidationReport:
    """Validate a trial field and advance only after all PyNA gates pass.

    The input state is never mutated.  A rejected report has no
    ``accepted_state``, so the calling optimizer keeps its preceding field and
    may shorten the trial step.  Grid/configuration errors and unavailable
    backends raise rather than masquerading as a physical topology rejection.
    """

    if not isinstance(continuation_state, ManifoldContinuationState):
        raise TypeError("continuation_state must be ManifoldContinuationState")
    if continuation_state.strike_target_state is None:
        if strike_validation_config is not None:
            raise ValueError("strike_validation_config requires an active strike target")
    elif not isinstance(strike_validation_config, ManifoldStrikeValidationConfig):
        raise TypeError(
            "an active strike target requires ManifoldStrikeValidationConfig"
        )
    (
        production_refresh_function,
        compare_jax_manifold_branch,
        refresh_manifold_sample_match,
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
            )
        accepted_strike_target = strike_validation.accepted_target

    accepted_state = accept_manifold_continuation_stage(
        continuation_state,
        candidate_branch,
        sample_refresh,
        correspondence,
        accepted_strike_target=accepted_strike_target,
    )
    return ManifoldContinuationValidationReport(
        production_refresh=production_refresh,
        correspondence=correspondence,
        sample_refresh=sample_refresh,
        accepted_state=accepted_state,
        accepted=True,
        strike_validation=strike_validation,
        diagnostics={
            "anchor_displacement_m": production_refresh.anchor_displacement_m,
            "sample_displacement_m": sample_refresh.sample_displacement_m,
            "max_jax_cyna_deviation_m": correspondence.max_deviation_m,
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
    )


__all__ = [
    "ManifoldContinuationValidationReport",
    "validate_manifold_continuation_candidate",
]
