"""Production refresh transaction for an ESSOS first-wall strike target."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from typing import Any, Callable, Mapping

from essos.manifold_strike_optimization import (
    ManifoldStrikeStage2Target,
    refresh_manifold_strike_stage2_target,
    trace_manifold_strike_reference,
)


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


__all__ = [
    "ManifoldStrikeValidationConfig",
    "ManifoldStrikeValidationReport",
    "validate_manifold_strike_candidate",
]
