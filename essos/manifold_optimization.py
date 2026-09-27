"""Stage-2 optimizer state for PyNA-owned invariant-manifold targets.

PyNA discovers and validates topology.  ESSOS stores one accepted, immutable
snapshot of that outer state while differentiating coil objectives.  This
module turns a ``ManifoldBranchReference`` and exact ``ManifoldSampleMatch``
into a JAX loss without re-inferring branch identity inside the optimizer.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from essos.losses import base_loss, composite_loss, custom_loss
from essos.manifold import trace_manifold_branch
from essos.manifold_leg_optimization import (
    ManifoldLegStage2Target,
    make_manifold_leg_stage2_loss,
    require_same_leg_configuration,
)
from essos.manifold_heat_optimization import (
    ManifoldHeatStage2Target,
    make_manifold_heat_stage2_loss,
)
from essos.manifold_strike_optimization import (
    ManifoldStrikeStage2Target,
    make_manifold_strike_stage2_loss,
)


def _pyna_correspondence_api():
    try:
        from pyna.topo.manifold_correspondence import (
            ManifoldBranchReference,
            ManifoldCorrespondenceReport,
            ManifoldSampleMatch,
            ManifoldSampleRefreshReport,
        )
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS manifold optimization requires PyNA's manifold "
            "correspondence backend"
        ) from exc
    return (
        ManifoldBranchReference,
        ManifoldCorrespondenceReport,
        ManifoldSampleMatch,
        ManifoldSampleRefreshReport,
    )


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


def _finite_rz(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).ravel()
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite physical (R, Z) point")
    return array


def _continuation_weight(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _continuation_index(value: object, n_stages: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("stage_index must be an integer")
    try:
        index = operator.index(value)
    except TypeError as exc:
        raise TypeError("stage_index must be an integer") from exc
    if not 0 <= index < n_stages:
        raise IndexError("stage_index is outside the continuation schedule")
    return index


@dataclass(frozen=True)
class ManifoldContinuationStage:
    """Topology weights held fixed during one inner Stage-2 solve."""

    name: str
    return_map_weight: float = 0.0
    xline_weight: float = 0.0
    manifold_weight: float = 0.0
    strike_weight: float = 0.0
    xline_clearance_weight: float = 0.0
    heat_weight: float = 0.0
    leg_clearance_weight: float = 0.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("continuation stage name must not be empty")
        object.__setattr__(self, "name", name)
        for attribute in (
            "return_map_weight",
            "xline_weight",
            "xline_clearance_weight",
            "manifold_weight",
            "strike_weight",
            "heat_weight",
            "leg_clearance_weight",
        ):
            object.__setattr__(
                self,
                attribute,
                _continuation_weight(getattr(self, attribute), attribute),
            )

    @property
    def topology_active(self) -> bool:
        return any(
            weight > 0.0
            for weight in (
                self.return_map_weight,
                self.xline_weight,
                self.xline_clearance_weight,
                self.manifold_weight,
                self.strike_weight,
                self.heat_weight,
                self.leg_clearance_weight,
            )
        )


@dataclass(frozen=True)
class ManifoldContinuationSchedule:
    """Ordered, user-scaled topology ramp for Stage-2 coil optimization."""

    stages: Sequence[ManifoldContinuationStage]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("continuation schedule must contain at least one stage")
        if not all(isinstance(stage, ManifoldContinuationStage) for stage in stages):
            raise TypeError("every continuation stage must be ManifoldContinuationStage")
        names = tuple(stage.name for stage in stages)
        if len(set(names)) != len(names):
            raise ValueError("continuation stage names must be unique")
        object.__setattr__(self, "stages", stages)

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, index: int) -> ManifoldContinuationStage:
        return self.stages[index]


@dataclass(frozen=True)
class ManifoldContinuationState:
    """Accepted target snapshot and the currently active ramp stage."""

    schedule: ManifoldContinuationSchedule
    target_state: ManifoldStage2Target
    stage_index: int = 0
    strike_target_state: ManifoldStrikeStage2Target | None = None
    heat_target_state: ManifoldHeatStage2Target | None = None
    leg_target_state: ManifoldLegStage2Target | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, ManifoldContinuationSchedule):
            raise TypeError("schedule must be ManifoldContinuationSchedule")
        if not isinstance(self.target_state, ManifoldStage2Target):
            raise TypeError("target_state must be ManifoldStage2Target")
        if self.leg_target_state is not None:
            if not isinstance(self.leg_target_state, ManifoldLegStage2Target):
                raise TypeError("leg_target_state must be ManifoldLegStage2Target or None")
            if self.leg_target_state.branch_reference is not self.target_state.branch_reference:
                raise ValueError("leg target must share the active branch snapshot")
        if self.strike_target_state is not None and not isinstance(
            self.strike_target_state,
            ManifoldStrikeStage2Target,
        ):
            raise TypeError(
                "strike_target_state must be ManifoldStrikeStage2Target or None"
            )
        if self.heat_target_state is not None and not isinstance(
            self.heat_target_state,
            ManifoldHeatStage2Target,
        ):
            raise TypeError(
                "heat_target_state must be ManifoldHeatStage2Target or None"
            )
        object.__setattr__(
            self,
            "stage_index",
            _continuation_index(self.stage_index, len(self.schedule)),
        )

    @property
    def stage(self) -> ManifoldContinuationStage:
        return self.schedule[self.stage_index]

    @property
    def final_stage(self) -> bool:
        return self.stage_index == len(self.schedule) - 1


@dataclass(frozen=True)
class ManifoldStage2Target:
    """Frozen topology and target geometry used during one inner solve."""

    branch_reference: Any
    sample_match: Any
    target_RZ_m: np.ndarray
    rz_scales_m: np.ndarray
    n_steps_per_span: int = 256
    newton_iterations: int = 8
    newton_damping: float = 1.0
    bphi_floor: float = 0.0
    derivative_mode: str = "frozen_reference"

    def __post_init__(self) -> None:
        (
            ManifoldBranchReference,
            _ManifoldCorrespondenceReport,
            ManifoldSampleMatch,
            _ManifoldSampleRefreshReport,
        ) = _pyna_correspondence_api()
        if not isinstance(self.branch_reference, ManifoldBranchReference):
            raise TypeError("branch_reference must be a PyNA ManifoldBranchReference")
        if not isinstance(self.sample_match, ManifoldSampleMatch):
            raise TypeError("sample_match must be a PyNA ManifoldSampleMatch")

        reference_point = self.branch_reference.point(self.sample_match.label)
        reference_seed_index = self.branch_reference.seed_index(
            self.sample_match.label.seed_order
        )
        if reference_seed_index != self.sample_match.jax_seed_index:
            raise ValueError("sample_match JAX seed index disagrees with branch_reference")
        if not np.allclose(
            reference_point,
            self.sample_match.point_RZ_m,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError("sample_match point disagrees with branch_reference")

        if self.derivative_mode not in ("frozen_reference", "moving_linear_seed"):
            raise ValueError("Unsupported derivative_mode")
        from pyna.topo.snapshot import immutable_array
        target = _finite_rz(self.target_RZ_m, "target_RZ_m")
        scales = _finite_rz(self.rz_scales_m, "rz_scales_m")
        if np.any(scales <= 0.0):
            raise ValueError("rz_scales_m must be positive")
        steps = _positive_integer(self.n_steps_per_span, "n_steps_per_span")
        iterations = _positive_integer(self.newton_iterations, "newton_iterations")
        damping = float(self.newton_damping)
        if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
            raise ValueError("newton_damping must lie in (0, 1]")
        bphi_floor = float(self.bphi_floor)
        if not np.isfinite(bphi_floor) or bphi_floor < 0.0:
            raise ValueError("bphi_floor must be non-negative and finite")

        object.__setattr__(self, "target_RZ_m", immutable_array(target))
        object.__setattr__(self, "rz_scales_m", immutable_array(scales))
        object.__setattr__(self, "n_steps_per_span", steps)
        object.__setattr__(self, "newton_iterations", iterations)
        object.__setattr__(self, "newton_damping", damping)
        object.__setattr__(self, "bphi_floor", bphi_floor)

    @property
    def label(self) -> str:
        return self.sample_match.label.key


def trace_manifold_reference(
    field: Any,
    branch_reference: Any,
    *,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
    derivative_mode: str = "frozen_reference",
):
    """Trace JAX generations for one accepted PyNA production reference."""

    ManifoldBranchReference, _, _, _ = _pyna_correspondence_api()
    if not isinstance(branch_reference, ManifoldBranchReference):
        raise TypeError("branch_reference must be a PyNA ManifoldBranchReference")
    return trace_manifold_branch(
        field,
        branch_reference.anchor_RZ_m,
        branch_reference.direction_RZ,
        branch_reference.seed_distances_m,
        stability=branch_reference.stability,
        side=branch_reference.seed_side,
        phi_span=branch_reference.map_span,
        phi_start=branch_reference.section_phi,
        map_power=1,
        n_generations=branch_reference.n_generations,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
        derivative_mode=derivative_mode,
    )


def manifold_stage2_target_loss(
    field: Any,
    *,
    target_state: ManifoldStage2Target,
):
    """Normalized location loss for one immutable manifold sample label."""

    if not isinstance(target_state, ManifoldStage2Target):
        raise TypeError("target_state must be ManifoldStage2Target")
    trace = trace_manifold_reference(
        field,
        target_state.branch_reference,
        n_steps_per_span=target_state.n_steps_per_span,
        newton_iterations=target_state.newton_iterations,
        newton_damping=target_state.newton_damping,
        bphi_floor=target_state.bphi_floor,
        derivative_mode=target_state.derivative_mode,
    )
    label = target_state.sample_match.label
    sample = trace.generations[
        label.generation_index,
        target_state.sample_match.jax_seed_index,
    ]
    residual = (sample - target_state.target_RZ_m) / target_state.rz_scales_m
    return 0.5 * (residual[0] ** 2 + residual[1] ** 2)


def make_manifold_stage2_loss(
    target_state: ManifoldStage2Target,
    *,
    field_dependency: str = "field",
) -> custom_loss:
    """Build an ESSOS ``custom_loss`` for one accepted manifold target."""

    if not isinstance(target_state, ManifoldStage2Target):
        raise TypeError("target_state must be ManifoldStage2Target")
    dependency = str(field_dependency).strip()
    if not dependency:
        raise ValueError("field_dependency must not be empty")
    return custom_loss(
        manifold_stage2_target_loss,
        dependency,
        target_state=target_state,
    )


def refresh_manifold_stage2_target(
    target_state: ManifoldStage2Target,
    candidate_branch: Any,
    sample_refresh: Any,
    correspondence: Any,
) -> ManifoldStage2Target:
    """Accept PyNA-validated outer state without relabelling in ESSOS."""

    if not isinstance(target_state, ManifoldStage2Target):
        raise TypeError("target_state must be ManifoldStage2Target")
    (
        ManifoldBranchReference,
        ManifoldCorrespondenceReport,
        _ManifoldSampleMatch,
        ManifoldSampleRefreshReport,
    ) = _pyna_correspondence_api()
    if not isinstance(candidate_branch, ManifoldBranchReference):
        raise TypeError("candidate_branch must be a PyNA ManifoldBranchReference")
    if not isinstance(sample_refresh, ManifoldSampleRefreshReport):
        raise TypeError("sample_refresh must be a PyNA ManifoldSampleRefreshReport")
    if not isinstance(correspondence, ManifoldCorrespondenceReport):
        raise TypeError("correspondence must be a PyNA ManifoldCorrespondenceReport")
    if not sample_refresh.accepted or sample_refresh.candidate_match is None:
        reason = sample_refresh.rejection_reason or "unknown"
        raise ValueError(f"PyNA rejected manifold sample refresh: {reason}")
    if not correspondence.accepted:
        raise ValueError("PyNA rejected JAX/Cyna manifold correspondence")
    if correspondence.branch_label != candidate_branch.label:
        raise ValueError("correspondence report belongs to a different branch")
    if sample_refresh.previous_match.label != target_state.sample_match.label:
        raise ValueError("sample refresh does not start from the active target label")
    if not np.allclose(
        sample_refresh.previous_match.point_RZ_m,
        target_state.sample_match.point_RZ_m,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("sample refresh does not start from the active target point")

    return replace(
        target_state,
        branch_reference=candidate_branch,
        sample_match=sample_refresh.candidate_match,
    )


def _require_loss(value: object, name: str) -> base_loss:
    if not isinstance(value, base_loss):
        raise TypeError(f"{name} must be an ESSOS loss")
    return value


def _scaled_loss(loss: base_loss, weight: float) -> base_loss:
    """Scale custom or composite losses without evaluating inactive terms."""

    if isinstance(loss, custom_loss):
        return weight * loss
    if isinstance(loss, composite_loss):
        return composite_loss(
            [_scaled_loss(component, weight) for component in loss.losses]
        )
    raise TypeError("continuation terms must contain custom ESSOS losses")


def _merged_dependencies(
    losses: Sequence[base_loss],
    dependencies: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if dependencies is not None:
        return dict(dependencies)

    merged: dict[str, Any] = {}
    for loss in losses:
        for name, dependency in loss.dependencies.items():
            if name in merged and merged[name] is not dependency:
                raise ValueError(
                    f"losses use different objects for dependency {name!r}; "
                    "pass dependencies explicitly"
                )
            merged[name] = dependency
    return merged


def compose_manifold_stage2_loss(
    base_stage2_loss: base_loss,
    continuation_state: ManifoldContinuationState,
    *,
    return_map_loss: base_loss | None = None,
    xline_loss: base_loss | None = None,
    xline_clearance_loss: base_loss | None = None,
    field_dependency: str = "field",
    dependencies: Mapping[str, Any] | None = None,
) -> base_loss:
    """Compose one fixed-weight inner loss for the active continuation stage.

    ``base_stage2_loss`` remains active with unit weight.  A topology term is
    only added when its stage weight is positive, so a zero-weight term does
    not incur tracing or compilation cost.  The manifold term is always built
    from the state's exact, immutable PyNA sample label.
    """

    base = _require_loss(base_stage2_loss, "base_stage2_loss")
    if not isinstance(continuation_state, ManifoldContinuationState):
        raise TypeError("continuation_state must be ManifoldContinuationState")

    stage = continuation_state.stage
    sources = [base]
    weighted_terms: list[base_loss] = []
    optional_terms = (
        (stage.return_map_weight, return_map_loss, "return_map_loss"),
        (stage.xline_weight, xline_loss, "xline_loss"),
        (
            stage.xline_clearance_weight,
            xline_clearance_loss,
            "xline_clearance_loss",
        ),
    )
    for weight, loss, name in optional_terms:
        if weight == 0.0:
            continue
        if loss is None:
            raise ValueError(f"{name} is required when its stage weight is positive")
        source = _require_loss(loss, name)
        sources.append(source)
        weighted_terms.append(_scaled_loss(source, weight))

    if stage.manifold_weight > 0.0:
        manifold_loss = make_manifold_stage2_loss(
            continuation_state.target_state,
            field_dependency=field_dependency,
        )
        sources.append(manifold_loss)
        weighted_terms.append(_scaled_loss(manifold_loss, stage.manifold_weight))

    if stage.strike_weight > 0.0:
        if continuation_state.strike_target_state is None:
            raise ValueError(
                "strike_target_state is required when strike_weight is positive"
            )
        strike_loss = make_manifold_strike_stage2_loss(
            continuation_state.strike_target_state,
            field_dependency=field_dependency,
        )
        sources.append(strike_loss)
        weighted_terms.append(_scaled_loss(strike_loss, stage.strike_weight))

    if stage.leg_clearance_weight > 0.0:
        if continuation_state.leg_target_state is None:
            raise ValueError("leg_target_state is required when leg_clearance_weight is positive")
        leg_loss = make_manifold_leg_stage2_loss(
            continuation_state.leg_target_state, field_dependency=field_dependency,
        )
        sources.append(leg_loss)
        weighted_terms.append(_scaled_loss(leg_loss, stage.leg_clearance_weight))

    if stage.heat_weight > 0.0:
        if continuation_state.heat_target_state is None:
            raise ValueError(
                "heat_target_state is required when heat_weight is positive"
            )
        heat_loss = make_manifold_heat_stage2_loss(
            continuation_state.heat_target_state,
            field_dependency=field_dependency,
        )
        sources.append(heat_loss)
        weighted_terms.append(_scaled_loss(heat_loss, stage.heat_weight))

    total = base
    for term in weighted_terms:
        total = total + term
    total.dependencies = _merged_dependencies(sources, dependencies)
    return total


def accept_manifold_continuation_stage(
    continuation_state: ManifoldContinuationState,
    candidate_branch: Any,
    sample_refresh: Any,
    correspondence: Any,
    *,
    accepted_strike_target: ManifoldStrikeStage2Target | None = None,
    accepted_heat_target: ManifoldHeatStage2Target | None = None,
    accepted_leg_target: ManifoldLegStage2Target | None = None,
) -> ManifoldContinuationState:
    """Advance only after PyNA accepts label refresh and JAX/Cyna parity."""

    if not isinstance(continuation_state, ManifoldContinuationState):
        raise TypeError("continuation_state must be ManifoldContinuationState")
    refreshed_target = refresh_manifold_stage2_target(
        continuation_state.target_state,
        candidate_branch,
        sample_refresh,
        correspondence,
    )
    active_strike_target = continuation_state.strike_target_state
    if active_strike_target is None:
        if accepted_strike_target is not None:
            raise ValueError("cannot add a strike target during stage acceptance")
        refreshed_strike_target = None
    else:
        if accepted_strike_target is None:
            raise ValueError("the active strike target requires accepted refresh state")
        if not isinstance(accepted_strike_target, ManifoldStrikeStage2Target):
            raise TypeError(
                "accepted_strike_target must be ManifoldStrikeStage2Target"
            )
        if accepted_strike_target.label != active_strike_target.label:
            raise ValueError("accepted strike refresh changed the active label")
        if accepted_strike_target.distance_mode != active_strike_target.distance_mode:
            raise ValueError("accepted strike refresh changed the distance mode")
        if not np.array_equal(
            accepted_strike_target.target_position_m,
            active_strike_target.target_position_m,
        ):
            raise ValueError("accepted strike refresh changed the physical target")
        refreshed_strike_target = accepted_strike_target
    active_heat_target = continuation_state.heat_target_state
    if active_heat_target is None:
        if accepted_heat_target is not None:
            raise ValueError("cannot add a heat target during stage acceptance")
        refreshed_heat_target = None
    else:
        if accepted_heat_target is None:
            raise ValueError("the active heat target requires accepted refresh state")
        if not isinstance(accepted_heat_target, ManifoldHeatStage2Target):
            raise TypeError("accepted_heat_target must be ManifoldHeatStage2Target")
        if accepted_heat_target.labels != active_heat_target.labels:
            raise ValueError("accepted heat refresh changed the active labels")
        if accepted_heat_target.power_provenance != active_heat_target.power_provenance:
            raise ValueError("accepted heat refresh changed power provenance")
        fixed_array_fields = (
            "strike_powers_W",
            "wall_cell_centers_xyz_m",
            "wall_cell_areas_m2",
        )
        for attribute in fixed_array_fields:
            if not np.array_equal(
                getattr(accepted_heat_target, attribute),
                getattr(active_heat_target, attribute),
            ):
                raise ValueError(
                    f"accepted heat refresh changed {attribute}"
                )
        fixed_scalar_fields = (
            "deposition_width_m",
            "maximum_heat_flux_W_m2",
            "heat_flux_scale_W_m2",
        )
        for attribute in fixed_scalar_fields:
            if getattr(accepted_heat_target, attribute) != getattr(
                active_heat_target,
                attribute,
            ):
                raise ValueError(
                    f"accepted heat refresh changed {attribute}"
                )
        if accepted_heat_target.branch_reference.label != candidate_branch.label:
            raise ValueError("accepted heat refresh belongs to a different branch")
        refreshed_heat_target = accepted_heat_target
    active_leg_target = continuation_state.leg_target_state
    if active_leg_target is None:
        if accepted_leg_target is not None:
            raise ValueError("cannot add a leg target during stage acceptance")
    else:
        if accepted_leg_target is None:
            raise ValueError("the active leg target requires accepted refresh state")
        require_same_leg_configuration(active_leg_target, accepted_leg_target)
        if accepted_leg_target.branch_reference is not candidate_branch:
            raise ValueError("accepted leg refresh must use the candidate branch snapshot")
    next_index = min(
        continuation_state.stage_index + 1,
        len(continuation_state.schedule) - 1,
    )
    return replace(
        continuation_state,
        target_state=refreshed_target,
        stage_index=next_index,
        strike_target_state=refreshed_strike_target,
        heat_target_state=refreshed_heat_target,
        leg_target_state=accepted_leg_target,
    )


__all__ = [
    "ManifoldContinuationSchedule",
    "ManifoldContinuationStage",
    "ManifoldContinuationState",
    "ManifoldStage2Target",
    "accept_manifold_continuation_stage",
    "compose_manifold_stage2_loss",
    "make_manifold_stage2_loss",
    "manifold_stage2_target_loss",
    "refresh_manifold_stage2_target",
    "trace_manifold_reference",
]
