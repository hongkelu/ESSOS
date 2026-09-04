"""Stage-2 optimizer state for PyNA-owned invariant-manifold targets.

PyNA discovers and validates topology.  ESSOS stores one accepted, immutable
snapshot of that outer state while differentiating coil objectives.  This
module turns a ``ManifoldBranchReference`` and exact ``ManifoldSampleMatch``
into a JAX loss without re-inferring branch identity inside the optimizer.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from essos.losses import custom_loss
from essos.manifold import trace_manifold_branch


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

        object.__setattr__(self, "target_RZ_m", target.copy())
        object.__setattr__(self, "rz_scales_m", scales.copy())
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


__all__ = [
    "ManifoldStage2Target",
    "make_manifold_stage2_loss",
    "manifold_stage2_target_loss",
    "refresh_manifold_stage2_target",
    "trace_manifold_reference",
]
