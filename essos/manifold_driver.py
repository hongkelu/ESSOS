"""Optimizer-facing rollback for manifold-aware Stage-2 trial steps."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from essos.manifold_optimization import ManifoldContinuationState
from essos.manifold_validation import ManifoldContinuationValidationReport


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


__all__ = [
    "ManifoldBacktrackingAttempt",
    "ManifoldBacktrackingResult",
    "validated_manifold_backtracking_step",
]
