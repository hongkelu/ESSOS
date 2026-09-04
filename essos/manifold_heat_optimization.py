"""Differentiable inner-loop heat loading from PyNA-labelled wall strikes.

PyNA/Cyna owns quantitative strike powers, exact manifold labels, and global
first-wall hits.  ESSOS holds one accepted outer snapshot while differentiating
the corresponding strike motion through its live coil field.  A fixed set of
wall monitor cells receives each strike's power through a Gaussian kernel that
is normalized with cell area, so the discrete surrogate conserves power
exactly.

The Gaussian model is deliberately local and sequential: it is smooth enough
for the Stage-2 inner solve, but it does not replace PyNA's production wall
trace, connection-length physics, or authoritative heat map.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from essos.losses import custom_loss
from essos.manifold import trace_manifold_wall_strike_bundle


def _pyna_strike_contract_api():
    try:
        from pyna.topo.manifold_correspondence import ManifoldBranchReference
        from pyna.topo.manifold_strike_contracts import (
            LocalWallPlane,
            ManifoldStrikeMatch,
        )
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS manifold heat optimization requires PyNA's strike contracts"
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


def _nonnegative_scalar(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _positive_scalar(value: object, name: str) -> float:
    result = _nonnegative_scalar(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _damping(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return result


@dataclass(frozen=True)
class ManifoldHeatStage2Target:
    """Frozen quantitative strike bundle and local wall deposition model."""

    branch_reference: Any
    strike_matches: tuple[Any, ...]
    wall_planes: tuple[Any, ...]
    strike_powers_W: np.ndarray
    power_provenance: str
    wall_cell_centers_xyz_m: np.ndarray
    wall_cell_areas_m2: np.ndarray
    deposition_width_m: float
    maximum_heat_flux_W_m2: float
    heat_flux_scale_W_m2: float
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
        try:
            matches = tuple(self.strike_matches)
            planes = tuple(self.wall_planes)
        except TypeError as exc:
            raise TypeError("strike_matches and wall_planes must be iterable") from exc
        if not matches:
            raise ValueError("strike_matches and wall_planes must be non-empty")
        if len(matches) != len(planes):
            raise ValueError("strike_matches and wall_planes must have equal length")
        if not all(isinstance(match, ManifoldStrikeMatch) for match in matches):
            raise TypeError("every strike match must be a PyNA ManifoldStrikeMatch")
        if not all(isinstance(plane, LocalWallPlane) for plane in planes):
            raise TypeError("every wall plane must be a PyNA LocalWallPlane")
        labels = tuple(match.label for match in matches)
        if len(set(labels)) != len(labels):
            raise ValueError("strike labels must be unique")
        expected_seed_orders = tuple(
            int(order) for order in self.branch_reference.seed_orders
        )
        if tuple(match.label.seed_order for match in matches) != expected_seed_orders:
            raise ValueError(
                "strike matches must follow the complete branch seed order"
            )
        for match, plane in zip(matches, planes, strict=True):
            if match.label.branch_label != self.branch_reference.label:
                raise ValueError("strike match belongs to a different manifold branch")
            if plane.strike_label != match.label:
                raise ValueError("wall plane belongs to a different strike label")
            self.branch_reference.seed_index(match.label.seed_order)

        powers = np.asarray(self.strike_powers_W, dtype=float).ravel()
        if powers.shape != (len(matches),):
            raise ValueError("strike_powers_W must have one value per strike label")
        if not np.all(np.isfinite(powers)) or np.any(powers < 0.0):
            raise ValueError("strike_powers_W must be non-negative and finite")
        if float(np.sum(powers)) <= 0.0:
            raise ValueError("strike_powers_W must contain positive total power")
        provenance = str(self.power_provenance).strip()
        if not provenance:
            raise ValueError("power_provenance must identify the quantitative source")

        centers = np.asarray(self.wall_cell_centers_xyz_m, dtype=float)
        if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] != 3:
            raise ValueError("wall_cell_centers_xyz_m must have shape (n_cells, 3)")
        if not np.all(np.isfinite(centers)):
            raise ValueError("wall_cell_centers_xyz_m must be finite")
        areas = np.asarray(self.wall_cell_areas_m2, dtype=float).ravel()
        if areas.shape != (centers.shape[0],):
            raise ValueError("wall_cell_areas_m2 must have shape (n_cells,)")
        if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError("wall_cell_areas_m2 must be positive and finite")

        object.__setattr__(self, "strike_matches", matches)
        object.__setattr__(self, "wall_planes", planes)
        object.__setattr__(self, "strike_powers_W", powers.copy())
        object.__setattr__(self, "power_provenance", provenance)
        object.__setattr__(self, "wall_cell_centers_xyz_m", centers.copy())
        object.__setattr__(self, "wall_cell_areas_m2", areas.copy())
        for attribute in (
            "deposition_width_m",
            "heat_flux_scale_W_m2",
            "maximum_phi_shift",
        ):
            object.__setattr__(
                self,
                attribute,
                _positive_scalar(getattr(self, attribute), attribute),
            )
        object.__setattr__(
            self,
            "maximum_heat_flux_W_m2",
            _nonnegative_scalar(
                self.maximum_heat_flux_W_m2,
                "maximum_heat_flux_W_m2",
            ),
        )
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
    def labels(self) -> tuple[str, ...]:
        return tuple(match.label.key for match in self.strike_matches)

    @property
    def total_power_W(self) -> float:
        return float(np.sum(self.strike_powers_W))


class ManifoldHeatFluxState(NamedTuple):
    """Differentiable labelled strikes and their local wall heat flux."""

    strike_trace: Any
    heat_flux_W_m2: Any
    deposited_power_W: Any


def power_conserving_gaussian_heat_flux(
    strike_points_xyz_m: Any,
    strike_powers_W: Any,
    wall_cell_centers_xyz_m: Any,
    wall_cell_areas_m2: Any,
    *,
    deposition_width_m: float,
) -> Any:
    """Deposit strike powers onto fixed wall cells with exact normalization.

    Euclidean chord distance supplies a local Gaussian proxy.  Cell areas enter
    the softmax normalization, so ``sum(q * area)`` equals the supplied total
    strike power up to floating-point roundoff even when every raw Gaussian
    weight would otherwise underflow.
    """

    strike_points = jnp.asarray(strike_points_xyz_m)
    powers = jnp.asarray(strike_powers_W)
    centers = jnp.asarray(wall_cell_centers_xyz_m)
    areas = jnp.asarray(wall_cell_areas_m2)
    if strike_points.ndim != 2 or strike_points.shape[1] != 3:
        raise ValueError("strike_points_xyz_m must have shape (n_strikes, 3)")
    if powers.shape != (strike_points.shape[0],):
        raise ValueError("strike_powers_W must have shape (n_strikes,)")
    if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] != 3:
        raise ValueError("wall_cell_centers_xyz_m must have shape (n_cells, 3)")
    if areas.shape != (centers.shape[0],):
        raise ValueError("wall_cell_areas_m2 must have shape (n_cells,)")
    width = _positive_scalar(deposition_width_m, "deposition_width_m")

    displacement = strike_points[:, None, :] - centers[None, :, :]
    squared_distance = jnp.sum(displacement * displacement, axis=-1)
    log_cell_power = (
        jnp.log(areas)[None, :] - 0.5 * squared_distance / (width * width)
    )
    cell_power_fraction = jax.nn.softmax(log_cell_power, axis=1)
    return jnp.sum(
        powers[:, None] * cell_power_fraction / areas[None, :],
        axis=0,
    )


def trace_manifold_heat_reference(
    field: Any,
    target_state: ManifoldHeatStage2Target,
) -> Any:
    """Trace every exact heat-bundle label through the live ESSOS field."""

    if not isinstance(target_state, ManifoldHeatStage2Target):
        raise TypeError("target_state must be ManifoldHeatStage2Target")
    return trace_manifold_wall_strike_bundle(
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


def manifold_heat_flux_state(
    field: Any,
    *,
    target_state: ManifoldHeatStage2Target,
) -> ManifoldHeatFluxState:
    """Return the smooth local wall heat flux for the live labelled strikes."""

    trace = trace_manifold_heat_reference(field, target_state)
    heat_flux = power_conserving_gaussian_heat_flux(
        trace.intersections.point_xyz,
        target_state.strike_powers_W,
        target_state.wall_cell_centers_xyz_m,
        target_state.wall_cell_areas_m2,
        deposition_width_m=target_state.deposition_width_m,
    )
    areas = jnp.asarray(target_state.wall_cell_areas_m2)
    return ManifoldHeatFluxState(
        strike_trace=trace,
        heat_flux_W_m2=heat_flux,
        deposited_power_W=jnp.sum(heat_flux * areas),
    )


def manifold_heat_stage2_limit_loss(
    field: Any,
    *,
    target_state: ManifoldHeatStage2Target,
) -> Any:
    """Area-weighted squared-hinge penalty above a physical heat-flux limit."""

    state = manifold_heat_flux_state(field, target_state=target_state)
    residual = jnp.maximum(
        state.heat_flux_W_m2 - target_state.maximum_heat_flux_W_m2,
        0.0,
    ) / target_state.heat_flux_scale_W_m2
    areas = jnp.asarray(target_state.wall_cell_areas_m2)
    return 0.5 * jnp.sum(areas * residual * residual) / jnp.sum(areas)


def make_manifold_heat_stage2_loss(
    target_state: ManifoldHeatStage2Target,
    *,
    field_dependency: str = "field",
) -> custom_loss:
    """Build an ESSOS custom loss for one accepted quantitative heat bundle."""

    if not isinstance(target_state, ManifoldHeatStage2Target):
        raise TypeError("target_state must be ManifoldHeatStage2Target")
    dependency = str(field_dependency).strip()
    if not dependency:
        raise ValueError("field_dependency must not be empty")
    return custom_loss(
        manifold_heat_stage2_limit_loss,
        dependency,
        target_state=target_state,
    )


def refresh_manifold_heat_stage2_target(
    target_state: ManifoldHeatStage2Target,
    candidate_branch: Any,
    heat_validation: Any,
) -> ManifoldHeatStage2Target:
    """Accept PyNA's quantitative heat report without changing fixed physics."""

    if not isinstance(target_state, ManifoldHeatStage2Target):
        raise TypeError("target_state must be ManifoldHeatStage2Target")
    try:
        from pyna.topo.manifold_correspondence import ManifoldBranchReference
        from pyna.topo.manifold_heat import ManifoldHeatValidationReport
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "ESSOS manifold heat refresh requires PyNA's production heat gate"
        ) from exc
    if not isinstance(candidate_branch, ManifoldBranchReference):
        raise TypeError("candidate_branch must be a PyNA ManifoldBranchReference")
    if not isinstance(heat_validation, ManifoldHeatValidationReport):
        raise TypeError("heat_validation must be a PyNA ManifoldHeatValidationReport")
    if not heat_validation.accepted:
        reason = heat_validation.rejection_reason or "unknown"
        raise ValueError(f"PyNA rejected manifold heat validation: {reason}")
    if heat_validation.branch_label != candidate_branch.label:
        raise ValueError("heat validation belongs to a different manifold branch")
    if tuple(match.label.key for match in heat_validation.strike_matches) != (
        target_state.labels
    ):
        raise ValueError("heat validation changed the ordered strike labels")
    if not np.array_equal(
        heat_validation.strike_powers_W,
        target_state.strike_powers_W,
    ):
        raise ValueError("heat validation changed strike_powers_W")
    if heat_validation.power_provenance != target_state.power_provenance:
        raise ValueError("heat validation changed power provenance")
    if heat_validation.maximum_heat_flux_W_m2 != (
        target_state.maximum_heat_flux_W_m2
    ):
        raise ValueError("heat validation used a different heat-flux limit")
    return replace(
        target_state,
        branch_reference=candidate_branch,
        strike_matches=heat_validation.strike_matches,
        wall_planes=heat_validation.wall_planes,
    )


__all__ = [
    "ManifoldHeatFluxState",
    "ManifoldHeatStage2Target",
    "make_manifold_heat_stage2_loss",
    "manifold_heat_flux_state",
    "manifold_heat_stage2_limit_loss",
    "power_conserving_gaussian_heat_flux",
    "refresh_manifold_heat_stage2_target",
    "trace_manifold_heat_reference",
]
