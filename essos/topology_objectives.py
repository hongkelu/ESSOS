"""Topology optimization targets and objectives.

Stage-2 manifold targets (branch samples, first-wall strikes, heat flux and
pre-strike leg clearance) with their continuation schedule, and the
refresh/model objectives used by :mod:`essos.topology_optimizer`: labelled
manifold samples, general open-line bundles, and closed invariant surfaces
with or without an equilibrium response.
"""
from __future__ import annotations
import operator
from dataclasses import dataclass, replace
from typing import Any
import jax.numpy as jnp
import numpy as np
from essos.losses import custom_loss
from essos.topology import trace_manifold_wall_strike
from typing import Any, NamedTuple
import jax
from essos.topology import trace_manifold_wall_strike_bundle
import math
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, NamedTuple
from essos.topology import trace_manifold_wall_strike_bundle_trajectories
from typing import Any, Mapping, Sequence
from essos.losses import base_loss, composite_loss, custom_loss
from essos.topology import trace_manifold_branch
from dataclasses import dataclass
import hashlib
import json
from essos.topology_optimizer import Evaluation
from pyna.topo.manifold_correspondence import ManifoldBranchReference,compare_jax_manifold_branch
from pyna.topo.manifold_refresh import refresh_manifold_branch_reference_field
import diffrax
from essos.topology_optimizer import content_identity
from essos.dynamics import connection_length


# ----------------------------------------------------------------------------
# From essos/manifold_strike_optimization.py: Stage-2 coil objectives for PyNA-labelled first-wall strikes.
# ----------------------------------------------------------------------------

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
    derivative_mode: str = "frozen_reference"

    def __post_init__(self) -> None:
        from pyna.topo.snapshot import immutable_array
        if self.derivative_mode not in ("frozen_reference", "moving_linear_seed"):
            raise ValueError("Unsupported derivative_mode")
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

        object.__setattr__(self, "target_position_m", immutable_array(target))
        object.__setattr__(self, "position_scales_m", immutable_array(scales))
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
        derivative_mode=target_state.derivative_mode,
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


def refresh_manifold_strike_stage2_target(
    target_state: ManifoldStrikeStage2Target,
    candidate_branch: Any,
    strike_refresh: Any,
    correspondence: Any,
    candidate_wall_plane: Any,
) -> ManifoldStrikeStage2Target:
    """Accept a fully PyNA-validated strike snapshot without relabelling."""

    if not isinstance(target_state, ManifoldStrikeStage2Target):
        raise TypeError("target_state must be ManifoldStrikeStage2Target")
    (
        ManifoldBranchReference,
        _ManifoldStrikeMatch,
        LocalWallPlane,
    ) = _pyna_strike_contract_api()
    from pyna.topo.manifold_strike_contracts import ManifoldStrikeRefreshReport
    from pyna.topo.manifold_strike_correspondence import (
        ManifoldStrikeCorrespondenceReport,
    )

    if not isinstance(candidate_branch, ManifoldBranchReference):
        raise TypeError("candidate_branch must be a PyNA ManifoldBranchReference")
    if not isinstance(strike_refresh, ManifoldStrikeRefreshReport):
        raise TypeError("strike_refresh must be a PyNA ManifoldStrikeRefreshReport")
    if not isinstance(correspondence, ManifoldStrikeCorrespondenceReport):
        raise TypeError(
            "correspondence must be a PyNA ManifoldStrikeCorrespondenceReport"
        )
    if not isinstance(candidate_wall_plane, LocalWallPlane):
        raise TypeError("candidate_wall_plane must be a PyNA LocalWallPlane")
    if not strike_refresh.accepted or strike_refresh.candidate_match is None:
        reason = strike_refresh.rejection_reason or "unknown"
        raise ValueError(f"PyNA rejected manifold strike refresh: {reason}")
    if not correspondence.accepted:
        reason = correspondence.rejection_reason or "unknown"
        raise ValueError(f"PyNA rejected JAX/Cyna strike correspondence: {reason}")

    previous_match = strike_refresh.previous_match
    candidate_match = strike_refresh.candidate_match
    if previous_match.label != target_state.strike_match.label or not np.allclose(
        previous_match.point_RZPhi_m_rad,
        target_state.strike_match.point_RZPhi_m_rad,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("strike refresh does not start from the active target")
    if candidate_match.label.branch_label != candidate_branch.label:
        raise ValueError("candidate strike belongs to a different manifold branch")
    if correspondence.label != candidate_match.label:
        raise ValueError("strike correspondence belongs to a different label")
    if not np.allclose(
        correspondence.production_coordinates_m,
        candidate_match.distance_coordinates_m,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("strike correspondence uses a different production hit")
    if candidate_wall_plane.strike_label != candidate_match.label:
        raise ValueError("candidate wall plane belongs to a different strike label")

    return replace(
        target_state,
        branch_reference=candidate_branch,
        strike_match=candidate_match,
        wall_plane=candidate_wall_plane,
    )


# ----------------------------------------------------------------------------
# From essos/manifold_heat_optimization.py: Differentiable inner-loop heat loading from PyNA-labelled wall strikes.
# ----------------------------------------------------------------------------

def _positive_scalar(value: object, name: str) -> float:
    result = _nonnegative_scalar(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
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
    derivative_mode: str = "frozen_reference"

    def __post_init__(self) -> None:
        from pyna.topo.snapshot import immutable_array
        if self.derivative_mode not in ("frozen_reference", "moving_linear_seed"):
            raise ValueError("Unsupported derivative_mode")
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
        object.__setattr__(self, "strike_powers_W", immutable_array(powers))
        object.__setattr__(self, "power_provenance", provenance)
        object.__setattr__(self, "wall_cell_centers_xyz_m", immutable_array(centers))
        object.__setattr__(self, "wall_cell_areas_m2", immutable_array(areas))
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
        derivative_mode=target_state.derivative_mode,
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


# ----------------------------------------------------------------------------
# From essos/manifold_leg_optimization.py: Differentiable pre-strike clearance for exact invariant-manifold legs.
# ----------------------------------------------------------------------------

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
    derivative_mode: str = "frozen_reference"

    def __post_init__(self) -> None:
        if self.derivative_mode not in ("frozen_reference", "moving_linear_seed"):
            raise ValueError("Unsupported derivative_mode")
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
    "wall_residual_tolerance", "minimum_abs_transversality", "derivative_mode",
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
        derivative_mode=target_state.derivative_mode,
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


# ----------------------------------------------------------------------------
# From essos/manifold_optimization.py: Stage-2 optimizer state for PyNA-owned invariant-manifold targets.
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# From essos/manifold_sample_objective.py: Production-refreshed labelled manifold samples for the constrained optimizer.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifoldSampleObjective:
    field_factory: object
    production_field_factory: object
    initial_branch: ManifoldBranchReference
    sample_label: object
    target: tuple[float,float]
    scales: tuple[float,float]
    identity: str
    maximum_anchor_displacement: float = .002
    minimum_direction_alignment: float = .9
    maximum_sample_displacement: float = .002
    correspondence_tolerance: float = 2e-4
    steps: int = 64
    derivative_mode: str = 'moving_linear_seed'

    def __post_init__(self):
        self.initial_branch.point(self.sample_label)
        for name in ('target','scales'):
            value=tuple(map(float,getattr(self,name)))
            if len(value)!=2 or not np.all(np.isfinite(value)):raise ValueError('Finite RZ target and scales required')
            object.__setattr__(self,name,value)
        if min(self.scales)<=0 or not self.identity:raise ValueError('Positive scales and identity required')
        if self.derivative_mode not in ('frozen_reference','moving_linear_seed'):raise ValueError('Unknown derivative mode')

    @property
    def fingerprint(self):
        controls=(self.identity,self.target,self.scales,self.sample_label.key,self.maximum_anchor_displacement,
                  self.minimum_direction_alignment,self.maximum_sample_displacement,self.correspondence_tolerance,
                  self.steps,self.derivative_mode)
        return hashlib.sha256(repr(controls).encode()+json.dumps(self.initial_branch.to_dict(),sort_keys=True,allow_nan=False).encode()).hexdigest()

    def _branch(self,snapshot):
        if not snapshot:return self.initial_branch
        if snapshot.get('definition')!=self.fingerprint:raise ValueError('Manifold objective definition changed')
        return ManifoldBranchReference.from_dict(snapshot['branch'])

    def _residual(self,parameters,branch):
        traced=trace_manifold_reference(self.field_factory(parameters),branch,n_steps_per_span=self.steps,
                                        bphi_floor=1e-8,derivative_mode=self.derivative_mode)
        point=traced.generations[self.sample_label.generation_index,branch.seed_index(self.sample_label.seed_order)]
        return (point-jnp.asarray(self.target))/jnp.asarray(self.scales)

    def model(self,parameters,snapshot):
        branch=self._branch(snapshot)
        local_residual=self._residual(jnp.asarray(parameters),branch)
        # Match the local model value to the independently refreshed physical
        # sample. The JAX-Cyna discretization offset is held fixed in this model.
        residual=(branch.point(self.sample_label)-np.asarray(self.target))/np.asarray(self.scales)
        jacobian=jax.jacfwd(lambda p:self._residual(p,branch))(jnp.asarray(parameters))
        valid=bool(np.all(np.isfinite(residual)) and np.all(np.isfinite(jacobian)))
        return Evaluation(np.asarray(residual),snapshot,valid=valid,status='valid' if valid else 'local_response_invalid',jacobian=np.asarray(jacobian),
                          diagnostics={'fixed_discretization_offset':(np.asarray(local_residual)-residual).tolist()})

    def refresh(self,parameters,snapshot):
        previous=self._branch(snapshot)
        production=self.production_field_factory(parameters)
        report=refresh_manifold_branch_reference_field(production,previous,
                    maximum_anchor_displacement_m=self.maximum_anchor_displacement,
                    minimum_direction_alignment=self.minimum_direction_alignment,
                    DPhi=abs(previous.map_span)/self.steps,fixed_point_residual_tolerance=1e-9)
        if not report.accepted:
            return Evaluation(np.zeros(2),snapshot,valid=False,status=report.rejection_reason or 'production_refresh_failed')
        branch=report.candidate_branch
        displacement=float(np.linalg.norm(branch.point(self.sample_label)-previous.point(self.sample_label)))
        trace=trace_manifold_reference(self.field_factory(parameters),branch,n_steps_per_span=self.steps,
                                       bphi_floor=1e-8,derivative_mode=self.derivative_mode)
        correspondence=compare_jax_manifold_branch(branch,trace.generations,absolute_tolerance_m=self.correspondence_tolerance,require_complete=True)
        valid=correspondence.accepted and displacement<=self.maximum_sample_displacement
        residual=(branch.point(self.sample_label)-np.asarray(self.target))/np.asarray(self.scales)
        updated=dict(definition=self.fingerprint,branch=branch.to_dict())
        status='valid' if valid else 'sample_displacement_exceeded' if displacement>self.maximum_sample_displacement else 'jax_cyna_correspondence_failed'
        return Evaluation(residual,updated,valid=valid,status=status,
                          diagnostics=dict(sample_displacement_m=displacement,jax_cyna_deviation_m=correspondence.max_deviation_m))


# ----------------------------------------------------------------------------
# From essos/open_bundle_optimization.py: General open-line objectives with independent production acceptance.
# ----------------------------------------------------------------------------

LAUNCH_CONVENTIONS = ("fixed_physical", "moving_physical", "flux_labelled", "manifold_generated")


@dataclass(frozen=True)
class LaunchBundle:
    """Labelled launch points ``(R, Z)`` on the plane ``phi = phi_start`` with quadrature weights.

    Moving, flux-labelled and manifold-generated launches are supplied by the
    objective's ``launch_fn`` as active arrays; the stored points are then only the
    declared reference geometry.
    """

    labels: tuple
    launch_RZ: np.ndarray
    weights: np.ndarray
    launch_convention: str = "fixed_physical"
    weights_move: bool = False

    def __post_init__(self):
        labels = tuple(self.labels)
        points = np.array(self.launch_RZ, dtype=float)
        weights = np.array(self.weights, dtype=float)
        if not labels or len(set(labels)) != len(labels) or not all(isinstance(s, str) and s for s in labels):
            raise ValueError("Nonempty unique launch labels required")
        if points.shape != (len(labels), 2) or weights.shape != (len(labels),):
            raise ValueError("One (R, Z) launch point and one weight per label required")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Launch points must be finite and weights positive")
        if self.launch_convention not in LAUNCH_CONVENTIONS:
            raise ValueError("Unknown launch convention")
        points.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "launch_RZ", points)
        object.__setattr__(self, "weights", weights)


class _ParametrizedField:
    """Adapts ``field_fn(xyz, parameters)`` to the field interface of ``connection_length``."""

    def __init__(self, field_fn, parameters):
        self.field_fn = field_fn
        self.parameters = parameters

    def B_contravariant(self, xyz):
        return self.field_fn(xyz, self.parameters)


@dataclass(frozen=True)
class OpenBundleObjective:
    """Residuals of observables of labelled open field lines that end on a wall.

    Args:
        bundle: :class:`LaunchBundle` of launch points on ``phi = phi_start``.
        field_fn: ``field_fn(xyz, parameters)`` returning Cartesian ``B``.
        wall_fn: ``wall_fn(xyz, wall_parameters)``, positive inside the wall and zero on it.
        wall_parameters: ``wall_parameters(parameters)`` giving the (possibly moving) wall.
        observables: ``observables(hit_RZPhi, lengths, weights, parameters)`` vector.
        production_validate: ``production_validate(parameters, snapshot)`` returning
            pyna's ``OpenBundleValidation``.
        target, scales: Observable target and positive residual scales.
        phi_guesses: Expected unwrapped hit angle of every launch; its sign relative to
            ``phi_start`` selects the direction along ``B`` that is traced.
        identity: Version string of the callables above, part of the fingerprint.
        max_length: Cap on the traced length [m].
        tolerance: Integration and event tolerance of ``connection_length``.
    """

    bundle: object
    field_fn: object
    wall_fn: object
    wall_parameters: object
    observables: object
    production_validate: object
    target: tuple
    scales: tuple
    phi_guesses: tuple
    identity: str
    phi_start: float = 0.
    maximum_phi_shift: float = .2
    maximum_hit_discrepancy: float = 1e-3
    max_length: float = 100.
    tolerance: float = 1e-10
    launch_fn: object = None
    weights_fn: object = None

    def __post_init__(self):
        for name in ("target", "scales", "phi_guesses"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError("Finite objective vectors required")
            object.__setattr__(self, name, tuple(map(float, values)))
        if not self.identity or not self.target or len(self.scales) != len(self.target) or min(self.scales) <= 0:
            raise ValueError("Invalid fixed objective definition")
        if len(self.phi_guesses) != len(self.bundle.labels):
            raise ValueError("One hit-angle guess per launch required")
        if any(guess == self.phi_start for guess in self.phi_guesses):
            raise ValueError("Hit-angle guesses must differ from phi_start")
        if self.bundle.launch_convention != "fixed_physical" and self.launch_fn is None:
            raise ValueError("Moving launches require a physical launch function")
        if self.bundle.weights_move and self.weights_fn is None:
            raise ValueError("Moving weights require a physical weight function")
        controls = [self.phi_start, self.maximum_phi_shift, self.maximum_hit_discrepancy,
                    self.max_length, self.tolerance]
        if not np.all(np.isfinite(controls)) or min(controls[1:]) <= 0:
            raise ValueError("Invalid trace controls")

    @property
    def fingerprint(self):
        # identity versions callable field, wall, launch and observable definitions.
        return content_identity(currents=None, geometry=None, wall=None, numerics=None, topology=None, target=dict(
            coordinate_contract="cartesian_hits_unwrapped_phi_v2", wall_contract="xyz_positive_inside",
            identity=self.identity, labels=self.bundle.labels, launches=self.bundle.launch_RZ,
            weights=self.bundle.weights, convention=self.bundle.launch_convention,
            weights_move=self.bundle.weights_move, target=self.target, scales=self.scales,
            guesses=self.phi_guesses, phi_start=self.phi_start, shift=self.maximum_phi_shift,
            discrepancy=self.maximum_hit_discrepancy, max_length=self.max_length, tolerance=self.tolerance))

    def _check_snapshot(self, snapshot):
        if snapshot and snapshot.get("definition") != self.fingerprint:
            raise ValueError("Open-bundle snapshot definition changed")

    def _weights(self, c):
        weights = jnp.asarray(self.bundle.weights) if self.weights_fn is None else self.weights_fn(c)
        return weights if self.bundle.weights_move else jax.lax.stop_gradient(weights)

    def _launches(self, c):
        launches = jnp.asarray(self.bundle.launch_RZ if self.launch_fn is None else self.launch_fn(c))
        return jax.lax.stop_gradient(launches) if self.bundle.launch_convention == "fixed_physical" else launches

    def _direction(self, c, reference_phi):
        """Sign along B that advances every launch towards its expected hit angle."""
        launches = np.asarray(self._launches(jnp.asarray(c)))
        seeds = np.column_stack((launches[:, 0] * np.cos(self.phi_start),
                                 launches[:, 0] * np.sin(self.phi_start), launches[:, 1]))
        B = np.asarray(jax.vmap(lambda x: self.field_fn(x, jnp.asarray(c)))(jnp.asarray(seeds)))
        B_phi = (seeds[:, 0] * B[:, 1] - seeds[:, 1] * B[:, 0]) / launches[:, 0]
        signs = np.sign(B_phi * (np.asarray(reference_phi) - self.phi_start))
        if np.any(signs == 0) or np.any(signs != signs[0]):
            raise ValueError("All launches must reach their hits along the same direction of B")
        return float(signs[0])

    def _trace(self, c, reference_phi, direction):
        """Local hits ``(n, 3)`` in ``(R, Z, phi)`` and ``(x, y, z)``, lengths and validity."""
        launches = self._launches(c)
        seeds = jnp.column_stack((launches[:, 0] * jnp.cos(self.phi_start),
                                  launches[:, 0] * jnp.sin(self.phi_start), launches[:, 1]))
        wall_parameters = self.wall_parameters(c)
        result = connection_length(_ParametrizedField(self.field_fn, c), seeds,
                                   lambda xyz: self.wall_fn(xyz, wall_parameters),
                                   max_length=self.max_length, tolerance=self.tolerance,
                                   adjoint=diffrax.ForwardMode(), directions=(direction,))
        xyz = result["strike_points"][:, 0]
        lengths = result["lengths"][:, 0]
        wrapped = jnp.arctan2(xyz[:, 1], xyz[:, 0])
        # Unwrap onto the branch of the accepted (production) hit angle.
        turns = jax.lax.stop_gradient(jnp.round((jnp.asarray(reference_phi) - wrapped) / (2 * jnp.pi)))
        hits = jnp.column_stack((jnp.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2], wrapped + 2 * jnp.pi * turns))
        valid = jnp.all(result["hit"][:, 0]) & jnp.all(jnp.isfinite(lengths))
        return hits, xyz, lengths, valid

    def _residual(self, hits, lengths, c):
        values = jnp.asarray(self.observables(hits, lengths, self._weights(c), c))
        return (values - jnp.asarray(self.target)) / jnp.asarray(self.scales)

    def refresh(self, c, snapshot):
        from pyna.topo.open_validation import OpenBundleValidation

        self._check_snapshot(snapshot)
        checked = self.production_validate(np.array(c, copy=True), dict(snapshot))
        if not isinstance(checked, OpenBundleValidation):
            raise TypeError("Production callback must return OpenBundleValidation")
        if not checked.valid:
            return Evaluation(np.zeros(len(self.target)), snapshot, valid=False, status=checked.status)
        if checked.hit_RZPhi.shape != (len(self.bundle.labels), 3) or checked.connection_lengths.shape != (len(self.bundle.labels),):
            raise ValueError("Production launch identity/shape changed")
        reference_phi = checked.hit_RZPhi[:, 2]
        hits, xyz, _, local_valid = self._trace(jnp.asarray(c), reference_phi, self._direction(c, reference_phi))
        R, Z, phi = checked.hit_RZPhi.T
        physical_hits = np.column_stack((R * np.cos(phi), R * np.sin(phi), Z))
        discrepancy = float(np.max(np.linalg.norm(np.asarray(xyz) - physical_hits, axis=1)))
        phase_error = float(np.max(np.abs(np.asarray(hits)[:, 2] - phi)))
        valid = (bool(local_valid) and np.isfinite(discrepancy) and discrepancy <= self.maximum_hit_discrepancy
                 and phase_error <= self.maximum_phi_shift)
        if snapshot and np.max(np.abs(phi - np.asarray(snapshot["hit_phi"]))) > self.maximum_phi_shift:
            valid = False
        updated = dict(definition=self.fingerprint, hit_phi=phi.tolist(), hits=checked.hit_RZPhi.tolist(),
                       lengths=checked.connection_lengths.tolist(), parameters=np.asarray(c).tolist())
        return Evaluation(np.asarray(self._residual(checked.hit_RZPhi, checked.connection_lengths, c)), updated,
                          valid=valid, status="valid" if valid else "local_production_correspondence_failed",
                          diagnostics={"hit_discrepancy_m": discrepancy, "unwrapped_phase_discrepancy_rad": phase_error})

    def model(self, c, snapshot):
        self._check_snapshot(snapshot)
        if not snapshot:
            raise ValueError("Production refresh required before local linearization")
        if not np.array_equal(c, np.asarray(snapshot["parameters"])):
            raise ValueError("Model parameters differ from accepted snapshot")
        reference_phi = np.asarray(snapshot["hit_phi"])
        direction = self._direction(c, reference_phi)

        def residual(parameters):
            hits, _, lengths, _ = self._trace(parameters, reference_phi, direction)
            return self._residual(hits, lengths, parameters)

        _, _, _, valid = self._trace(jnp.asarray(c), reference_phi, direction)
        # The model value uses accepted production observables; the smooth
        # model Jacobian is qualified separately against production differences.
        value = self._residual(jnp.asarray(snapshot["hits"]), jnp.asarray(snapshot["lengths"]), jnp.asarray(c))
        return Evaluation(np.asarray(value), snapshot, jacobian=np.asarray(jax.jacfwd(residual)(jnp.asarray(c))),
                          valid=bool(valid))


# ----------------------------------------------------------------------------
# From essos/surface_optimization.py: Closed-surface objectives for the shared refreshed-merit optimizer.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CircleObjective:
    """Fixed observables, targets, and scaling, with separately refreshed tori.

    observables(unknowns, parameters) must compute the same physical quantities
    in model and refresh calls. Map omega is an angle per declared map span;
    callers convert it to transform using that span, never a hardcoded period.
    The torus problem chooses the physical label (currently signed section
    area). It must remain fixed throughout one optimization problem.
    """
    problem: object
    observables: object
    target: tuple[float,...]
    scales: tuple[float,...]
    initial_omega: float
    identity: str

    def __post_init__(self):
        target=np.asarray(self.target);scales=np.asarray(self.scales)
        if not self.identity or target.ndim!=1 or scales.shape!=target.shape or not np.all(np.isfinite(target)) or not np.all(np.isfinite(scales)) or np.any(scales<=0):raise ValueError('Invalid surface objective definition')
        object.__setattr__(self,'target',tuple(map(float,target)));object.__setattr__(self,'scales',tuple(map(float,scales)))

    @property
    def fingerprint(self):
        data=repr((self.identity,self.target,self.scales,self.initial_omega,self.problem.area,self.problem.length_scale,
                   self.problem.tolerance,self.problem.dense_tolerance,self.problem.maximum_condition,self.problem.minimum_speed)).encode()+self.problem.reference.tobytes()
        return hashlib.sha256(data).hexdigest()

    def _evaluate(self,parameters,snapshot,*,derivatives):
        if snapshot and snapshot.get('definition')!=self.fingerprint:raise ValueError('Surface snapshot definition changed')
        initial=snapshot.get('unknowns') if snapshot else None
        omega=snapshot.get('omega',self.initial_omega) if snapshot else self.initial_omega
        solution=self.problem.solve(parameters,omega=omega,initial=initial)
        diagnostics=dict(circle_status=solution.status,dense_invariance_residual=solution.dense_residual,
                         root_condition=solution.condition,minimum_speed=solution.minimum_speed,
                         counterterm=solution.counterterm,minimum_divisor=solution.minimum_divisor)
        if not solution.valid:
            return Evaluation(np.zeros(len(self.target)),snapshot,valid=False,status=solution.status,diagnostics=diagnostics)
        def residual(z,c):return (jnp.asarray(self.observables(z,c))-jnp.asarray(self.target))/jnp.asarray(self.scales)
        r=np.asarray(residual(solution.unknowns,parameters));jacobian=None
        if derivatives:
            dz=np.column_stack([self.problem.jvp(solution,d) for d in np.eye(len(parameters))])
            jacobian=np.asarray(jax.jacfwd(residual,0)(solution.unknowns,parameters))@dz+np.asarray(jax.jacfwd(residual,1)(solution.unknowns,parameters))
        updated=dict(definition=self.fingerprint,unknowns=solution.unknowns.tolist(),omega=solution.omega,
                     parameters=np.asarray(parameters).tolist(),label='signed_section_area',area=self.problem.area)
        return Evaluation(r,updated,jacobian=jacobian,diagnostics=diagnostics)

    def model(self,parameters,snapshot):
        return self._evaluate(parameters,snapshot,derivatives=True)

    def refresh(self,parameters,snapshot):
        return self._evaluate(parameters,snapshot,derivatives=False)


# ----------------------------------------------------------------------------
# From essos/equilibrium_surface_optimization.py: Joint equilibrium and closed-surface acceptance for reduced response models.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EquilibriumCircleObjective:
    """Compose an equilibrium response with a physical invariant-circle target.

    surface is a CircleObjective whose map and observables take the combined
    parameter vector concatenate((controls, equilibrium_unknowns)). Its map
    must use the declared Eulerian field and spatial domain. Every refresh
    re-solves the equilibrium and the surface; both validity gates must pass.
    The dense reference chain rule is intended for reduced problems. Material
    VMEC grid samples cannot be substituted for an Eulerian map here.
    """
    equilibrium: object
    surface: object
    identity: str

    def __post_init__(self):
        if not self.identity:raise ValueError('A versioned coupled problem identity is required')

    @property
    def fingerprint(self):
        return hashlib.sha256(repr((self.identity,self.equilibrium.definition,self.surface.fingerprint)).encode()).hexdigest()

    def _evaluate(self,c,snapshot,derivatives):
        if snapshot and snapshot.get('definition')!=self.fingerprint:raise ValueError('Coupled problem definition changed')
        equilibrium=self.equilibrium.solve(c,initial=snapshot.get('equilibrium_state'))
        diagnostics=dict(equilibrium_status=equilibrium.status,equilibrium_residual=equilibrium.residual_norm,
                         equilibrium_condition=equilibrium.condition,field_mode=self.equilibrium.definition.field_mode)
        if not equilibrium.valid:
            return Evaluation(np.zeros(len(self.surface.target)),snapshot,valid=False,status=equilibrium.status,diagnostics=diagnostics)
        combined=np.concatenate((np.asarray(c),equilibrium.state))
        surface_snapshot=snapshot.get('surface',{})
        result=(self.surface.model if derivatives else self.surface.refresh)(combined,surface_snapshot)
        if derivatives and result.valid:
            response=np.column_stack([self.equilibrium.state_jvp(equilibrium,d) for d in np.eye(len(c))])
            result.jacobian=result.jacobian@np.vstack((np.eye(len(c)),response))
        result.diagnostics={**result.diagnostics,**diagnostics}
        if result.valid:
            result.snapshot=dict(definition=self.fingerprint,parameters=np.asarray(c).tolist(),
                                 equilibrium_state=equilibrium.state.tolist(),surface=result.snapshot)
        else:result.snapshot=snapshot
        return result

    def model(self,c,snapshot):return self._evaluate(c,snapshot,True)

    def refresh(self,c,snapshot):return self._evaluate(c,snapshot,False)
