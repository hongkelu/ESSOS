"""End-to-end outer validation across ESSOS, PyNA JAX, and Cyna."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.topo.manifold_refresh")

from pyna.topo.manifold_correspondence import (
    ManifoldSampleMatch,
    manifold_branch_reference_from_trace,
)
from pyna.topo.manifold_strike_contracts import (
    LocalWallPlane,
    ManifoldStrikeLabel,
    ManifoldStrikeMatch,
)
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.flt import trace_fixed_point_manifolds_field
from pyna.toroidal.geometry import ToroidalWall

from essos.manifold import essos_field_to_pyna_cylindrical_grid
from essos.manifold_driver import validated_manifold_backtracking_step
from essos.manifold_optimization import (
    ManifoldContinuationSchedule,
    ManifoldContinuationStage,
    ManifoldContinuationState,
    ManifoldStage2Target,
)
from essos.manifold_validation import (
    validate_manifold_continuation_candidate,
)
from essos.manifold_strike_optimization import ManifoldStrikeStage2Target
from essos.manifold_strike_validation import ManifoldStrikeValidationConfig


RATE = 0.35
MAP_SPAN = 0.6
N_STEPS_PER_SPAN = 60
N_GENERATIONS = 3
SEED_DISTANCES = np.array([1.0e-3, 2.0e-3])
SEED_ORDERS = np.array([3, 7])
R_GRID = np.linspace(0.7, 1.5, 49)
Z_GRID = np.linspace(-0.4, 0.4, 49)
PHI_GRID = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)


def _require_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind is unavailable")


@jax.tree_util.register_pytree_node_class
class _ShiftedHyperbolicField:
    def __init__(self, parameters):
        self.parameters = jnp.asarray(parameters)

    def B(self, xyz):
        rate, center_r, center_z = self.parameters
        x, y, z = xyz
        radius = jnp.sqrt(x * x + y * y)
        cos_phi = x / radius
        sin_phi = y / radius
        b_r = rate * (radius - center_r)
        b_z = -rate * (z - center_z)
        b_phi = radius
        return jnp.array(
            [
                b_r * cos_phi - b_phi * sin_phi,
                b_r * sin_phi + b_phi * cos_phi,
                b_z,
            ]
        )

    def tree_flatten(self):
        return (self.parameters,), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        return cls(children[0])


def _initial_branch():
    field = _ShiftedHyperbolicField(jnp.array([RATE, 1.0, 0.0]))
    production_field = essos_field_to_pyna_cylindrical_grid(
        field,
        R_GRID,
        Z_GRID,
        PHI_GRID,
        batch_size=2048,
        axisymmetric=True,
    )
    expansion = np.exp(RATE * MAP_SPAN)
    fixed_point = FixedPoint(
        phi=0.0,
        R=1.0,
        Z=0.0,
        kind="X",
        DPm=np.diag([expansion, 1.0 / expansion]),
    )
    fixed_point.metadata.update(
        {
            "orbit_id": 21,
            "map_order_index": 0,
            "monodromy_map_span": MAP_SPAN,
        }
    )
    payload = trace_fixed_point_manifolds_field(
        production_field,
        [fixed_point],
        phi_section=0.0,
        map_span=MAP_SPAN,
        N_turns=N_GENERATIONS,
        DPhi=MAP_SPAN / N_STEPS_PER_SPAN,
        seed_distances=SEED_DISTANCES,
        seed_orders=SEED_ORDERS,
        refine_stable_inverse_anchor=False,
    )[0]
    return manifold_branch_reference_from_trace(
        payload,
        stability="unstable",
        seed_side=1,
        n_generations=N_GENERATIONS,
    )


def _continuation_state(*, generation_index=N_GENERATIONS):
    branch = _initial_branch()
    seed_index = 0
    label = branch.sample_label(generation_index, seed_index)
    point = branch.point(label)
    match = ManifoldSampleMatch(
        label=label,
        point_RZ_m=point,
        distance_m=0.0,
        jax_seed_index=seed_index,
    )
    target = ManifoldStage2Target(
        branch_reference=branch,
        sample_match=match,
        target_RZ_m=point,
        rz_scales_m=np.array([0.01, 0.01]),
        n_steps_per_span=N_STEPS_PER_SPAN,
        newton_iterations=4,
    )
    schedule = ManifoldContinuationSchedule(
        (
            ManifoldContinuationStage(name="base"),
            ManifoldContinuationStage(name="manifold", manifold_weight=0.1),
        )
    )
    return ManifoldContinuationState(schedule, target)


def _strike_wall(n_phi=8, n_pol=64):
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pol, endpoint=False)
    R = np.broadcast_to(1.0 + 0.02 * np.cos(theta), (n_phi, n_pol)).copy()
    Z = np.broadcast_to(0.02 * np.sin(theta), (n_phi, n_pol)).copy()
    return ToroidalWall(phi, R, Z)


def _strike_target(branch):
    seed_index = branch.seed_index(7)
    direction_sign = float(np.sign(branch.direction_RZ[0]))
    hit_R = 1.0 + direction_sign * 0.02
    seed_R = branch.anchor_RZ_m[0] + (
        branch.seed_side
        * branch.seed_distances_m[seed_index]
        * branch.direction_RZ[0]
    )
    hit_phi = branch.section_phi + np.log(
        (hit_R - branch.anchor_RZ_m[0])
        / (seed_R - branch.anchor_RZ_m[0])
    ) / RATE
    label = ManifoldStrikeLabel(branch.label, "+", 7)
    match = ManifoldStrikeMatch(
        label=label,
        point_RZPhi_m_rad=np.array([hit_R, 0.0, hit_phi]),
        target_distance_m=0.0,
        connection_length_m=abs(hit_R * (hit_phi - branch.section_phi)),
        bundle_seed_index=seed_index,
        distance_mode="rz",
    )
    plane = LocalWallPlane(
        strike_label=label,
        point_xyz_m=np.array(
            [hit_R * np.cos(hit_phi), hit_R * np.sin(hit_phi), 0.0]
        ),
        normal_xyz=direction_sign
        * np.array([np.cos(hit_phi), np.sin(hit_phi), 0.0]),
        wall_phi_rad=hit_phi,
        wall_s=0.0 if direction_sign > 0.0 else 0.5,
        projection_distance_m=0.0,
    )
    return ManifoldStrikeStage2Target(
        branch_reference=branch,
        strike_match=match,
        wall_plane=plane,
        target_position_m=np.array([hit_R, 0.0]),
        position_scales_m=np.array([0.01, 0.01]),
        maximum_phi_shift=0.2,
        n_steps_per_span=N_STEPS_PER_SPAN,
        xline_newton_iterations=4,
        wall_n_steps=512,
        wall_newton_iterations=6,
    )


def _analytic_strike_trace(parameters, hit_R):
    rate, center_r, _center_z = map(float, parameters)

    def trace(
        field,
        R,
        Z,
        phi_start,
        max_turns,
        DPhi,
        wall_phi,
        wall_R,
        wall_Z,
        *,
        extend_phi,
        direction,
    ):
        del field, max_turns, DPhi, wall_phi, wall_R, wall_Z, extend_phi
        assert direction == "+"
        phi = phi_start + np.log(
            (hit_R - center_r) / (R - center_r)
        ) / rate
        return {
            "Lc_plus": np.abs(hit_R * (phi - phi_start)),
            "hit_plus": np.column_stack((np.full(R.size, hit_R), Z, phi)),
            "term_plus": np.ones(R.size, dtype=int),
        }

    return trace


def _strike_continuation_state():
    state = _continuation_state()
    schedule = ManifoldContinuationSchedule(
        (
            ManifoldContinuationStage(name="base"),
            ManifoldContinuationStage(
                name="manifold-strike",
                manifold_weight=0.1,
                strike_weight=0.2,
            ),
        )
    )
    return ManifoldContinuationState(
        schedule,
        state.target_state,
        strike_target_state=_strike_target(state.target_state.branch_reference),
    )


def _validate(candidate_field, state, **overrides):
    options = {
        "maximum_anchor_displacement_m": 2.0e-3,
        "minimum_direction_alignment": 0.95,
        "maximum_sample_displacement_m": 2.0e-3,
        "jax_cyna_tolerance_m": 5.0e-12,
        "sampling_batch_size": 2048,
        "axisymmetric": True,
        "production_DPhi": MAP_SPAN / N_STEPS_PER_SPAN,
    }
    options.update(overrides)
    return validate_manifold_continuation_candidate(
        candidate_field,
        state,
        R_GRID,
        Z_GRID,
        PHI_GRID,
        **options,
    )


def test_candidate_validation_advances_after_all_three_pyna_gates():
    _require_cyna()
    state = _continuation_state()
    candidate = _ShiftedHyperbolicField(jnp.array([0.36, 1.0003, -0.0002]))
    report = _validate(candidate, state)

    assert report.accepted
    assert report.production_refresh.accepted
    assert report.correspondence.accepted
    assert report.sample_refresh.accepted
    assert report.accepted_state is not None
    assert report.accepted_state.stage_index == 1
    assert report.accepted_state.target_state.label == state.target_state.label
    assert report.candidate_branch.origin_label == "21:P0"
    np.testing.assert_array_equal(report.candidate_branch.seed_orders, SEED_ORDERS)
    assert state.stage_index == 0


def test_candidate_validation_advances_strike_state_only_after_its_gates():
    _require_cyna()
    pytest.importorskip("joblib")
    state = _strike_continuation_state()
    candidate_parameters = np.array([0.36, 1.0003, 0.0])
    candidate = _ShiftedHyperbolicField(jnp.asarray(candidate_parameters))
    hit_R = state.strike_target_state.target_position_m[0]
    strike_config = ManifoldStrikeValidationConfig(
        wall=_strike_wall(),
        maximum_hit_displacement_m=2.0e-3,
        maximum_projection_distance_m=2.0e-4,
        jax_cyna_tolerance_m=2.0e-8,
        max_turns=4,
        production_DPhi=0.01,
        production_trace_function=_analytic_strike_trace(
            candidate_parameters,
            hit_R,
        ),
    )

    with pytest.raises(TypeError, match="requires ManifoldStrikeValidationConfig"):
        _validate(candidate, state)
    report = _validate(
        candidate,
        state,
        strike_validation_config=strike_config,
    )

    assert report.accepted
    assert report.strike_validation.accepted
    assert report.accepted_state is not None
    assert report.accepted_state.stage_index == 1
    assert report.accepted_state.strike_target_state is not None
    assert (
        report.accepted_state.strike_target_state.label
        == state.strike_target_state.label
    )
    assert (
        report.accepted_state.strike_target_state.strike_match
        is report.strike_validation.strike_refresh.candidate_match
    )
    assert report.diagnostics["jax_cyna_strike_deviation_m"] < 2.0e-8


def test_candidate_validation_stops_after_production_anchor_rejection():
    _require_cyna()
    state = _continuation_state()
    candidate = _ShiftedHyperbolicField(jnp.array([0.36, 1.01, 0.0]))
    report = _validate(
        candidate,
        state,
        maximum_anchor_displacement_m=1.0e-3,
    )

    assert not report.accepted
    assert report.rejection_reason.endswith("anchor_displacement_exceeds_limit")
    assert report.correspondence is None
    assert report.sample_refresh is None
    assert report.accepted_state is None


def test_candidate_validation_rejects_exact_sample_motion_without_relabelling():
    _require_cyna()
    state = _continuation_state()
    candidate = _ShiftedHyperbolicField(jnp.array([0.36, 1.0003, -0.0002]))
    report = _validate(
        candidate,
        state,
        maximum_sample_displacement_m=1.0e-8,
    )

    assert not report.accepted
    assert report.production_refresh.accepted
    assert report.correspondence.accepted
    assert not report.sample_refresh.accepted
    assert report.rejection_reason.endswith("sample_displacement_exceeds_limit")
    assert report.accepted_state is None


def test_candidate_validation_can_require_a_complete_production_branch():
    _require_cyna()
    state = _continuation_state(generation_index=1)
    candidate = _ShiftedHyperbolicField(jnp.array([RATE, 1.0, 0.0]))
    report = _validate(
        candidate,
        state,
        RZlimit=(0.99, 1.0025, -0.1, 0.1),
        require_complete_correspondence=True,
    )

    assert not report.accepted
    assert report.production_refresh.accepted
    assert not report.correspondence.complete
    assert not report.correspondence.accepted
    assert report.rejection_reason == "jax_cyna_correspondence_failed"
    assert report.accepted_state is None


def test_backtracking_contracts_an_optimizer_proposal_to_an_accepted_state():
    _require_cyna()
    state = _continuation_state()
    current = np.array([RATE, 1.0, 0.0])
    proposed = np.array([0.36, 1.01, 0.0])

    result = validated_manifold_backtracking_step(
        current,
        proposed,
        lambda dofs: _ShiftedHyperbolicField(jnp.asarray(dofs)),
        state,
        lambda field, accepted_state: _validate(
            field,
            accepted_state,
            maximum_anchor_displacement_m=2.0e-3,
        ),
        contraction=0.25,
        maximum_attempts=4,
    )

    assert result.accepted
    np.testing.assert_allclose(
        [attempt.step_fraction for attempt in result.attempts],
        [1.0, 0.25, 0.0625],
    )
    assert not result.attempts[0].validation.accepted
    assert not result.attempts[1].validation.accepted
    assert result.attempts[2].validation.accepted
    np.testing.assert_allclose(
        result.dofs,
        current + 0.0625 * (proposed - current),
    )
    np.testing.assert_allclose(result.field.parameters, result.dofs)
    assert result.continuation_state.stage_index == 1
    assert state.stage_index == 0


def test_backtracking_returns_the_unchanged_field_when_no_attempt_accepts():
    _require_cyna()
    state = _continuation_state()
    current = np.array([RATE, 1.0, 0.0])
    proposed = np.array([0.36, 1.1, 0.0])

    result = validated_manifold_backtracking_step(
        current,
        proposed,
        lambda dofs: _ShiftedHyperbolicField(jnp.asarray(dofs)),
        state,
        lambda field, accepted_state: _validate(
            field,
            accepted_state,
            maximum_anchor_displacement_m=2.0e-3,
        ),
        contraction=0.5,
        maximum_attempts=2,
    )

    assert not result.accepted
    np.testing.assert_allclose(result.dofs, current)
    np.testing.assert_allclose(result.field.parameters, current)
    assert result.continuation_state is state
    assert result.step_fraction == 0.0
    assert len(result.attempts) == 2
    assert all(not attempt.validation.accepted for attempt in result.attempts)
