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
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.flt import trace_fixed_point_manifolds_field

from essos.manifold import essos_field_to_pyna_cylindrical_grid
from essos.manifold_optimization import (
    ManifoldContinuationSchedule,
    ManifoldContinuationStage,
    ManifoldContinuationState,
    ManifoldStage2Target,
)
from essos.manifold_validation import (
    validate_manifold_continuation_candidate,
)


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
