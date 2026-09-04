"""Physical ESSOS coil-field integration for PyNA manifold sensitivities."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.topo.jax_manifold")

from pyna.topo.jax_manifold import (
    fundamental_segment_distances,
    hyperbolic_directions,
)
from pyna.topo.manifold_correspondence import (
    ManifoldSampleMatch,
    manifold_branch_reference_from_trace,
)
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.flt import (
    refine_fixed_points_monodromy_span_field,
    trace_fixed_point_manifolds_field,
)

from essos.coils import Coils, Curves
from essos.fields import BiotSavart
from essos.manifold_driver import validated_manifold_backtracking_step
from essos.manifold import (
    essos_field_to_pyna_cylindrical_grid,
    periodic_xline_state,
    trace_manifold_branch,
)
from essos.manifold_optimization import (
    ManifoldContinuationSchedule,
    ManifoldContinuationStage,
    ManifoldContinuationState,
    ManifoldStage2Target,
    manifold_stage2_target_loss,
    trace_manifold_reference,
)
from essos.manifold_validation import validate_manifold_continuation_candidate


N_TF = 8
N_SEGMENTS = 64
MAJOR_RADIUS = 1.4
TF_RADIUS = 0.9
TF_CURRENT = -8.0e5
PLASMA_CURRENT = 1.0e6
PF_CURRENT = 1.0e6
MAP_SPAN = 2.0 * jnp.pi / N_TF
INITIAL_GUESS = jnp.array([1.48, -0.092])
PRODUCTION_R = np.linspace(0.8, 2.4, 49)
PRODUCTION_Z = np.linspace(-1.1, 0.8, 49)
PRODUCTION_PHI = np.linspace(0.0, float(MAP_SPAN), 12, endpoint=False)


def _vertical_tf_dofs(phi):
    radial = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
    dofs = jnp.zeros((3, 3))
    dofs = dofs.at[:, 0].set(MAJOR_RADIUS * radial)
    dofs = dofs.at[:, 2].set(TF_RADIUS * radial)
    return dofs.at[2, 1].set(TF_RADIUS)


def _horizontal_loop_dofs(radius, height):
    dofs = jnp.zeros((3, 3))
    dofs = dofs.at[0, 2].set(radius)
    dofs = dofs.at[1, 1].set(radius)
    return dofs.at[2, 0].set(height)


COIL_DOFS = jnp.stack(
    [
        _vertical_tf_dofs(2.0 * jnp.pi * index / N_TF)
        for index in range(N_TF)
    ]
    + [
        _horizontal_loop_dofs(1.0, 0.0),
        _horizontal_loop_dofs(3.0, -1.2),
    ]
)
CURVES = Curves(
    COIL_DOFS,
    n_segments=N_SEGMENTS,
    nfp=1,
    stellsym=False,
)


def _physical_field(pf_control):
    currents = jnp.concatenate(
        (
            TF_CURRENT * jnp.ones(N_TF),
            jnp.array([PLASMA_CURRENT, PF_CURRENT * pf_control]),
        )
    )
    return BiotSavart(Coils(CURVES, currents, currents_scale=PF_CURRENT))


def _require_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind is unavailable")


def _tokamak_production_branch(field):
    xline = periodic_xline_state(
        field,
        INITIAL_GUESS,
        phi_span=MAP_SPAN,
        n_steps_per_span=64,
        newton_iterations=6,
        bphi_floor=1.0e-8,
    )
    production_field = essos_field_to_pyna_cylindrical_grid(
        field,
        PRODUCTION_R,
        PRODUCTION_Z,
        PRODUCTION_PHI,
        nfp=N_TF,
        batch_size=512,
    )
    fixed_point = FixedPoint(
        phi=0.0,
        R=float(xline.position[0]),
        Z=float(xline.position[1]),
        kind="X",
        DPm=np.asarray(xline.monodromy),
    )
    fixed_point.map_power = 1
    fixed_point.metadata.update(
        {
            "orbit_id": 1,
            "map_order_index": 0,
            "monodromy_map_span": float(MAP_SPAN),
        }
    )
    production_xline = refine_fixed_points_monodromy_span_field(
        production_field,
        [fixed_point],
        field_period=float(MAP_SPAN),
        map_power=1,
        DPhi=float(MAP_SPAN) / 64,
        keep_unconverged=False,
    )[0]
    payload = trace_fixed_point_manifolds_field(
        production_field,
        [production_xline],
        phi_section=0.0,
        map_span=float(MAP_SPAN),
        N_turns=2,
        DPhi=float(MAP_SPAN) / 64,
        seed_distances=np.array([1.0e-5, 3.0e-5]),
        seed_orders=np.array([2, 5]),
        refine_stable_inverse_anchor=False,
    )[0]
    return manifold_branch_reference_from_trace(
        payload,
        stability="unstable",
        seed_side=1,
        n_generations=2,
    )


def _tokamak_continuation_state(
    branch,
    *,
    target_RZ_m=None,
    rz_scales_m=(1.0e-3, 1.0e-3),
    stage_index=0,
):
    label = branch.sample_label(2, 0)
    point = branch.point(label)
    jax_seed_index = branch.seed_index(label.seed_order)
    match = ManifoldSampleMatch(
        label=label,
        point_RZ_m=point,
        distance_m=0.0,
        jax_seed_index=jax_seed_index,
    )
    target = ManifoldStage2Target(
        branch_reference=branch,
        sample_match=match,
        target_RZ_m=point if target_RZ_m is None else target_RZ_m,
        rz_scales_m=np.asarray(rz_scales_m),
        n_steps_per_span=64,
        newton_iterations=6,
        bphi_floor=1.0e-8,
    )
    schedule = ManifoldContinuationSchedule(
        (
            ManifoldContinuationStage(name="base"),
            ManifoldContinuationStage(name="manifold", manifold_weight=0.1),
        )
    )
    return ManifoldContinuationState(
        schedule,
        target,
        stage_index=stage_index,
    )


def _validate_tokamak_candidate(field, continuation):
    return validate_manifold_continuation_candidate(
        field,
        continuation,
        PRODUCTION_R,
        PRODUCTION_Z,
        PRODUCTION_PHI,
        nfp=N_TF,
        sampling_batch_size=512,
        maximum_anchor_displacement_m=2.0e-3,
        minimum_direction_alignment=0.9,
        maximum_sample_displacement_m=2.0e-3,
        jax_cyna_tolerance_m=2.0e-4,
        production_DPhi=float(MAP_SPAN) / 64,
        require_complete_correspondence=True,
    )


def test_coil_current_moves_a_hyperbolic_xline_and_its_manifold():
    nominal_field = _physical_field(1.0)
    state = periodic_xline_state(
        nominal_field,
        INITIAL_GUESS,
        phi_span=MAP_SPAN,
        n_steps_per_span=64,
        newton_iterations=6,
        bphi_floor=1.0e-8,
    )
    directions = hyperbolic_directions(
        state.monodromy,
        stable_reference=jnp.array([0.0, -1.0]),
        unstable_reference=jnp.array([1.0, 0.0]),
    )

    assert bool(state.converged)
    assert float(state.residual_norm) < 1.0e-11
    assert float(state.hyperbolicity_margin) > 0.5
    np.testing.assert_allclose(state.determinant, 1.0, atol=1.0e-8)
    assert bool(directions.valid)

    unstable_distances = fundamental_segment_distances(
        directions.unstable_expansion,
        minimum_distance=1.0e-5,
        maximum_distance=1.0e-3,
        n_seeds=4,
    )
    unstable = trace_manifold_branch(
        nominal_field,
        state.position,
        directions.unstable_direction,
        unstable_distances,
        stability="unstable",
        side=1,
        phi_span=MAP_SPAN,
        n_generations=3,
        n_steps_per_span=64,
        newton_iterations=6,
        bphi_floor=1.0e-8,
    )
    stable_distances = fundamental_segment_distances(
        directions.stable_backward_expansion,
        minimum_distance=1.0e-5,
        maximum_distance=1.0e-3,
        n_seeds=4,
    )
    stable = trace_manifold_branch(
        nominal_field,
        state.position,
        directions.stable_direction,
        stable_distances,
        stability="stable",
        side=1,
        phi_span=MAP_SPAN,
        n_generations=3,
        n_steps_per_span=64,
        newton_iterations=6,
        bphi_floor=1.0e-8,
    )

    unstable_growth = jnp.linalg.norm(
        unstable.generations[-1, 0] - unstable.anchor
    ) / jnp.linalg.norm(unstable.generations[0, 0] - unstable.anchor)
    stable_growth = jnp.linalg.norm(
        stable.generations[-1, 0] - stable.anchor
    ) / jnp.linalg.norm(stable.generations[0, 0] - stable.anchor)
    assert float(unstable_growth) > 10.0
    assert float(stable_growth) > 10.0

    frozen_direction = directions.unstable_direction

    def endpoint(pf_control):
        trace = trace_manifold_branch(
            _physical_field(pf_control),
            INITIAL_GUESS,
            frozen_direction,
            jnp.array([2.0e-5]),
            stability="unstable",
            side=1,
            phi_span=MAP_SPAN,
            n_generations=2,
            n_steps_per_span=64,
            newton_iterations=6,
            bphi_floor=1.0e-8,
        )
        return trace.generations[-1, 0]

    forward = jax.jacfwd(endpoint)(1.0)
    reverse = jax.jacrev(endpoint)(1.0)
    delta = 1.0e-4
    finite_difference = (endpoint(1.0 + delta) - endpoint(1.0 - delta)) / (
        2.0 * delta
    )

    assert bool(jnp.all(jnp.isfinite(forward)))
    assert float(jnp.linalg.norm(forward)) > 0.0
    np.testing.assert_allclose(reverse, forward, atol=2.0e-11)
    np.testing.assert_allclose(
        forward,
        finite_difference,
        rtol=3.0e-7,
        atol=3.0e-10,
    )


def test_tokamak_pf_coil_candidate_passes_full_outer_validation():
    _require_cyna()
    nominal_field = _physical_field(1.0)
    branch = _tokamak_production_branch(nominal_field)
    continuation = _tokamak_continuation_state(branch)
    target = continuation.target_state

    report = _validate_tokamak_candidate(_physical_field(1.0002), continuation)

    assert report.accepted
    assert report.accepted_state is not None
    assert report.accepted_state.stage_index == 1
    assert report.accepted_state.target_state.label == target.label
    assert report.correspondence.max_deviation_m < 2.0e-4
    assert report.production_refresh.anchor_displacement_m < 2.0e-3


def test_bounded_pf_current_step_reaches_a_manifold_target():
    _require_cyna()
    branch = _tokamak_production_branch(_physical_field(1.0))
    desired_control = 1.001
    desired_trace = trace_manifold_reference(
        _physical_field(desired_control),
        branch,
        n_steps_per_span=64,
        newton_iterations=6,
        bphi_floor=1.0e-8,
    )
    label = branch.sample_label(2, 0)
    jax_seed_index = branch.seed_index(label.seed_order)
    target_RZ_m = np.asarray(
        desired_trace.generations[label.generation_index, jax_seed_index]
    )
    rz_scales_m = np.array([1.0e-4, 1.0e-4])
    continuation = _tokamak_continuation_state(
        branch,
        target_RZ_m=target_RZ_m,
        rz_scales_m=rz_scales_m,
        stage_index=1,
    )

    def endpoint(control):
        trace = trace_manifold_reference(
            _physical_field(control),
            branch,
            n_steps_per_span=64,
            newton_iterations=6,
            bphi_floor=1.0e-8,
        )
        return trace.generations[label.generation_index, jax_seed_index]

    initial_control = jnp.asarray(1.0)
    initial_endpoint = endpoint(initial_control)
    normalized_residual = (initial_endpoint - target_RZ_m) / rz_scales_m
    normalized_jacobian = jax.jacfwd(endpoint)(initial_control) / rz_scales_m
    gauss_newton_step = -jnp.vdot(
        normalized_jacobian,
        normalized_residual,
    ) / jnp.vdot(normalized_jacobian, normalized_jacobian)
    proposed_control = np.clip(
        float(initial_control + gauss_newton_step),
        0.999,
        1.0015,
    )
    initial_loss = manifold_stage2_target_loss(
        _physical_field(initial_control),
        target_state=continuation.target_state,
    )

    result = validated_manifold_backtracking_step(
        np.array([float(initial_control)]),
        np.array([proposed_control]),
        lambda dofs: _physical_field(dofs[0]),
        continuation,
        _validate_tokamak_candidate,
        contraction=0.5,
        maximum_attempts=5,
    )
    final_loss = manifold_stage2_target_loss(
        result.field,
        target_state=continuation.target_state,
    )

    assert result.accepted
    assert result.step_fraction == 1.0
    assert abs(result.dofs[0] - desired_control) < 2.0e-6
    assert float(final_loss) < 1.0e-5 * float(initial_loss)
    assert result.continuation_state.stage_index == 1
    assert result.continuation_state.target_state.label == label.key
    assert result.attempts[-1].validation.correspondence.max_deviation_m < 2.0e-4
