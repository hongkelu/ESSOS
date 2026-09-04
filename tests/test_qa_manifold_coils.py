"""Physical QA-coil checkpoint for manifold-aware Stage-2 targets.

The magnetic field comes from the repository's Landreman--Paul QA modular
coils plus a weak, explicit toroidal-field trim set.  The shaped wall is a
frozen regression geometry, not a proposed reactor wall: it encloses all five
components of the reference X-line and presents one non-axisymmetric divertor
notch to a labelled unstable branch.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.topo.jax_manifold")
pytest.importorskip("pyna.topo.jax_strike")

from pyna.topo.manifold_correspondence import (
    ManifoldSampleMatch,
    compare_jax_manifold_branch,
    manifold_branch_reference_from_trace,
)
from pyna.topo.manifold_strike import (
    local_wall_plane_from_strike,
    manifold_branch_strike_seed_bundle,
    select_manifold_strike_match,
)
from pyna.topo.manifold_strike_correspondence import (
    compare_jax_manifold_strike,
)
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.control.strike_heat import trace_wall_strikes_field
from pyna.toroidal.flt import (
    refine_fixed_points_monodromy_span_field,
    trace_fixed_point_manifolds_field,
    trace_orbit_along_phi_field,
    trace_wall_hits_twall_field,
)
from pyna.toroidal.geometry import ToroidalWall

from essos.coils import Coils, Curves
from essos.fields import BiotSavart
from essos.manifold_driver import validated_manifold_backtracking_step
from essos.manifold import (
    essos_field_to_pyna_cylindrical_grid,
    fixed_phi_poincare_map_from_coils,
    periodic_xline_clearance_loss,
    periodic_xline_state,
)
from essos.manifold_optimization import (
    ManifoldContinuationSchedule,
    ManifoldContinuationStage,
    ManifoldContinuationState,
    ManifoldStage2Target,
    trace_manifold_reference,
)
from essos.manifold_heat_optimization import (
    ManifoldHeatStage2Target,
    manifold_heat_flux_state,
)
from essos.manifold_strike_optimization import (
    ManifoldStrikeStage2Target,
    manifold_strike_stage2_target_loss,
    trace_manifold_strike_reference,
)
from essos.manifold_strike_validation import ManifoldStrikeValidationConfig
from essos.manifold_validation import (
    ManifoldHeatValidationConfig,
    XLineClearanceValidationConfig,
    validate_manifold_continuation_candidate,
)


QA_NFP = 2
QA_N_TF_TRIM = 16
QA_TRIM_CURRENT_KA = 0.10299325375395912
QA_FIELD_PERIOD = np.pi
QA_INITIAL_GUESS = np.array([1.0904618121104814, -0.14])
QA_R_GRID = np.linspace(0.42, 1.48, 257)
QA_Z_GRID = np.linspace(-0.65, 0.65, 313)
QA_PHI_GRID = np.linspace(0.0, QA_FIELD_PERIOD, 160, endpoint=False)
QA_PRODUCTION_DPHI = QA_FIELD_PERIOD / 320
QA_N_WALL_PHI = 1280
QA_WALL_DPHI = QA_FIELD_PERIOD / QA_N_WALL_PHI


def _vertical_tf_trim_dofs(phi):
    radial = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
    dofs = jnp.zeros((3, 11))
    dofs = dofs.at[:, 0].set(1.2 * radial)
    dofs = dofs.at[:, 2].set(1.0 * radial)
    return dofs.at[2, 1].set(1.0)


_QA_INPUT = Path(__file__).parents[1] / (
    "examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"
)
_QA_SOURCE_COILS = Coils.from_json(str(_QA_INPUT))
_QA_SOURCE_CURRENTS = _QA_SOURCE_COILS.currents
_QA_BASE_DOFS = _QA_SOURCE_COILS.curves.dofs
QA_SHAPE_DOF_INDEX = (0, 2, 1)
QA_SHAPE_DOF_NOMINAL = float(_QA_BASE_DOFS[QA_SHAPE_DOF_INDEX])
_QA_TRIM_DOFS = jnp.stack(
    [
        _vertical_tf_trim_dofs(2.0 * jnp.pi * index / QA_N_TF_TRIM)
        for index in range(QA_N_TF_TRIM)
    ]
)
_QA_CURVES = Curves(
    jnp.concatenate((_QA_SOURCE_COILS.curves.curves, _QA_TRIM_DOFS)),
    n_segments=_QA_SOURCE_COILS.n_segments,
    nfp=1,
    stellsym=False,
)


def _qa_coils(trim_current_kA):
    currents = jnp.concatenate(
        (
            _QA_SOURCE_CURRENTS,
            1000.0
            * jnp.asarray(trim_current_kA)
            * jnp.ones(QA_N_TF_TRIM),
        )
    )
    return Coils(_QA_CURVES, currents, currents_scale=1.0e5)


def _qa_field(trim_current_kA=QA_TRIM_CURRENT_KA):
    return BiotSavart(_qa_coils(trim_current_kA))


def _qa_shape_field(shape_dof):
    base_dofs = _QA_BASE_DOFS.at[QA_SHAPE_DOF_INDEX].set(shape_dof)
    symmetric_qa_curves = Curves(
        base_dofs,
        n_segments=_QA_SOURCE_COILS.n_segments,
        nfp=QA_NFP,
        stellsym=True,
    )
    curves = Curves(
        jnp.concatenate((symmetric_qa_curves.curves, _QA_TRIM_DOFS)),
        n_segments=_QA_SOURCE_COILS.n_segments,
        nfp=1,
        stellsym=False,
    )
    currents = jnp.concatenate(
        (
            _QA_SOURCE_CURRENTS,
            1000.0
            * QA_TRIM_CURRENT_KA
            * jnp.ones(QA_N_TF_TRIM),
        )
    )
    return BiotSavart(
        Coils(curves, currents, currents_scale=1.0e5)
    )


def _require_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind is unavailable")


def _production_xline(field, live_xline):
    production_field = essos_field_to_pyna_cylindrical_grid(
        field,
        QA_R_GRID,
        QA_Z_GRID,
        QA_PHI_GRID,
        nfp=QA_NFP,
        batch_size=1024,
    )
    fixed_point = FixedPoint(
        phi=0.0,
        R=float(live_xline.position[0]),
        Z=float(live_xline.position[1]),
        kind="X",
        DPm=np.asarray(live_xline.monodromy),
    )
    fixed_point.map_power = 5
    fixed_point.metadata.update(
        {
            "orbit_id": 1,
            "map_order_index": 0,
            "monodromy_map_span": 5.0 * QA_FIELD_PERIOD,
            "field_period": QA_FIELD_PERIOD,
        }
    )
    production_xline = refine_fixed_points_monodromy_span_field(
        production_field,
        [fixed_point],
        field_period=QA_FIELD_PERIOD,
        map_power=5,
        DPhi=QA_PRODUCTION_DPHI,
        fd_eps=1.0e-5,
        max_iter=120,
        tol=1.0e-12,
        residual_tol=1.0e-8,
        keep_unconverged=False,
        n_threads=1,
    )[0]
    return production_field, production_xline


def _qa_divertor_wall(production_field, production_xline):
    orbit = trace_orbit_along_phi_field(
        production_field,
        production_xline.R,
        production_xline.Z,
        0.0,
        5.0 * QA_FIELD_PERIOD,
        QA_WALL_DPHI,
        dphi_out=QA_WALL_DPHI,
    )
    orbit_R = np.asarray(orbit[0])
    orbit_Z = np.asarray(orbit[1])
    wall_phi = np.linspace(
        0.0,
        QA_FIELD_PERIOD,
        QA_N_WALL_PHI,
        endpoint=False,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
    center_R = 0.95
    center_Z = 0.0
    ellipse_R = 0.75
    ellipse_Z = 0.80
    base_radius = 1.0 / np.sqrt(
        (np.cos(theta) / ellipse_R) ** 2
        + (np.sin(theta) / ellipse_Z) ** 2
    )
    wall_radius = np.broadcast_to(
        base_radius,
        (wall_phi.size, theta.size),
    ).copy()

    # A smooth, toroidally localized notch follows one X-line component while
    # its clearance is reduced from 25 mm to 5 mm.  It is repeated by nfp=2.
    for index, phi in enumerate(wall_phi):
        if phi <= 0.15 or phi >= 0.85:
            continue
        delta_R = orbit_R[index] - center_R
        delta_Z = orbit_Z[index] - center_Z
        orbit_radius = np.hypot(delta_R, delta_Z)
        orbit_theta = np.arctan2(delta_Z, delta_R)
        clearance_fraction = np.clip((phi - 0.25) / 0.35, 0.0, 1.0)
        clearance = 0.025 - 0.020 * clearance_fraction
        if phi < 0.25:
            phase_window = 0.5 * (
                1.0 - np.cos(np.pi * (phi - 0.15) / 0.10)
            )
        elif phi <= 0.70:
            phase_window = 1.0
        else:
            phase_window = 0.5 * (
                1.0 + np.cos(np.pi * (phi - 0.70) / 0.15)
            )
        angular_delta = np.arctan2(
            np.sin(theta - orbit_theta),
            np.cos(theta - orbit_theta),
        )
        notch = np.exp(-0.5 * (angular_delta / 0.045) ** 2)
        target_radius = orbit_radius + clearance
        wall_radius[index] -= (
            phase_window * notch * (base_radius - target_radius)
        )

    # The period-five X-line has five simultaneous points in every physical
    # field-period sector.  Preserve at least 8 mm around every component so
    # the wall selects a manifold strike rather than clipping the X-line.
    for index in range(wall_phi.size):
        for orbit_number in range(5):
            orbit_index = orbit_number * QA_N_WALL_PHI + index
            delta_R = orbit_R[orbit_index] - center_R
            delta_Z = orbit_Z[orbit_index] - center_Z
            orbit_radius = np.hypot(delta_R, delta_Z)
            orbit_theta = np.arctan2(delta_Z, delta_R)
            angular_delta = np.arctan2(
                np.sin(theta - orbit_theta),
                np.cos(theta - orbit_theta),
            )
            protection = (orbit_radius + 0.008) * np.exp(
                -0.5 * (angular_delta / 0.025) ** 2
            )
            wall_radius[index] = np.maximum(
                wall_radius[index],
                protection,
            )

    return ToroidalWall(
        wall_phi,
        center_R + wall_radius * np.cos(theta)[None, :],
        center_Z + wall_radius * np.sin(theta)[None, :],
        nfp=QA_NFP,
        name="QA divertor-notch regression wall",
    )


def _qa_wall_radial_signed_distance(wall):
    """Freeze the regression wall as a JAX periodic radial interpolant."""

    center_R = 0.95
    center_Z = 0.0
    wall_radius = jnp.asarray(
        np.hypot(np.asarray(wall.R) - center_R, np.asarray(wall.Z) - center_Z)
    )
    n_phi, n_theta = wall_radius.shape
    phi0 = float(np.asarray(wall.phi)[0])

    def signed_distance(point_xyz):
        radius = jnp.hypot(point_xyz[0], point_xyz[1])
        phi = jnp.arctan2(point_xyz[1], point_xyz[0])
        radial_R = radius - center_R
        radial_Z = point_xyz[2] - center_Z
        rho = jnp.hypot(radial_R, radial_Z)
        theta = jnp.mod(jnp.arctan2(radial_Z, radial_R), 2.0 * jnp.pi)

        phi_coordinate = (
            jnp.mod(phi - phi0, QA_FIELD_PERIOD)
            * n_phi
            / QA_FIELD_PERIOD
        )
        theta_coordinate = theta * n_theta / (2.0 * jnp.pi)
        phi_lower = jnp.floor(phi_coordinate).astype(jnp.int32) % n_phi
        theta_lower = jnp.floor(theta_coordinate).astype(jnp.int32) % n_theta
        phi_upper = (phi_lower + 1) % n_phi
        theta_upper = (theta_lower + 1) % n_theta
        phi_fraction = phi_coordinate - jnp.floor(phi_coordinate)
        theta_fraction = theta_coordinate - jnp.floor(theta_coordinate)

        lower_radius = (
            (1.0 - theta_fraction) * wall_radius[phi_lower, theta_lower]
            + theta_fraction * wall_radius[phi_lower, theta_upper]
        )
        upper_radius = (
            (1.0 - theta_fraction) * wall_radius[phi_upper, theta_lower]
            + theta_fraction * wall_radius[phi_upper, theta_upper]
        )
        interpolated_radius = (
            (1.0 - phi_fraction) * lower_radius
            + phi_fraction * upper_radius
        )
        return interpolated_radius - rho

    return signed_distance


def test_qa_period_five_xline_has_differentiable_multi_turn_strike():
    _require_cyna()
    pytest.importorskip("joblib")
    field = _qa_field()
    live_xline = periodic_xline_state(
        field,
        QA_INITIAL_GUESS,
        phi_span=QA_FIELD_PERIOD,
        map_power=5,
        n_steps_per_span=128,
        newton_iterations=8,
        newton_damping=0.8,
        bphi_floor=1.0e-10,
        residual_tolerance=1.0e-8,
    )
    one_period_displacement = (
        fixed_phi_poincare_map_from_coils(
            field.coils,
            live_xline.position,
            phi_span=QA_FIELD_PERIOD,
            n_steps=128,
            bphi_floor=1.0e-10,
        )
        - live_xline.position
    )

    assert _QA_SOURCE_COILS.nfp == QA_NFP
    assert _QA_SOURCE_COILS.stellsym
    assert bool(live_xline.converged)
    assert float(live_xline.residual_norm) < 1.0e-10
    assert float(live_xline.trace) > 2.005
    assert float(live_xline.hyperbolicity_margin) > 5.0e-3
    np.testing.assert_allclose(live_xline.determinant, 1.0, atol=1.0e-4)
    assert float(jnp.linalg.norm(one_period_displacement)) > 0.1

    production_field, production_xline = _production_xline(
        field,
        live_xline,
    )
    np.testing.assert_allclose(
        [production_xline.R, production_xline.Z],
        live_xline.position,
        atol=2.0e-4,
    )
    assert production_xline.residual < 1.0e-10
    assert np.trace(production_xline.DPm) > 2.005
    np.testing.assert_allclose(
        np.linalg.det(production_xline.DPm),
        1.0,
        atol=5.0e-3,
    )

    wall = _qa_divertor_wall(production_field, production_xline)
    assert wall.nfp == QA_NFP
    assert not np.allclose(wall.R[0], wall.R[100])
    root_hits = trace_wall_hits_twall_field(
        production_field,
        np.array([production_xline.R]),
        np.array([production_xline.Z]),
        0.0,
        5,
        QA_WALL_DPHI,
        wall,
    )
    np.testing.assert_array_equal(root_hits["term_plus"], [0])
    np.testing.assert_array_equal(root_hits["term_minus"], [0])

    payload = trace_fixed_point_manifolds_field(
        production_field,
        [production_xline],
        phi_section=0.0,
        map_span=5.0 * QA_FIELD_PERIOD,
        N_turns=1,
        DPhi=QA_PRODUCTION_DPHI,
        seed_distances=np.array([0.03]),
        seed_orders=np.array([30]),
        RZlimit=(
            QA_R_GRID[0],
            QA_R_GRID[-1],
            QA_Z_GRID[0],
            QA_Z_GRID[-1],
        ),
        refine_stable_inverse_anchor=False,
    )[0]
    branch = manifold_branch_reference_from_trace(
        payload,
        stability="unstable",
        seed_side=1,
        n_generations=1,
    )
    live_branch = trace_manifold_reference(
        field,
        branch,
        n_steps_per_span=640,
        newton_iterations=8,
        newton_damping=0.8,
        bphi_floor=1.0e-10,
    )
    branch_comparison = compare_jax_manifold_branch(
        branch,
        live_branch.generations,
        absolute_tolerance_m=2.0e-4,
        require_complete=True,
    )
    assert branch_comparison.accepted

    bundle = manifold_branch_strike_seed_bundle(branch)
    strikes = trace_wall_strikes_field(
        production_field,
        (bundle,),
        wall,
        max_turns=3,
        DPhi=QA_WALL_DPHI,
    )[0]
    np.testing.assert_array_equal(strikes.seed_index, [0])
    assert strikes.unresolved_weight == pytest.approx(0.0)
    assert 2.0 * np.pi < strikes.phi[0] < 2.0 * np.pi + 0.85
    assert strikes.connection_length[0] > 1.0

    target_xyz = np.array(
        [
            strikes.R[0] * np.cos(strikes.phi[0]),
            strikes.R[0] * np.sin(strikes.phi[0]),
            strikes.Z[0],
        ]
    )
    match = select_manifold_strike_match(
        bundle,
        strikes,
        target_xyz,
        distance_mode="xyz",
    )
    wall_plane = local_wall_plane_from_strike(
        match,
        wall,
        maximum_projection_distance_m=4.0e-3,
    )
    assert match.label.seed_order == 30
    assert match.distance_mode == "xyz"
    assert wall_plane.wall_phi_rad > 2.0 * np.pi
    assert wall_plane.projection_distance_m < 4.0e-3

    target = ManifoldStrikeStage2Target(
        branch_reference=branch,
        strike_match=match,
        wall_plane=wall_plane,
        target_position_m=match.point_xyz_m,
        position_scales_m=np.full(3, 1.0e-3),
        maximum_phi_shift=0.1,
        n_steps_per_span=640,
        xline_newton_iterations=8,
        xline_newton_damping=0.8,
        wall_n_steps=1024,
        wall_newton_iterations=8,
        wall_newton_damping=1.0,
        bphi_floor=1.0e-10,
        xline_residual_tolerance=1.0e-8,
        wall_residual_tolerance=1.0e-8,
    )
    live_strike = trace_manifold_strike_reference(field, target)
    strike_comparison = compare_jax_manifold_strike(
        match,
        live_strike,
        absolute_tolerance_m=1.5e-3,
    )
    assert strike_comparison.accepted
    assert bool(live_strike.intersection.converged)
    assert abs(float(live_strike.intersection.transversality)) > 1.0
    assert abs(float(live_strike.intersection.phi_shift_from_guess)) < 1.0e-3

    desired_shape_dof = QA_SHAPE_DOF_NOMINAL + 1.0e-5
    desired_trace = trace_manifold_strike_reference(
        _qa_shape_field(desired_shape_dof),
        target,
    )
    assert bool(desired_trace.periodic_point.converged)
    assert bool(desired_trace.intersection.converged)
    shape_target = replace(
        target,
        target_position_m=np.asarray(desired_trace.intersection.point_xyz),
        position_scales_m=np.full(3, 1.0e-4),
    )
    wall_normal = wall_plane.normal_xyz
    toroidal_tangent = np.array(
        [-np.sin(match.point_RZPhi_m_rad[2]), np.cos(match.point_RZPhi_m_rad[2]), 0.0]
    )
    toroidal_tangent -= np.dot(toroidal_tangent, wall_normal) * wall_normal
    toroidal_tangent /= np.linalg.norm(toroidal_tangent)
    poloidal_tangent = np.cross(wall_normal, toroidal_tangent)
    heat_monitor_centers = np.stack(
        (
            match.point_xyz_m,
            match.point_xyz_m + 5.0e-3 * toroidal_tangent,
            match.point_xyz_m - 5.0e-3 * toroidal_tangent,
            match.point_xyz_m + 5.0e-3 * poloidal_tangent,
            match.point_xyz_m - 5.0e-3 * poloidal_tangent,
        )
    )
    heat_target = ManifoldHeatStage2Target(
        branch_reference=branch,
        strike_matches=(match,),
        wall_planes=(wall_plane,),
        strike_powers_W=np.array([1.0]),
        power_provenance="prescribed 1 W QA regression; not a transport prediction",
        wall_cell_centers_xyz_m=heat_monitor_centers,
        wall_cell_areas_m2=np.full(heat_monitor_centers.shape[0], 2.5e-5),
        deposition_width_m=5.0e-3,
        maximum_heat_flux_W_m2=1.0e5,
        heat_flux_scale_W_m2=1.0e5,
        maximum_phi_shift=0.1,
        n_steps_per_span=640,
        xline_newton_iterations=8,
        xline_newton_damping=0.8,
        wall_n_steps=1024,
        wall_newton_iterations=8,
        wall_newton_damping=1.0,
        bphi_floor=1.0e-10,
        xline_residual_tolerance=1.0e-8,
        wall_residual_tolerance=1.0e-8,
    )

    sample_label = branch.sample_label(branch.n_generations, 0)
    sample_match = ManifoldSampleMatch(
        label=sample_label,
        point_RZ_m=branch.point(sample_label),
        distance_m=0.0,
        jax_seed_index=branch.seed_index(sample_label.seed_order),
    )
    sample_target = ManifoldStage2Target(
        branch_reference=branch,
        sample_match=sample_match,
        target_RZ_m=sample_match.point_RZ_m,
        rz_scales_m=np.full(2, 1.0e-3),
        n_steps_per_span=640,
        newton_iterations=8,
        newton_damping=0.8,
        bphi_floor=1.0e-10,
    )
    continuation = ManifoldContinuationState(
        ManifoldContinuationSchedule(
            (
                ManifoldContinuationStage(name="base"),
                ManifoldContinuationStage(
                    name="strike",
                    strike_weight=1.0,
                    xline_clearance_weight=0.1,
                    heat_weight=0.1,
                ),
            )
        ),
        sample_target,
        stage_index=1,
        strike_target_state=shape_target,
        heat_target_state=heat_target,
    )
    strike_config = ManifoldStrikeValidationConfig(
        wall=wall,
        maximum_hit_displacement_m=2.0e-3,
        maximum_projection_distance_m=4.0e-3,
        jax_cyna_tolerance_m=1.5e-3,
        max_turns=3,
        production_DPhi=QA_WALL_DPHI,
    )
    xline_clearance_config = XLineClearanceValidationConfig(
        wall=wall,
        required_minimum_clearance_m=5.0e-4,
        maximum_closure_error_m=1.0e-4,
        production_DPhi=QA_WALL_DPHI,
    )
    heat_validation_config = ManifoldHeatValidationConfig(
        wall=wall,
        phi_edges=np.linspace(0.0, QA_FIELD_PERIOD, 65),
        s_edges=np.linspace(0.0, 1.0, 129),
        maximum_hit_displacement_m=2.0e-3,
        jax_cyna_tolerance_m=1.5e-3,
        maximum_unresolved_power_W=0.0,
        maximum_projection_distance_m=4.0e-3,
        max_turns=3,
        production_DPhi=QA_WALL_DPHI,
        field_period=QA_FIELD_PERIOD,
    )
    wall_signed_distance = _qa_wall_radial_signed_distance(wall)

    def clearance_loss(shape_dof):
        return periodic_xline_clearance_loss(
            _qa_shape_field(shape_dof),
            QA_INITIAL_GUESS,
            wall_signed_distance,
            minimum_clearance_m=5.0e-4,
            clearance_scale_m=1.0e-3,
            phi_span=QA_FIELD_PERIOD,
            map_power=5,
            n_steps_per_span=128,
            newton_iterations=8,
            newton_damping=0.8,
            bphi_floor=1.0e-10,
            residual_tolerance=1.0e-8,
            sample_stride=4,
        )

    def shape_hit(shape_dof):
        return trace_manifold_strike_reference(
            _qa_shape_field(shape_dof),
            shape_target,
        ).intersection.point_xyz

    initial_shape_dof = jnp.asarray(QA_SHAPE_DOF_NOMINAL)
    initial_endpoint = shape_hit(initial_shape_dof)
    normalized_residual = (
        initial_endpoint - shape_target.target_position_m
    ) / shape_target.position_scales_m
    normalized_jacobian = (
        jax.jacfwd(shape_hit)(initial_shape_dof)
        / shape_target.position_scales_m
    )
    gauss_newton_step = -jnp.vdot(
        normalized_jacobian,
        normalized_residual,
    ) / jnp.vdot(normalized_jacobian, normalized_jacobian)
    proposed_shape_dof = np.clip(
        float(initial_shape_dof + gauss_newton_step),
        QA_SHAPE_DOF_NOMINAL - 2.0e-5,
        QA_SHAPE_DOF_NOMINAL + 2.0e-5,
    )

    finite_difference_delta = 1.0e-7
    finite_difference = (
        shape_hit(initial_shape_dof + finite_difference_delta)
        - shape_hit(initial_shape_dof - finite_difference_delta)
    ) / (2.0 * finite_difference_delta)
    np.testing.assert_allclose(
        normalized_jacobian * shape_target.position_scales_m,
        finite_difference,
        rtol=3.0e-5,
        atol=2.0e-6,
    )

    def validate_shape_candidate(candidate_field, state):
        return validate_manifold_continuation_candidate(
            candidate_field,
            state,
            QA_R_GRID,
            QA_Z_GRID,
            QA_PHI_GRID,
            nfp=QA_NFP,
            sampling_batch_size=1024,
            maximum_anchor_displacement_m=1.0e-3,
            minimum_direction_alignment=0.99,
            maximum_sample_displacement_m=2.0e-3,
            jax_cyna_tolerance_m=3.0e-4,
            production_DPhi=QA_PRODUCTION_DPHI,
            production_fd_eps=1.0e-5,
            fixed_point_max_iter=120,
            fixed_point_tolerance=1.0e-12,
            fixed_point_residual_tolerance=1.0e-8,
            require_complete_correspondence=True,
            RZlimit=(
                QA_R_GRID[0],
                QA_R_GRID[-1],
                QA_Z_GRID[0],
                QA_Z_GRID[-1],
            ),
            refine_stable_inverse_anchor=False,
            n_threads=1,
            strike_validation_config=strike_config,
            xline_clearance_config=xline_clearance_config,
            heat_validation_config=heat_validation_config,
        )

    initial_loss = manifold_strike_stage2_target_loss(
        _qa_shape_field(initial_shape_dof),
        target_state=shape_target,
    )
    initial_clearance_loss = clearance_loss(initial_shape_dof)
    initial_heat_state = manifold_heat_flux_state(
        _qa_shape_field(initial_shape_dof),
        target_state=heat_target,
    )
    result = validated_manifold_backtracking_step(
        np.array([QA_SHAPE_DOF_NOMINAL]),
        np.array([proposed_shape_dof]),
        lambda dofs: _qa_shape_field(dofs[0]),
        continuation,
        validate_shape_candidate,
        contraction=0.5,
        maximum_attempts=4,
    )
    final_loss = manifold_strike_stage2_target_loss(
        result.field,
        target_state=shape_target,
    )
    final_clearance_loss = clearance_loss(result.dofs[0])
    final_heat_state = manifold_heat_flux_state(
        result.field,
        target_state=heat_target,
    )
    nominal_lengths = np.asarray(
        _qa_shape_field(QA_SHAPE_DOF_NOMINAL).coils.length
    )
    final_lengths = np.asarray(result.field.coils.length)
    maximum_length_change = float(
        np.max(np.abs(final_lengths - nominal_lengths))
    )

    assert result.accepted
    assert result.step_fraction == 1.0
    assert abs(result.dofs[0] - desired_shape_dof) < 1.0e-8
    assert float(final_loss) < 1.0e-5 * float(initial_loss)
    np.testing.assert_allclose(initial_clearance_loss, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(final_clearance_loss, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(initial_heat_state.deposited_power_W, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(final_heat_state.deposited_power_W, 1.0, atol=1.0e-12)
    assert float(np.max(initial_heat_state.heat_flux_W_m2)) < 1.0e5
    assert float(np.max(final_heat_state.heat_flux_W_m2)) < 1.0e5
    assert 2.0e-5 < maximum_length_change < 4.0e-5
    accepted = result.attempts[-1].validation
    assert accepted.production_refresh.anchor_displacement_m < 6.0e-4
    assert accepted.correspondence.max_deviation_m < 1.0e-4
    assert accepted.sample_refresh.sample_displacement_m < 1.0e-3
    assert accepted.xline_clearance_validation.accepted
    assert (
        1.0e-3
        < accepted.xline_clearance_validation.minimum_clearance_m
        < 2.0e-3
    )
    assert (
        accepted.xline_clearance_validation.closure_error_m
        < 2.0e-6
    )
    assert accepted.strike_validation.accepted
    assert accepted.heat_validation.accepted
    assert accepted.heat_validation.unresolved_power_W == pytest.approx(0.0)
    assert accepted.heat_validation.peak_heat_flux_W_m2 < 1.0e5
    assert accepted.heat_correspondence.accepted
    assert accepted.heat_correspondence.max_deviation_m < 1.0e-3
    assert float(
        np.sum(
            accepted.heat_validation.heat_state.heat
            * accepted.heat_validation.heat_state.cell_areas
        )
    ) == pytest.approx(1.0)
    assert (
        accepted.strike_validation.strike_refresh.hit_displacement_m
        < 6.0e-4
    )
    assert accepted.strike_validation.correspondence.deviation_m < 1.0e-3
    assert (
        accepted.strike_validation.wall_plane.projection_distance_m
        < 4.0e-3
    )
    assert (
        result.continuation_state.strike_target_state.label
        == shape_target.label
    )
    np.testing.assert_allclose(
        result.continuation_state.strike_target_state.target_position_m,
        shape_target.target_position_m,
    )
    assert result.continuation_state.heat_target_state.labels == heat_target.labels
    np.testing.assert_allclose(
        result.continuation_state.heat_target_state.strike_powers_W,
        heat_target.strike_powers_W,
    )
