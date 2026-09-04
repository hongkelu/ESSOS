"""Physical QA-coil checkpoint for manifold-aware Stage-2 targets.

The magnetic field comes from the repository's Landreman--Paul QA modular
coils plus a weak, explicit toroidal-field trim set.  The shaped wall is a
frozen regression geometry, not a proposed reactor wall: it encloses all five
components of the reference X-line and presents one non-axisymmetric divertor
notch to a labelled unstable branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

pytest.importorskip("pyna.topo.jax_manifold")
pytest.importorskip("pyna.topo.jax_strike")

from pyna.topo.manifold_correspondence import (
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
from essos.manifold import (
    essos_field_to_pyna_cylindrical_grid,
    fixed_phi_poincare_map_from_coils,
    periodic_xline_state,
)
from essos.manifold_optimization import trace_manifold_reference
from essos.manifold_strike_optimization import (
    ManifoldStrikeStage2Target,
    trace_manifold_strike_reference,
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

    def hit_xyz(trim_current_kA):
        return trace_manifold_strike_reference(
            _qa_field(trim_current_kA),
            target,
        ).intersection.point_xyz

    trim_current = jnp.asarray(QA_TRIM_CURRENT_KA)
    gradient = jax.jacfwd(hit_xyz)(trim_current)
    delta = 1.0e-6
    finite_difference = (
        hit_xyz(trim_current + delta) - hit_xyz(trim_current - delta)
    ) / (2.0 * delta)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.5
    np.testing.assert_allclose(
        gradient,
        finite_difference,
        rtol=3.0e-5,
        atol=1.0e-6,
    )
