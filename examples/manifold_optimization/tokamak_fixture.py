"""Tokamak regression field: TF/PF coils plus a prescribed current loop.

The loop is a frozen plasma-background surrogate, not a self-consistent plasma.
Dimensions and currents are regression settings, not device engineering limits.
Copied from the existing ESSOS physical manifold integration fixture.
"""
import numpy as np
import jax.numpy as jnp
from essos.coils import Coils,Curves
from essos.fields import BiotSavart
from essos.manifold import periodic_xline_state,essos_field_to_pyna_cylindrical_grid
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.flt import refine_fixed_points_monodromy_span_field,trace_fixed_point_manifolds_field
from pyna.topo.manifold_correspondence import manifold_branch_reference_from_trace

N_TF = 8
N_SEGMENTS = 64
MAJOR_RADIUS = 1.4
TF_RADIUS = 0.9
TF_CURRENT = -8.0e5
PLASMA_CURRENT = 1.0e6
PF_CURRENT = 1.0e6
PF_HEIGHT = -1.2
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
        _horizontal_loop_dofs(3.0, PF_HEIGHT),
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


def _physical_field_with_pf_height(pf_height):
    curve_dofs = COIL_DOFS.at[-1, 2, 0].set(pf_height)
    curves = Curves(
        curve_dofs,
        n_segments=N_SEGMENTS,
        nfp=1,
        stellsym=False,
    )
    currents = jnp.concatenate(
        (
            TF_CURRENT * jnp.ones(N_TF),
            jnp.array([PLASMA_CURRENT, PF_CURRENT]),
        )
    )
    return BiotSavart(
        Coils(curves, currents, currents_scale=PF_CURRENT)
    )


def _tokamak_production_field(field):
    return essos_field_to_pyna_cylindrical_grid(
        field,
        PRODUCTION_R,
        PRODUCTION_Z,
        PRODUCTION_PHI,
        nfp=N_TF,
        batch_size=512,
    )


def _tokamak_production_branch(
    field,
    *,
    production_field=None,
    seed_distances=(1.0e-5, 3.0e-5),
    seed_orders=(2, 5),
    n_generations=2,
    stability="unstable",
):
    xline = periodic_xline_state(
        field,
        INITIAL_GUESS,
        phi_span=MAP_SPAN,
        n_steps_per_span=64,
        newton_iterations=6,
        bphi_floor=1.0e-8,
    )
    if production_field is None:
        production_field = _tokamak_production_field(field)
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
        N_turns=n_generations,
        DPhi=float(MAP_SPAN) / 64,
        seed_distances=np.asarray(seed_distances, dtype=float),
        seed_orders=np.asarray(seed_orders, dtype=int),
        refine_stable_inverse_anchor=False,
    )[0]
    return manifold_branch_reference_from_trace(
        payload,
        stability=stability,
        seed_side=1,
        n_generations=n_generations,
    )



def physical_field_controls(controls):
    """PF current factor and PF vertical position, with all other coils fixed."""
    dofs=COIL_DOFS.at[-1,2,0].set(controls[1])
    curves=Curves(dofs,n_segments=N_SEGMENTS,nfp=1,stellsym=False)
    currents=jnp.concatenate((TF_CURRENT*jnp.ones(N_TF),jnp.array([PLASMA_CURRENT,PF_CURRENT*controls[0]])))
    return BiotSavart(Coils(curves,currents,currents_scale=PF_CURRENT))
