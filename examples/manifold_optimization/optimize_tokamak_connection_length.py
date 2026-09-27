"""Raise the scrape-off-layer connection length of a tokamak by coil control.

Field: the regression tokamak of ``tokamak_fixture.py`` (8 TF coils, a frozen plasma
current loop and a PF coil making a lower X-point) plus a divertor coil below the
vessel floor that starts with zero current. Controls: PF current factor and height,
divertor-coil current and a TF current factor limited to +-3 %.

Objective (least squares): the mean connection length Lc of field lines launched
across the outboard-midplane scrape-off layer reaches TARGET_GAIN times its initial
value, while the X-point stays where it was (so the gain is not obtained by moving
the separatrix onto the seeds) and the controls change as little as possible.
Lc and its forward-mode derivatives come from ``essos.dynamics.connection_length``;
the X-point and its derivatives from ``essos.topology.periodic_xline_state``.

Lc is piecewise smooth in the controls: it jumps where a line changes the wall
face it strikes or the number of turns it makes. Automatic differentiation sees
only the smooth part, which is why the objective averages many seeds. The final
design is re-traced with pyna/Cyna on a sampled grid and at a tighter tolerance.
"""
import json
import os
import sys
from time import perf_counter

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokamak_fixture as tokamak  # noqa: E402
from essos.coils import Coils, Curves  # noqa: E402
from essos.dynamics import connection_length  # noqa: E402
from essos.fields import BiotSavart  # noqa: E402
from essos.topology import essos_field_to_pyna_cylindrical_grid, periodic_xline_state  # noqa: E402

OUTPUT = os.environ.get("OUTPUT", os.path.dirname(os.path.abspath(__file__)))
TARGET_GAIN = 1.3                  # requested increase of the SOL-mean connection length
LC_SCALE = 0.05                    # relative residual scale of the Lc target
XPOINT_SCALE = 0.005               # [m] allowed X-point displacement scale
CONTROL_SCALES = np.array([0.2, 0.2, 0.2, 0.05])   # regularisation of control changes
DIVERTOR_RADIUS, DIVERTOR_HEIGHT = 1.5, -0.75      # [m] divertor coil below the vessel
WALL_R, WALL_Z = (0.85, 2.3), (-0.6, 0.7)          # [m] extent of the vessel
WALL_EXPONENT = 4                                   # superellipse: box-like but smooth
SEEDS_R = np.linspace(1.64, 2.04, 64)              # [m] outboard midplane SOL band
MAX_LENGTH, TOLERANCE = 300.0, 1e-8
CONTROLS0 = np.array([1.0, tokamak.PF_HEIGHT, 0.0, 1.0])
LOWER = np.array([0.8, tokamak.PF_HEIGHT - 0.2, -0.3, 0.97])
UPPER = np.array([1.2, tokamak.PF_HEIGHT + 0.2, 0.3, 1.03])

""" Field, wall and objective """
dofs = jnp.concatenate([tokamak.COIL_DOFS, tokamak._horizontal_loop_dofs(DIVERTOR_RADIUS, DIVERTOR_HEIGHT)[None]])


def field_from_controls(controls):
    pf_factor, pf_height, divertor_factor, tf_factor = controls
    curves = Curves(dofs.at[-2, 2, 0].set(pf_height), n_segments=tokamak.N_SEGMENTS, nfp=1, stellsym=False)
    currents = jnp.concatenate([tokamak.TF_CURRENT * tf_factor * jnp.ones(tokamak.N_TF),
                                jnp.array([tokamak.PLASMA_CURRENT, tokamak.PF_CURRENT * pf_factor,
                                           tokamak.PF_CURRENT * divertor_factor])])
    return BiotSavart(Coils(curves, currents, currents_scale=tokamak.PF_CURRENT))


WALL_CENTER = (0.5 * sum(WALL_R), 0.5 * sum(WALL_Z))
WALL_HALF = (0.5 * (WALL_R[1] - WALL_R[0]), 0.5 * (WALL_Z[1] - WALL_Z[0]))


def wall(xyz):
    """Smooth superellipse vessel, positive inside. Corners of a box wall make Lc jump
    wherever lines switch faces; a smooth convex wall leaves only grazing hits."""
    u = (jnp.hypot(xyz[0], xyz[1]) - WALL_CENTER[0]) / WALL_HALF[0]
    v = (xyz[2] - WALL_CENTER[1]) / WALL_HALF[1]
    return 1.0 - u**WALL_EXPONENT - v**WALL_EXPONENT


seeds = jnp.stack([jnp.asarray(SEEDS_R), jnp.zeros(len(SEEDS_R)), jnp.zeros(len(SEEDS_R))], axis=1)


def sol_lengths(controls, tolerance=TOLERANCE):
    return connection_length(field_from_controls(controls), seeds, wall, max_length=MAX_LENGTH,
                             tolerance=tolerance, adjoint=diffrax.ForwardMode())


def xpoint(controls):
    state = periodic_xline_state(field_from_controls(controls), jnp.asarray(xpoint0), phi_span=tokamak.MAP_SPAN,
                                 n_steps_per_span=64, newton_iterations=6, bphi_floor=1e-8)
    return state.position


xpoint0 = np.asarray(tokamak.INITIAL_GUESS)
xpoint0 = np.asarray(jax.jit(xpoint)(jnp.asarray(CONTROLS0)))
Lc0 = np.asarray(jax.jit(lambda c: sol_lengths(c)["connection_length"])(jnp.asarray(CONTROLS0)))
target = TARGET_GAIN * float(np.mean(Lc0))
print(f"X-point {xpoint0}, initial SOL-mean Lc {np.mean(Lc0):.3f} m, target {target:.3f} m", flush=True)


@jax.jit
def residuals(controls):
    mean_Lc = jnp.mean(sol_lengths(controls)["connection_length"])
    return jnp.concatenate([jnp.array([(mean_Lc / target - 1.0) / LC_SCALE]),
                            (xpoint(controls) - xpoint0) / XPOINT_SCALE,
                            (controls - CONTROLS0) / CONTROL_SCALES])


jacobian = jax.jit(jax.jacfwd(residuals))

""" Optimization """
history = []


def fun(x):
    value = np.asarray(residuals(jnp.asarray(x)))
    history.append(dict(controls=x.tolist(), residuals=value.tolist(), cost=0.5 * float(value @ value)))
    print(f"  cost {history[-1]['cost']:.4e}  controls {np.array2string(x, precision=4)}", flush=True)
    return value


time0 = perf_counter()
result = least_squares(fun, CONTROLS0, jac=lambda x: np.asarray(jacobian(jnp.asarray(x))),
                       bounds=(LOWER, UPPER), x_scale=CONTROL_SCALES, max_nfev=60, verbose=1)
optimized = result.x
print(f"Optimization: {result.message} in {perf_counter() - time0:.1f} s", flush=True)

""" Validation: tighter tolerance and independent pyna/Cyna tracing on a sampled grid """
Lc1 = np.asarray(sol_lengths(jnp.asarray(optimized))["connection_length"])
Lc1_tight = np.asarray(sol_lengths(jnp.asarray(optimized), tolerance=1e-11)["connection_length"])
xpoint1 = np.asarray(xpoint(jnp.asarray(optimized)))


def native_lengths(controls):
    from pyna.toroidal.flt import trace_wall_hits_twall_field
    from pyna.toroidal.geometry import ToroidalWall
    grid = essos_field_to_pyna_cylindrical_grid(field_from_controls(jnp.asarray(controls)), tokamak.PRODUCTION_R,
                                                tokamak.PRODUCTION_Z, tokamak.PRODUCTION_PHI, nfp=tokamak.N_TF,
                                                batch_size=512)
    phi = np.linspace(0.0, 2 * np.pi, 4, endpoint=False)
    angle = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)
    c, s = np.cos(angle), np.sin(angle)
    R = WALL_CENTER[0] + WALL_HALF[0] * np.sign(c) * np.abs(c) ** (2 / WALL_EXPONENT)
    Z = WALL_CENTER[1] + WALL_HALF[1] * np.sign(s) * np.abs(s) ** (2 / WALL_EXPONENT)
    vessel = ToroidalWall(phi, np.tile(R, (4, 1)), np.tile(Z, (4, 1)))
    output = trace_wall_hits_twall_field(grid, SEEDS_R, np.zeros(len(SEEDS_R)), 0.0, 60, 0.002, vessel)
    return output["Lc_sum"], output["term_plus"], output["term_minus"]


native0, *_ = native_lengths(CONTROLS0)
native1, term_plus, term_minus = native_lengths(optimized)
relative = np.abs(native1 / Lc1_tight - 1.0)
report = dict(
    case="Regression tokamak SOL connection-length increase by coil control",
    controls_names=["pf_current_factor", "pf_height_m", "divertor_current_factor", "tf_current_factor"],
    initial_controls=CONTROLS0.tolist(), optimized_controls=optimized.tolist(), bounds=[LOWER.tolist(), UPPER.tolist()],
    optimizer_status=result.status, optimizer_message=result.message, function_evaluations=result.nfev,
    initial_mean_Lc_m=float(np.mean(Lc0)), target_mean_Lc_m=target, optimized_mean_Lc_m=float(np.mean(Lc1)),
    optimized_mean_Lc_tight_tolerance_m=float(np.mean(Lc1_tight)),
    native_initial_mean_Lc_m=float(np.mean(native0)), native_optimized_mean_Lc_m=float(np.mean(native1)),
    native_vs_essos_median_relative_difference=float(np.median(relative)),
    native_vs_essos_seeds_differing_over_1_percent=int(np.sum(relative > 0.01)),
    native_all_wall_terminated=bool(np.all(term_plus == 1) and np.all(term_minus == 1)),
    xpoint_initial_RZ=xpoint0.tolist(), xpoint_optimized_RZ=xpoint1.tolist(),
    xpoint_displacement_m=float(np.linalg.norm(xpoint1 - xpoint0)),
    seeds_R_m=SEEDS_R.tolist(), Lc_initial_m=Lc0.tolist(), Lc_optimized_m=Lc1.tolist(),
    Lc_native_initial_m=np.asarray(native0).tolist(), Lc_native_optimized_m=np.asarray(native1).tolist(),
    history=history, elapsed_seconds=perf_counter() - time0,
    limitation="Vacuum coils with a frozen plasma-current loop; superellipse vessel sampled as a 2048-point polygon "
               "for Cyna; regression geometry, not a device design.")
with open(os.path.join(OUTPUT, "optimize_tokamak_connection_length.json"), "w") as file:
    json.dump(report, file, indent=1)
print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=1), flush=True)

""" Plot """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.semilogy(SEEDS_R, Lc0, "o-", ms=3, label="initial (ESSOS)")
ax1.semilogy(SEEDS_R, Lc1, "o-", ms=3, label="optimized (ESSOS)")
ax1.semilogy(SEEDS_R, native1, "k.", ms=4, label="optimized (Cyna grid)")
ax1.axhline(np.mean(Lc0), color="C0", ls=":")
ax1.axhline(np.mean(Lc1), color="C1", ls=":")
ax1.set(xlabel="R at outboard midplane [m]", ylabel=r"$L_c$ [m]", title="Scrape-off-layer connection length")
ax1.legend(frameon=False)
costs = [h["cost"] for h in history]
ax2.semilogy(costs, "o-")
ax2.set(xlabel="function evaluation", ylabel="least-squares cost", title="Optimization history")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "optimize_tokamak_connection_length.png"), dpi=150)
