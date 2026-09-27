"""Super-X divertor by coil control: move the outer strike point outward at fixed plasma shape.

Field: an axisymmetric 1/R toroidal field (no TF ripple), a frozen plasma current
carried by 13 filaments inside an elongated cross-section, and eight poloidal-field
coils: four shaping coils and four divertor coils below and inboard of the vessel.
All coil currents are controls; the coils and filaments share one ESSOS ``Coils``
object, so the poloidal flux psi = R A_phi and the traced field come from the same
discretisation.

Step 1 fits a lower-single-null equilibrium with isoflux conditions: grad psi = 0 at
the X-point and psi = psi_X at LCFS control points. Step 2 keeps those conditions and
drives the outer strike radius on the divertor floor, R_t (outer root of
psi(R, Z_floor) = psi_X), towards a Super-X target while raising the mean connection
length of the near scrape-off layer. psi, R_t and their derivatives are exact
functions of the coil currents; the connection length and its forward-mode derivatives
come from ``essos.dynamics.connection_length``. The design is checked by tracing a
field line next to the separatrix to the floor and by re-tracing the scrape-off layer
with pyna/Cyna on a sampled grid.

The plasma current is frozen (no free-boundary response): this is stage 1 of a
divertor-coil design, before coupling to a Grad-Shafranov solver.
"""
import json
import os
from time import perf_counter

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)
from essos.coils import Coils, Curves  # noqa: E402
from essos.dynamics import connection_length  # noqa: E402
from essos.fields import BiotSavart, CircularTokamakField, CombinedField  # noqa: E402

OUTPUT = os.environ.get("OUTPUT", os.path.dirname(os.path.abspath(__file__)))
R0, B0, IP = 1.6, 2.0, 1.0e6                       # [m], [T], [A]
A_MINOR, KAPPA = 0.45, 1.5
XPOINT = np.array([1.40, -0.85])                   # [m] held fixed
CONTROL = np.array([[2.05, 0.0], [1.15, 0.0], [1.55, 0.70], [1.85, 0.52]])  # LCFS points
PF_POSITIONS = np.array([[1.10, 1.25], [2.40, 1.25], [2.85, 0.00], [2.80, -0.85],    # shaping
                         [0.80, -1.35], [1.45, -1.70], [1.95, -1.70], [2.45, -1.65],   # divertor
                         [0.75, -0.90], [1.20, -1.75], [1.70, -1.80], [2.20, -1.80], [2.85, -1.40]])
PF_BOUND = 2.5e6                                   # [A]
WALL_R, WALL_Z, WALL_EXPONENT = (0.9, 2.65), (-1.5, 1.1), 8
Z_FLOOR = WALL_Z[0]
# DESIGN = "superx": drive the outer strike radius to TARGET_RADIUS (longer leg, more total flux
# expansion) with a secondary Lc target. DESIGN = "long_lc": make the near-SOL connection length the
# primary objective (low poloidal field along the leg); the strike point only has to stay on the floor.
# DESIGN = "xpt": X-point target -- a secondary null on the outer leg above the floor, placed just
# inside the near scrape-off layer, where the vanishing poloidal field lengthens field lines.
DESIGN = os.environ.get("DESIGN", "superx")
SECOND_NULL = np.array([1.85, -1.30])              # [m] X-point-target location (xpt)
SECOND_NULL_SOL_DEPTH = float(os.environ.get("SECOND_NULL_SOL_DEPTH", 0.03))  # [m] flux label as midplane distance
SECOND_NULL_DETERMINANT = -0.3                     # weak saddle: a wide low-poloidal-field region
TARGET_RADIUS = 2.20                               # [m] Super-X outer strike radius
FLOOR_WINDOW = (1.6, 2.5)                          # [m] allowed outer strike radius for long_lc
LC_GAIN = {"superx": 1.3, "long_lc": 2.0, "xpt": 3.0}[DESIGN]  # requested near-SOL connection-length gain
SOL_R = np.linspace(2.055, 2.13, 16)               # [m] outboard midplane seeds, 5-80 mm outside the LCFS
MAX_LENGTH, TOLERANCE = 400.0, 1e-9
SCALE = 1.0e6


def _loop(radius, height):
    dofs = np.zeros((3, 3))
    dofs[0, 2], dofs[1, 1], dofs[2, 0] = radius, radius, height
    return dofs


def _plasma_filaments():
    rings = [(R0 + A_MINOR * x, A_MINOR * KAPPA * z, 1.0 - (x * x + z * z))
             for x in np.linspace(-0.8, 0.8, 5) for z in np.linspace(-0.8, 0.8, 5) if x * x + z * z <= 0.64 + 1e-9]
    rings = np.array(rings)
    rings[:, 2] *= IP / rings[:, 2].sum()
    return rings


FILAMENTS = _plasma_filaments()
CURVES = Curves(jnp.asarray(np.stack([_loop(R, Z) for R, Z, _ in FILAMENTS] + [_loop(R, Z) for R, Z in PF_POSITIONS])),
                n_segments=128, nfp=1, stellsym=False)
# Curves caches gamma lazily on the object; fill the cache eagerly so a jit trace never stores
# a tracer in this shared, fixed geometry.
GAMMA, DL_Y = np.asarray(CURVES.gamma), np.asarray(CURVES.gamma_dash)[..., 1] / CURVES.gamma.shape[1]
TOROIDAL = CircularTokamakField(R0, B0, 0.0)       # zero poloidal part: B_phi = B0 R0 / R


def coils(pf_scaled):
    currents = jnp.concatenate([jnp.asarray(FILAMENTS[:, 2]), pf_scaled * SCALE])
    return Coils(CURVES, currents, currents_scale=SCALE)


def field(pf_scaled):
    return CombinedField(BiotSavart(coils(pf_scaled)), TOROIDAL)


def psi(point, pf_scaled):
    """Poloidal flux R A_phi at (R, phi=0, Z) from the discretised loops [Wb/rad]."""
    currents = jnp.concatenate([jnp.asarray(FILAMENTS[:, 2]), pf_scaled * SCALE])
    x = jnp.array([point[0], 0.0, point[1]])
    distance = jnp.linalg.norm(x - GAMMA, axis=-1)
    return point[0] * 1e-7 * jnp.sum(currents[:, None] * DL_Y / distance)


def target_radius(pf_scaled, guess):
    """Outer root of psi(R, Z_floor) = psi_X by Newton from ``guess``."""
    psi_x = psi(jnp.asarray(XPOINT), pf_scaled)
    f = lambda R: psi(jnp.array([R, Z_FLOOR]), pf_scaled) - psi_x  # noqa: E731
    return jax.lax.fori_loop(0, 30, lambda _, R: R - f(R) / jax.grad(f)(R), guess)


def wall(xyz):
    u = (jnp.hypot(xyz[0], xyz[1]) - 0.5 * sum(WALL_R)) / (0.5 * (WALL_R[1] - WALL_R[0]))
    v = (xyz[2] - 0.5 * sum(WALL_Z)) / (0.5 * (WALL_Z[1] - WALL_Z[0]))
    return 1.0 - u**WALL_EXPONENT - v**WALL_EXPONENT


sol_seeds = jnp.stack([jnp.asarray(SOL_R), jnp.zeros(len(SOL_R)), jnp.zeros(len(SOL_R))], 1)


def sol_lengths(pf_scaled, tolerance=TOLERANCE):
    return connection_length(field(pf_scaled), sol_seeds, wall, max_length=MAX_LENGTH, tolerance=tolerance,
                             adjoint=diffrax.ForwardMode())["connection_length"]


def isoflux(pf_scaled):
    psi_x = psi(jnp.asarray(XPOINT), pf_scaled)
    grad = jax.grad(psi)(jnp.asarray(XPOINT), pf_scaled)
    return jnp.concatenate([grad, jnp.array([psi(jnp.asarray(p), pf_scaled) - psi_x for p in CONTROL])])


""" Step 1: lower-single-null base equilibrium """
equilibrium_residuals = jax.jit(lambda x: jnp.concatenate([10.0 * isoflux(x), 3e-3 * x]))
time0 = perf_counter()
base = least_squares(lambda x: np.asarray(equilibrium_residuals(jnp.asarray(x))),
                     np.array([0.5, 0.0, -0.3, -0.3, 1.3, 0.5, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
                     jac=lambda x: np.asarray(jax.jit(jax.jacfwd(equilibrium_residuals))(jnp.asarray(x))),
                     bounds=(-PF_BOUND / SCALE, PF_BOUND / SCALE)).x
floor_R = np.linspace(XPOINT[0] + 0.05, WALL_R[1] - 0.05, 400)
floor_psi = np.asarray(jax.vmap(lambda R: psi(jnp.array([R, Z_FLOOR]), jnp.asarray(base)))(jnp.asarray(floor_R)))
crossing = np.nonzero(np.diff(np.sign(floor_psi - float(psi(jnp.asarray(XPOINT), jnp.asarray(base))))))[0]
guess0 = float(floor_R[crossing[0]])
Rt0 = float(target_radius(jnp.asarray(base), guess0))
Lc0 = np.asarray(jax.jit(sol_lengths)(jnp.asarray(base)))
target_Lc = LC_GAIN * float(np.mean(Lc0))
print(f"Base PF currents [MA] {np.round(base, 4)}; outer strike R_t = {Rt0:.4f} m; "
      f"near-SOL mean Lc = {np.mean(Lc0):.3f} m", flush=True)

""" Step 2: Super-X at fixed X-point and LCFS """


def flux_expansion_traced(x, Rt):
    """Poloidal flux expansion f_x = (|grad psi|)_midplane / (|grad psi|)_target along the separatrix."""
    return (jnp.linalg.norm(jax.grad(psi)(jnp.asarray(CONTROL[0]), x))
            / jnp.linalg.norm(jax.grad(psi)(jnp.stack([Rt, jnp.asarray(Z_FLOOR)]), x)))


fx0 = float(flux_expansion_traced(jnp.asarray(base), jnp.asarray(Rt0)))


# In vacuum psi is linear in the coil currents, so the isoflux conditions are exactly A x = b.
# Moving only along the null space of A keeps the X-point and LCFS points fixed exactly.
def second_null_conditions(pf_scaled):
    """grad psi = 0 at SECOND_NULL and psi there on the flux surface SECOND_NULL_SOL_DEPTH outside the LCFS."""
    psi_x = psi(jnp.asarray(XPOINT), pf_scaled)
    depth = (psi(jnp.array([CONTROL[0, 0] + SECOND_NULL_SOL_DEPTH, 0.0]), pf_scaled) - psi_x)
    return jnp.concatenate([jax.grad(psi)(jnp.asarray(SECOND_NULL), pf_scaled),
                            jnp.array([psi(jnp.asarray(SECOND_NULL), pf_scaled) - psi_x - depth])])


def linear_conditions(pf_scaled):
    if DESIGN == "xpt":
        return jnp.concatenate([isoflux(pf_scaled), second_null_conditions(pf_scaled)])
    return isoflux(pf_scaled)


# All conditions are affine in the currents: c(x) = c(start) + A (x - start). Start from the
# least-change currents that satisfy them exactly and move only in the null space of A.
A = np.asarray(jax.jacfwd(linear_conditions)(jnp.asarray(base)))
start = base - np.linalg.pinv(A) @ np.asarray(linear_conditions(jnp.asarray(base)))
NULL = np.linalg.svd(A)[2][A.shape[0]:].T           # (n_coils, n_coils - n_conditions)
print(f"{A.shape[0]} exact linear conditions, {NULL.shape[1]} free directions", flush=True)


def currents(z):
    return jnp.asarray(start) + jnp.asarray(NULL) @ z


def second_null_hessian_determinant(pf_scaled):
    return jnp.linalg.det(jax.hessian(psi)(jnp.asarray(SECOND_NULL), pf_scaled))


log_Lc0 = float(np.mean(np.log(Lc0)))


@jax.jit
def residuals(z):
    x = currents(z)
    Rt = target_radius(x, jnp.asarray(guess0))
    Lc = sol_lengths(x)
    if DESIGN == "superx":
        # The poloidal flux expansion may not fall below its base value (one-sided).
        return jnp.concatenate([jnp.array([(Rt - TARGET_RADIUS) / 0.02, (jnp.mean(Lc) / target_Lc - 1.0) / 0.1,
                                           100.0 * jnp.minimum(flux_expansion_traced(x, Rt) / fx0 - 1.0, 0.0)]),
                                0.05 * (x - base)])
    # Mean log Lc weights every seed of the band equally (the lines next to the separatrix would
    # otherwise dominate); the strike point is only kept inside the floor window (one-sided).
    terms = [(jnp.mean(jnp.log(Lc)) - log_Lc0 - jnp.log(LC_GAIN)) / 0.05]
    if DESIGN == "xpt":
        # A weak saddle (small negative det Hessian psi) widens the region of low poloidal field; the
        # coil currents stay within PF_BOUND (one-sided).
        terms.append((second_null_hessian_determinant(x) - SECOND_NULL_DETERMINANT) / 0.05)
        terms.append(jnp.sum(jnp.maximum(jnp.abs(x) - PF_BOUND / SCALE, 0.0)) / 0.01)
    else:
        terms += [jnp.maximum(FLOOR_WINDOW[0] - Rt, 0.0) / 0.01, jnp.maximum(Rt - FLOOR_WINDOW[1], 0.0) / 0.01]
    return jnp.concatenate([jnp.array(terms), 0.02 * (x - start)])


jacobian = jax.jit(jax.jacfwd(residuals))
history = []


def fun(z):
    value = np.asarray(residuals(jnp.asarray(z)))
    x = np.asarray(currents(jnp.asarray(z)))
    history.append(dict(z=z.tolist(), currents_MA=x.tolist(), cost=0.5 * float(value @ value)))
    print(f"  cost {history[-1]['cost']:.4e}  currents {np.array2string(x, precision=3)}", flush=True)
    return value


result = least_squares(fun, np.zeros(NULL.shape[1]), jac=lambda z: np.asarray(jacobian(jnp.asarray(z))),
                       max_nfev=150, verbose=1)
optimized = np.asarray(currents(jnp.asarray(result.x)))
if np.max(np.abs(optimized)) > PF_BOUND / SCALE:
    print(f"warning: a coil current exceeds {PF_BOUND / SCALE} MA", flush=True)
print(f"Optimization: {result.message} ({result.nfev} evaluations, {perf_counter() - time0:.1f} s)", flush=True)

""" Validation """
Rt1 = float(target_radius(jnp.asarray(optimized), guess0))
Lc1 = np.asarray(sol_lengths(jnp.asarray(optimized)))
Lc1_tight = np.asarray(sol_lengths(jnp.asarray(optimized), tolerance=1e-11))
isoflux1 = np.asarray(isoflux(jnp.asarray(optimized)))


def separatrix_strike(pf_scaled):
    """Trace the field line 0.1 mm outside the LCFS at the outboard midplane to the wall."""
    seed = jnp.array([[CONTROL[0, 0] + 1e-4, 0.0, 0.0]])
    out = connection_length(field(pf_scaled), seed, wall, max_length=MAX_LENGTH, tolerance=1e-11)
    points = np.asarray(out["strike_points"][0])
    lower = int(np.argmin(points[:, 2]))
    return float(np.hypot(points[lower, 0], points[lower, 1])), float(points[lower, 2])


traced0, traced1 = separatrix_strike(jnp.asarray(base)), separatrix_strike(jnp.asarray(optimized))


def poloidal_field(point, pf_scaled):
    grad = jax.grad(psi)(jnp.asarray(point), pf_scaled)
    return float(jnp.linalg.norm(grad) / point[0])


def flux_expansion(pf_scaled, Rt):
    """Poloidal flux expansion f_x = (B_p R)_midplane / (B_p R)_target along the separatrix."""
    mid, tgt = CONTROL[0], np.array([Rt, Z_FLOOR])
    return (poloidal_field(mid, pf_scaled) * mid[0]) / (poloidal_field(tgt, pf_scaled) * tgt[0])


def native_lengths(pf_scaled):
    from pyna.toroidal.flt import trace_wall_hits_twall_field
    from pyna.toroidal.geometry import ToroidalWall
    from essos.topology import essos_field_to_pyna_cylindrical_grid
    grid = essos_field_to_pyna_cylindrical_grid(field(jnp.asarray(pf_scaled)), np.linspace(0.8525, 2.7025, 186),
                                                np.linspace(-1.5475, 1.1525, 271), np.linspace(0, 2 * np.pi, 4, endpoint=False),
                                                batch_size=2048)
    angle = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)
    c, s = np.cos(angle), np.sin(angle)
    R = 0.5 * sum(WALL_R) + 0.5 * (WALL_R[1] - WALL_R[0]) * np.sign(c) * np.abs(c) ** (2 / WALL_EXPONENT)
    Z = 0.5 * sum(WALL_Z) + 0.5 * (WALL_Z[1] - WALL_Z[0]) * np.sign(s) * np.abs(s) ** (2 / WALL_EXPONENT)
    vessel = ToroidalWall(np.linspace(0, 2 * np.pi, 4, endpoint=False), np.tile(R, (4, 1)), np.tile(Z, (4, 1)))
    out = trace_wall_hits_twall_field(grid, SOL_R, np.zeros(len(SOL_R)), 0.0, 80, 0.002, vessel)
    return np.asarray(out["Lc_sum"]), bool(np.all(out["term_plus"] == 1) and np.all(out["term_minus"] == 1))


native0, _ = native_lengths(base)
native1, native_terminated = native_lengths(optimized)
relative = np.abs(native1 / Lc1_tight - 1.0)
report = dict(
    case="Tokamak Super-X by divertor-coil currents at fixed X-point and LCFS (vacuum + frozen plasma)",
    pf_positions_m=PF_POSITIONS.tolist(), base_currents_MA=base.tolist(), optimized_currents_MA=optimized.tolist(),
    optimizer_message=result.message, function_evaluations=result.nfev,
    target_radius_m=dict(base=Rt0, optimized=Rt1, target=TARGET_RADIUS),
    separatrix_trace_strike_RZ_m=dict(base=traced0, optimized=traced1),
    total_flux_expansion_Rt_over_Ru=dict(base=Rt0 / CONTROL[0, 0], optimized=Rt1 / CONTROL[0, 0]),
    poloidal_flux_expansion=dict(base=flux_expansion(jnp.asarray(base), Rt0),
                                 optimized=flux_expansion(jnp.asarray(optimized), Rt1)),
    near_sol_mean_Lc_m=dict(base=float(np.mean(Lc0)), optimized=float(np.mean(Lc1)),
                            optimized_tight=float(np.mean(Lc1_tight)), target=target_Lc,
                            native_base=float(np.mean(native0)), native_optimized=float(np.mean(native1))),
    native_vs_essos_max_relative_difference=float(np.max(relative)), native_all_wall_terminated=native_terminated,
    optimized_isoflux_residuals=isoflux1.tolist(), seeds_R_m=SOL_R.tolist(),
    second_null=dict(position_m=SECOND_NULL.tolist(), sol_depth_m=SECOND_NULL_SOL_DEPTH,
                     conditions=np.asarray(second_null_conditions(jnp.asarray(optimized))).tolist(),
                     hessian_determinant=float(second_null_hessian_determinant(jnp.asarray(optimized))))
    if DESIGN == "xpt" else None,
    Lc_base_m=Lc0.tolist(), Lc_optimized_m=Lc1.tolist(), Lc_native_optimized_m=native1.tolist(), history=history,
    elapsed_seconds=perf_counter() - time0,
    limitation="Frozen plasma filaments (no free-boundary response), no coil force or vertical-stability limits.")
report["design"] = DESIGN
with open(os.path.join(OUTPUT, f"optimize_tokamak_{DESIGN}.json"), "w") as file:
    json.dump(report, file, indent=1)
print(json.dumps({k: v for k, v in report.items() if k not in ("history", "Lc_base_m", "Lc_optimized_m",
                                                             "Lc_native_optimized_m", "seeds_R_m")}, indent=1), flush=True)

""" Plot """
Rg, Zg = np.meshgrid(np.linspace(0.6, 3.0, 121), np.linspace(-1.9, 1.45, 168))
grid_points = jnp.stack([Rg.ravel(), Zg.ravel()], 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
label = {"superx": "Super-X", "long_lc": "long Lc", "xpt": "X-point target"}[DESIGN]
for ax, coil_currents, title in ((axes[0], base, "base"), (axes[1], optimized, label)):
    values = np.asarray(jax.jit(jax.vmap(lambda p: psi(p, jnp.asarray(coil_currents))))(grid_points)).reshape(Rg.shape)
    level = float(psi(jnp.asarray(XPOINT), jnp.asarray(coil_currents)))
    ax.contour(Rg, Zg, values, 40, colors="0.8", linewidths=0.5)
    ax.contour(Rg, Zg, values, [level], colors="C3")
    ax.scatter(*PF_POSITIONS.T, c=coil_currents, cmap="coolwarm", vmin=-2.5, vmax=2.5, marker="s", s=70, edgecolors="k")
    ax.plot(FILAMENTS[:, 0], FILAMENTS[:, 1], "b.", ms=3)
    ax.plot(*XPOINT, "kx")
    angle = np.linspace(0, 2 * np.pi, 400)
    ax.plot(0.5 * sum(WALL_R) + 0.5 * (WALL_R[1] - WALL_R[0]) * np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** 0.25,
            0.5 * sum(WALL_Z) + 0.5 * (WALL_Z[1] - WALL_Z[0]) * np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** 0.25, "k")
    ax.set(aspect="equal", xlim=(0.6, 3.0), ylim=(-1.9, 1.45), title=f"{title}: R_t = "
           f"{float(target_radius(jnp.asarray(coil_currents), guess0)):.2f} m", xlabel="R [m]", ylabel="Z [m]")
axes[2].plot(100 * (SOL_R - CONTROL[0, 0]), Lc0, "o-", ms=3, label="base (ESSOS)")
axes[2].plot(100 * (SOL_R - CONTROL[0, 0]), Lc1, "o-", ms=3, label=f"{label} (ESSOS)")
axes[2].plot(100 * (SOL_R - CONTROL[0, 0]), native1, "k.", label=f"{label} (Cyna grid)")
axes[2].set(xlabel="distance outside LCFS at outboard midplane [cm]", ylabel=r"$L_c$ [m]",
            title="Near scrape-off-layer connection length")
axes[2].legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, f"optimize_tokamak_{DESIGN}.png"), dpi=140)
