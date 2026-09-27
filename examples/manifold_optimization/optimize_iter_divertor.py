"""ITER divertor: longer scrape-off-layer connection length with strikes kept on the real targets.

Machine: the ITER first wall/divertor contour and CS/PF coils of the Open FUSION Toolkit
TokaMaker example (``ITER_geom.json``; set ``ITER_GEOM`` to its path), an axisymmetric 1/R
toroidal field (F0 = 5.3 T x 6.2 m) and the 15.6 MA plasma of a TokaMaker free-boundary
equilibrium with the ITER reference coil currents. Outside the plasma its current is
represented exactly enough by 181 exterior-equivalent filaments fitted to the full TokaMaker
current distribution (``examples/input_files/iter_tokamaker_reference.json``). All fields are
exact circular-loop fields (:class:`essos.fields.CircularLoopsField`), so nothing is
discretised in the toroidal direction.

Controls: the 12 CS/PF coil currents. The X-point and six LCFS points are held exactly
(isoflux conditions are linear in the currents, so the optimizer moves in their null space).
Objective: the mean log connection length of field lines launched 0.3-5 cm outside the LCFS
at the outboard midplane, traced to the ITER wall polygon with ``connection_length``.
Constraints (one-sided): both separatrix strike points on their vertical targets at least
TARGET_MARGIN from the ends, field-line incidence angle on both targets at least
MIN_GRAZING_ANGLE (a smaller angle comes with a lower poloidal field and a longer
connection length, but tiles need a minimum angle), and approximate ITER coil current limits.
The design is re-traced with pyna/Cyna on the same wall.

The plasma current is frozen (no free-boundary response): coil changes move the flux
surfaces but not the plasma current. This is stage 1 before coupling back to TokaMaker.
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
from essos.dynamics import connection_length  # noqa: E402
from essos.fields import CircularLoopsField, CircularTokamakField, CombinedField  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.environ.get("OUTPUT", HERE)
ITER_GEOM = os.environ.get("ITER_GEOM", os.path.join(
    HERE, "../../../.dependencies/external/OpenFUSIONToolkit/src/examples/TokaMaker/ITER/ITER_geom.json"))
DATA = json.load(open(os.path.join(HERE, "../input_files/iter_tokamaker_reference.json")))
WALL = np.array(json.load(open(ITER_GEOM))["limiter"])            # closed polygon, first point repeated
OUTER_TARGET, INNER_TARGET = WALL[23:32], WALL[41:47]              # vertical targets, listed along the wall
COIL_NAMES, COIL_RZ = DATA["coil_names"], np.array(DATA["coil_RZ"])
REFERENCE = np.array(DATA["coil_At"]) / 1e6                        # [MA-turns]
# Approximate ITER limits [MA-turns]: CS 45 kA x 553 turns; PF1 48 kA, PF2-4 55 kA, PF5-6 52 kA x turns.
LIMITS = np.array([24.9] * 6 + [11.9, 6.3, 10.2, 9.3, 11.3, 23.9])
PLASMA = np.array([DATA["plasma_R"], DATA["plasma_Z"], DATA["plasma_I"]])
XPOINT, LCFS = np.array(DATA["xpoint"]), np.array(DATA["lcfs"])
LCFS_POINTS = LCFS[np.argmin(np.abs(np.arctan2(LCFS[:, 1] - 0.5, LCFS[:, 0] - 6.3)[:, None]
                                    - np.deg2rad([0, 55, 100, 150, 200, 320])[None]), axis=0)]
TOROIDAL = CircularTokamakField(6.2, 5.3, 0.0)
MIDPLANE = LCFS[np.argmax(LCFS[:, 0])]                            # outboard midplane LCFS point
SOL_R = None  # set from the reference first-wall gap below
TARGET_MARGIN, MIN_GRAZING_ANGLE = 0.10, np.deg2rad(1.0)
# First-wall gap: every main-chamber wall vertex (outside the divertor, WALL[18:48]) must stay outside
# the flux surface WALL_GAP beyond the LCFS at the outboard midplane, WALL_GAP being the reference
# equilibrium's own clearance, so the scrape-off layer keeps connecting to the divertor targets and
# is neither limited nor skimmed by the first wall. The seed band stays inside 75 % of that gap.
MAIN_CHAMBER = np.r_[WALL[:18], WALL[48:-1]]
# DESIGN = "heat" (default): seeds inside the heat channel, 0.25-6 lambda_q from the separatrix, with the
# objective weighted by the exponential heat-flux profile exp(-r / lambda_q); lambda_q from the Eich
# regression #14, 0.63 mm * B_pol,omp[T]^-1.19 (Eich et al., Nucl. Fusion 53, 093031). DESIGN = "band":
# unweighted band out to 75 % of the first-wall gap.
DESIGN = os.environ.get("DESIGN", "heat")
LC_GAIN = 1.5 if DESIGN == "heat" else 2.0
# Optional X-point flattening target, det(Hessian psi) at the X-point (reference about -8.2); a
# flatter X-point lowers B_pol around it and lengthens near-separatrix field lines.
XPOINT_DET = float(os.environ["XPOINT_DET"]) if "XPOINT_DET" in os.environ else None
# A loose tolerance lets the optimizer exploit tracing error on long lines (seen at 1e-8).
MAX_LENGTH, TOLERANCE = 2000.0, 1e-10


def loops(x):
    return CircularLoopsField(np.r_[PLASMA[0], COIL_RZ[:, 0]], np.r_[PLASMA[1], COIL_RZ[:, 1]],
                              jnp.concatenate([jnp.asarray(PLASMA[2]), x * 1e6]))


def field(x):
    return CombinedField(loops(x), TOROIDAL)


def psi(point, x):
    return loops(x).psi(point[0], point[1])


def wall(xyz):
    """Signed distance to the ITER wall polygon in the poloidal plane, positive inside."""
    point = jnp.array([jnp.hypot(xyz[0], xyz[1]), xyz[2]])
    a, b = jnp.asarray(WALL[:-1]), jnp.asarray(WALL[1:])
    t = jnp.clip(jnp.sum((point - a) * (b - a), 1) / jnp.sum((b - a) ** 2, 1), 0.0, 1.0)
    distance = jnp.min(jnp.linalg.norm(point - a - t[:, None] * (b - a), axis=1))
    crossings = ((a[:, 1] > point[1]) != (b[:, 1] > point[1])) & (
        point[0] < a[:, 0] + (point[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
    return jnp.where(jnp.sum(crossings) % 2 == 1, distance, -distance)


def strike(x, target):
    """Separatrix crossing of a target polyline: point, distance from each end, flux miss, incidence angle."""
    psi_x = psi(jnp.asarray(XPOINT), x)
    vertices = jnp.asarray(target)
    f = jax.vmap(lambda p: psi(p, x))(vertices) - psi_x
    change = f[:-1] * f[1:] <= 0.0
    i = jnp.argmax(change)
    a, b = vertices[i], vertices[i + 1]
    g = lambda s: psi(a + s * (b - a), x) - psi_x  # noqa: E731
    s = jnp.clip(f[i] / (f[i] - f[i + 1]), 0.0, 1.0)
    for _ in range(4):
        s = s - g(s) / jax.grad(g)(s)
    point = a + s * (b - a)
    lengths = jnp.linalg.norm(vertices[1:] - vertices[:-1], axis=1)
    before = jnp.sum(jnp.where(jnp.arange(len(lengths)) < i, lengths, 0.0)) + s * lengths[i]
    miss = jnp.where(jnp.any(change), 0.0, jnp.min(jnp.abs(f[jnp.array([0, -1])])))
    tangent = (b - a) / jnp.linalg.norm(b - a)
    normal = jnp.array([tangent[1], -tangent[0]])
    grad = jax.grad(psi)(point, x)
    B_pol = jnp.array([-grad[1], grad[0]]) / point[0]
    B = jnp.sqrt(jnp.sum(B_pol**2) + (DATA["F0"] / point[0]) ** 2)
    angle = jnp.arcsin(jnp.abs(B_pol @ normal) / B)
    return point, before, jnp.sum(lengths) - before, miss, angle


def isoflux(x):
    psi_x = psi(jnp.asarray(XPOINT), x)
    return jnp.concatenate([jax.grad(psi)(jnp.asarray(XPOINT), x),
                            jnp.array([psi(jnp.asarray(p), x) - psi_x for p in LCFS_POINTS])])


def wall_gap_violation(x):
    """Flux by which main-chamber wall vertices fall inside the WALL_GAP flux surface (one-sided) [Wb]."""
    psi_x = psi(jnp.asarray(XPOINT), x)
    edge = psi(jnp.array([MIDPLANE[0] + WALL_GAP, MIDPLANE[1]]), x) - psi_x
    wall_psi = jax.vmap(lambda p: psi(p, x))(jnp.asarray(MAIN_CHAMBER)) - psi_x
    return jnp.maximum(jnp.sign(edge) * (edge - wall_psi), 0.0)


def reference_wall_gap():
    """Outboard-midplane distance of the flux surface through the innermost main-chamber wall vertex."""
    x = jnp.asarray(REFERENCE)
    psi_x = float(psi(jnp.asarray(XPOINT), x))
    sign = np.sign(float(psi(jnp.array([MIDPLANE[0] + 0.01, MIDPLANE[1]]), x)) - psi_x)
    wall_psi = sign * (np.asarray(jax.vmap(lambda p: psi(p, x))(jnp.asarray(MAIN_CHAMBER))) - psi_x)
    target = wall_psi.min()
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sign * (float(psi(jnp.array([MIDPLANE[0] + mid, MIDPLANE[1]]), x)) - psi_x) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


WALL_GAP = reference_wall_gap()
B_pol_omp = float(jnp.hypot(*loops(jnp.asarray(REFERENCE)).B_cylindrical(MIDPLANE[0] + 1e-4, MIDPLANE[1])))
LAMBDA_Q = 0.63e-3 * B_pol_omp ** -1.19
if DESIGN == "heat":
    SOL_R = MIDPLANE[0] + LAMBDA_Q * np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    WEIGHTS = np.exp(-(SOL_R - MIDPLANE[0]) / LAMBDA_Q)
else:
    SOL_R = MIDPLANE[0] + np.linspace(0.002, 0.75 * WALL_GAP, 12)
    WEIGHTS = np.ones(len(SOL_R))
WEIGHTS = WEIGHTS / WEIGHTS.sum()
print(f"B_pol,omp = {B_pol_omp:.3f} T, lambda_q (Eich #14) = {1e3 * LAMBDA_Q:.3f} mm; seeds "
      f"{np.round(1e3 * (SOL_R - MIDPLANE[0]), 3)} mm", flush=True)
print(f"reference first-wall gap mapped to the outboard midplane: {100 * WALL_GAP:.2f} cm", flush=True)
seeds = jnp.stack([jnp.asarray(SOL_R), jnp.zeros(len(SOL_R)), jnp.full(len(SOL_R), MIDPLANE[1])], 1)


def sol_lengths(x, tolerance=TOLERANCE):
    out = connection_length(field(x), seeds, wall, max_length=MAX_LENGTH, tolerance=tolerance,
                            adjoint=diffrax.ForwardMode())
    return out["connection_length"], out["hit"]


""" Reference and exact constraints """
time0 = perf_counter()
A = np.asarray(jax.jacfwd(isoflux)(jnp.asarray(REFERENCE)))
start = REFERENCE - np.linalg.pinv(A) @ np.asarray(isoflux(jnp.asarray(REFERENCE)))
NULL = np.linalg.svd(A)[2][A.shape[0]:].T
currents = lambda z: jnp.asarray(start) + jnp.asarray(NULL) @ z  # noqa: E731
Lc0, hit0 = map(np.asarray, jax.jit(sol_lengths)(jnp.asarray(start)))
log_Lc0 = float(np.sum(WEIGHTS * np.log(Lc0)))
outer0, inner0 = [jax.jit(lambda x, t=t: strike(x, t))(jnp.asarray(start)) for t in (OUTER_TARGET, INNER_TARGET)]
print(f"{A.shape[0]} exact conditions, {NULL.shape[1]} free directions; reference mean Lc {np.mean(Lc0):.1f} m "
      f"(all hit wall: {bool(hit0.all())}); outer strike {np.asarray(outer0[0])}, angle "
      f"{np.rad2deg(float(outer0[4])):.2f} deg; inner strike {np.asarray(inner0[0])}, angle "
      f"{np.rad2deg(float(inner0[4])):.2f} deg", flush=True)


@jax.jit
def residuals(z):
    x = currents(z)
    Lc, _ = sol_lengths(x)
    terms = [(jnp.sum(jnp.asarray(WEIGHTS) * jnp.log(Lc)) - log_Lc0 - jnp.log(LC_GAIN)) / 0.05]
    if XPOINT_DET is not None:
        terms.append((jnp.linalg.det(jax.hessian(psi)(jnp.asarray(XPOINT), x)) - XPOINT_DET) / 0.1)
    for target in (OUTER_TARGET, INNER_TARGET):
        _, before, after, miss, angle = strike(x, target)
        terms += [miss / 0.01, jnp.maximum(TARGET_MARGIN - before, 0.0) / 0.01,
                  jnp.maximum(TARGET_MARGIN - after, 0.0) / 0.01,
                  jnp.maximum(MIN_GRAZING_ANGLE - angle, 0.0) / np.deg2rad(0.05)]
    return jnp.concatenate([jnp.array(terms), wall_gap_violation(x) / 0.0002,
                            jnp.maximum(jnp.abs(x) - LIMITS, 0.0) / 0.05, 0.05 * (x - start)])


jacobian = jax.jit(jax.jacfwd(residuals))
history = []


def fun(z):
    value = np.asarray(residuals(jnp.asarray(z)))
    history.append(dict(currents_MAt=np.asarray(currents(jnp.asarray(z))).tolist(), cost=0.5 * float(value @ value)))
    print(f"  cost {history[-1]['cost']:.4e}", flush=True)
    return value


result = least_squares(fun, np.zeros(NULL.shape[1]), jac=lambda z: np.asarray(jacobian(jnp.asarray(z))),
                       max_nfev=80, verbose=1)
optimized = np.asarray(currents(jnp.asarray(result.x)))
print(f"Optimization: {result.message} ({result.nfev} evaluations, {perf_counter() - time0:.0f} s)", flush=True)

""" Validation """
Lc1, hit1 = map(np.asarray, sol_lengths(jnp.asarray(optimized)))
Lc1_tight, _ = map(np.asarray, sol_lengths(jnp.asarray(optimized), tolerance=1e-12))
outer1, inner1 = [strike(jnp.asarray(optimized), t) for t in (OUTER_TARGET, INNER_TARGET)]


def native_lengths(x):
    from pyna.toroidal.flt import trace_wall_hits_twall_field
    from pyna.toroidal.geometry import ToroidalWall
    from essos.topology import essos_field_to_pyna_cylindrical_grid
    grid = essos_field_to_pyna_cylindrical_grid(field(jnp.asarray(x)), np.linspace(3.9512, 8.5512, 185),
                                                np.linspace(-4.6488, 4.8012, 379),
                                                np.linspace(0, 2 * np.pi, 4, endpoint=False), batch_size=4096)
    phi = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    vessel = ToroidalWall(phi, np.tile(WALL[:-1, 0], (4, 1)), np.tile(WALL[:-1, 1], (4, 1)))
    out = trace_wall_hits_twall_field(grid, SOL_R, np.full(len(SOL_R), MIDPLANE[1]), 0.0, 200, 0.002, vessel)
    return np.asarray(out["Lc_sum"]), bool(np.all(out["term_plus"] == 1) and np.all(out["term_minus"] == 1))


native0, _ = native_lengths(start)
native1, native_terminated = native_lengths(optimized)
summary = lambda s: dict(point_RZ_m=np.asarray(s[0]).tolist(), from_first_end_m=float(s[1]),  # noqa: E731
                         from_second_end_m=float(s[2]), flux_miss=float(s[3]), incidence_deg=float(np.rad2deg(s[4])))
report = dict(
    case="ITER divertor: near-SOL connection length with strikes on the vertical targets (frozen TokaMaker plasma)",
    coil_names=COIL_NAMES, reference_MAt=REFERENCE.tolist(), start_MAt=start.tolist(), optimized_MAt=optimized.tolist(),
    limits_MAt=LIMITS.tolist(), optimizer_message=result.message, function_evaluations=result.nfev,
    outer_strike=dict(start=summary(outer0), optimized=summary(outer1)),
    inner_strike=dict(start=summary(inner0), optimized=summary(inner1)),
    near_sol_mean_Lc_m=dict(start=float(np.mean(Lc0)), optimized=float(np.mean(Lc1)),
                            optimized_tight=float(np.mean(Lc1_tight)), native_start=float(np.mean(native0)),
                            native_optimized=float(np.mean(native1))),
    all_lines_hit_wall=dict(start=bool(hit0.all()), optimized=bool(hit1.all()), native=native_terminated),
    native_vs_essos_max_relative_difference=float(np.max(np.abs(native1 / Lc1_tight - 1.0))),
    isoflux_residual_max=float(np.max(np.abs(np.asarray(isoflux(jnp.asarray(optimized)))))),
    design=DESIGN, lambda_q_mm=1e3 * LAMBDA_Q, B_pol_omp_T=B_pol_omp, seed_weights=WEIGHTS.tolist(),
    heat_weighted_Lc_m=dict(start=float(np.exp(np.sum(WEIGHTS * np.log(Lc0)))),
                            optimized=float(np.exp(np.sum(WEIGHTS * np.log(Lc1_tight)))),
                            native_optimized=float(np.exp(np.sum(WEIGHTS * np.log(native1))))),
    xpoint_hessian_determinant=dict(start=float(jnp.linalg.det(jax.hessian(psi)(jnp.asarray(XPOINT), jnp.asarray(start)))),
                                    optimized=float(jnp.linalg.det(jax.hessian(psi)(jnp.asarray(XPOINT),
                                                                                  jnp.asarray(optimized))))),
    wall_gap_violation_max_Wb=dict(start=float(np.max(np.asarray(wall_gap_violation(jnp.asarray(start))))),
                                   optimized=float(np.max(np.asarray(wall_gap_violation(jnp.asarray(optimized)))))),
    seeds_R_m=SOL_R.tolist(), seeds_Z_m=float(MIDPLANE[1]), Lc_start_m=Lc0.tolist(), Lc_optimized_m=Lc1.tolist(),
    Lc_native_optimized_m=native1.tolist(), history=history, elapsed_seconds=perf_counter() - time0,
    source=DATA["source"], plasma_field_model_error=DATA["max_poloidal_field_error"],
    limitation="Frozen plasma current (no free-boundary response); coils as single circular filaments; "
               "approximate coil limits; no forces, vertical stability or heat-flux model.")
with open(os.path.join(OUTPUT, f"optimize_iter_divertor_{DESIGN}.json"), "w") as file:
    json.dump(report, file, indent=1)
print(json.dumps({k: v for k, v in report.items() if k not in ("history", "Lc_start_m", "Lc_optimized_m",
                                                             "Lc_native_optimized_m", "seeds_R_m")}, indent=1), flush=True)

""" Plot """
Rg, Zg = np.meshgrid(np.linspace(3.9, 8.6, 141), np.linspace(-4.7, 4.8, 286))
grid_points = jnp.c_[Rg.ravel(), Zg.ravel()]
fig, axes = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw=dict(width_ratios=[0.8, 1.2, 1.1]))
for x, color, label in ((start, "C0", "reference"), (optimized, "C3", "optimized")):
    values = np.asarray(jax.jit(jax.vmap(lambda p: psi(p, jnp.asarray(x))))(grid_points)).reshape(Rg.shape)
    level = float(psi(jnp.asarray(XPOINT), jnp.asarray(x)))
    for ax in axes[:2]:
        ax.contour(Rg, Zg, values, [level], colors=color)
        ax.contour(Rg, Zg, values, np.linspace(level - 1.5, level + 1.5, 13), colors=color, linewidths=0.3, alpha=0.5)
    for s in (outer0, inner0) if label == "reference" else (outer1, inner1):
        axes[1].plot(*np.asarray(s[0]), "*", color=color, ms=12)
for ax, lim in ((axes[0], (3.9, 8.6, -4.7, 4.8)), (axes[1], (4.0, 6.4, -4.65, -3.0))):
    ax.plot(*WALL.T, "k-", lw=1.2)
    ax.plot(*XPOINT, "kx")
    ax.set(aspect="equal", xlim=lim[:2], ylim=lim[2:], xlabel="R [m]", ylabel="Z [m]")
axes[0].set_title("ITER: reference (blue) and optimized (red)")
axes[1].set_title("divertor: separatrix and strike points")
axes[2].plot(1e3 * (SOL_R - MIDPLANE[0]), Lc0, "o-", ms=3, label="reference (ESSOS)")
axes[2].plot(1e3 * (SOL_R - MIDPLANE[0]), Lc1, "o-", ms=3, color="C3", label="optimized (ESSOS)")
axes[2].plot(1e3 * (SOL_R - MIDPLANE[0]), native1, "k.", label="optimized (Cyna)")
if DESIGN == "heat":
    axes[2].axvline(1e3 * LAMBDA_Q, color="0.6", ls=":", label=r"$\lambda_q$ (Eich)")
axes[2].set(xlabel="distance outside LCFS at outboard midplane [mm]", ylabel=r"$L_c$ [m]",
            title="Near scrape-off-layer connection length")
axes[2].legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, f"optimize_iter_divertor_{DESIGN}.png"), dpi=140)
