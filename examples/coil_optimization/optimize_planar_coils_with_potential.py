#  Encircling coils optimized together with a current potential (the idea behind single-stage planar-coil
#  design, Dudt et al., Fusion Eng. Des. 230 (2026) 115903): for any encircling coils, the best current
#  sheet on the shaping-coil surface is one linear solve, so the cost of the encircling coils is
#      J(coils) = min over Phi of  mean (B.n error)^2 + regularization * mean Phi^2 ,
#  a quadratic form r^T Q r in the normal field r the encircling coils leave. Minimizing J shapes, tilts
#  and powers the encircling coils so that the shaping coils have the least left to do. The sheet that
#  remains is then handed to discrete circular shaping coils.
#  Requires the optional packages vmex (wout reader) and virtual-casing-jax >= 0.0.10.
import json
import os
import sys
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import lsq_linear, minimize
from virtual_casing_jax import VirtualCasingJAX
from vmex.core.virtual_casing import surface_field_data_from_wout
from vmex.core.wout import read_wout

from essos.planar_coils import (CurrentPotential, FourierBoundary, convexity_loss, encircling_coils, minimum_distance,
                                offset_surface, separation_loss, shaping_coils)

jax.config.update("jax_enable_x64", True)

vmec_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "input_files", "wout_pwO_nfp4_aspect7_finite_beta.nc")
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "planar_coils_with_potential"

NPHI, NTHETA, VC_DIGITS = 32, 32, 4

""" Coil set, lengths in units of the minor radius (Helios values) """
N_ENCIRCLING = int(os.environ.get("N_ENCIRCLING", 3))                  # per half field period
ENCIRCLING_ORDER = int(os.environ.get("ENCIRCLING_ORDER", 4))
PLASMA_DISTANCE = float(os.environ.get("PLASMA_DISTANCE", 0.67))
ENCIRCLING_DISTANCE = 0.67
ENCIRCLING_SHAPING_DISTANCE = 0.34
SHAPING_DISTANCE = float(os.environ.get("SHAPING_DISTANCE", 0.08))
# Shaping-coil radii, largest first; smaller sizes fill the gaps the larger ones leave:
SHAPING_RADII = [float(v) for v in os.environ.get("SHAPING_RADII", os.environ.get("SHAPING_RADIUS", "0.2")).split(",")]
SHAPING_RADIUS = max(SHAPING_RADII)
# True scales each coil's current limit with its radius (same field at the conductor); False gives every coil the same limit:
LIMIT_SCALES_WITH_RADIUS = os.environ.get("LIMIT_SCALES_WITH_RADIUS", "1") == "1"
CURVATURE_LIMIT = 2.67
LENGTH_LIMIT = float(os.environ.get("LENGTH_LIMIT", 24.0))             # encircling-coil length (Helios: 20 minor radii)
REGULARIZATION = float(os.environ.get("REGULARIZATION", 1e-13))        # T^2 / A^2
PENALTY_WEIGHT = float(os.environ.get("PENALTY_WEIGHT", 1e-2))
FREE = tuple(os.environ.get("FREE", "shift,tilt,size,shape,current").split(","))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 2000))
CURRENT_LIMITS = (0.5, 1.0, 2.0, np.inf)                               # shaping current / mean encircling current
# The goal is the shortest coil set with a small boundary error: shaping coils are removed, weakest first, at this
# current limit, and the total coil length is reported against the error left:
PRUNE_LIMIT = float(os.environ.get("PRUNE_LIMIT", 0.5))                # Helios's limit scales to 0.44-0.60
PRUNE_FRACTION = 0.12
# The most shaping coils that can be afforded, counted over the whole torus (Helios has 324). The set pruned to
# this number is then refined: encircling coils re-optimized for these coils, then everything together.
SHAPING_BUDGET = int(os.environ.get("SHAPING_BUDGET", 324))

""" Field the coils must produce on the boundary """
wout = read_wout(vmec_file)
NFP, minor = int(wout.nfp), float(wout.Aminor_p)
data = surface_field_data_from_wout(wout, nphi=NPHI, ntheta=NTHETA)
vc = VirtualCasingJAX()
vc.setup(VC_DIGITS, NFP, False, NPHI, NTHETA, data.gamma, NPHI, NTHETA, NPHI, NTHETA)
B_external = vc.compute_external_B(data.B_total.reshape(3, -1), digits=VC_DIGITS)
half = slice(0, NPHI // 2)
points = jnp.moveaxis(data.gamma, 0, -1)[half].reshape(-1, 3)
B_total = jnp.moveaxis(data.B_total, 0, -1)[half].reshape(-1, 3)
B_target = jnp.moveaxis(B_external.reshape(3, NPHI, NTHETA), 0, -1)[half].reshape(-1, 3)
normals = jnp.moveaxis(data.normal, 0, -1)[half].reshape(-1, 3)
normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
B_norm = np.asarray(jnp.linalg.norm(B_total, axis=1))
normal_goal = jnp.sum(B_target * normals, axis=1)
print(f"NFP {NFP}, minor radius {minor:.4f} m, aspect {float(wout.aspect):.2f}, beta {float(wout.betatotal):.3%}")

""" Encircling coils and the sheet """
boundary = FourierBoundary(np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1], np.asarray(wout.xm), np.asarray(wout.xn))
poloidal_current = 2 * np.pi * float(np.mean(np.linalg.norm(np.asarray(points)[:, :2], axis=1) * B_norm)) / (4e-7 * np.pi)
clearance = (PLASMA_DISTANCE + ENCIRCLING_SHAPING_DISTANCE + SHAPING_RADIUS) * minor
encircling = encircling_coils(boundary, NFP, N_ENCIRCLING, clearance, poloidal_current / (2 * NFP * N_ENCIRCLING),
                              order=ENCIRCLING_ORDER)
parameters = encircling.initial_parameters()
unit = encircling.B(parameters, points)
parameters["current"] = parameters["current"] * float(jnp.vdot(unit, B_target) / jnp.vdot(unit, unit))
total_current = float(jnp.sum(parameters["current"]))
mean_current = abs(float(jnp.mean(encircling.currents(parameters))))

sheet = CurrentPotential(offset_surface(boundary, PLASMA_DISTANCE * minor), NFP, 10, 10)
A = sheet.normal_field_matrix(points, normals)
n_points = A.shape[0]
R = sheet.basis.T @ (sheet.basis * (sheet.area / sheet.area.sum())[:, None])
M = A.T @ A / n_points + REGULARIZATION * R
solve_sheet = np.linalg.solve(M, A.T / n_points)                       # normal field left -> best coefficients
Q = jnp.asarray((np.eye(n_points) - A @ solve_sheet) / n_points)
field_scale = float(np.mean(B_norm**2))
surface_points = points[None]


def left(p):
    """Normal field the encircling coils leave for the sheet."""
    return normal_goal - jnp.sum(encircling.B(p, points) * normals, axis=1)


def describe(label, p):
    r = np.asarray(left(p))
    coefficients = solve_sheet @ r
    potential = sheet.potential(coefficients)
    after = np.abs(r - A @ coefficients) / B_norm
    gamma, _, _ = encircling.expanded(p)
    kappa = encircling.signed_curvature(p)
    row = dict(encircling_only=[float(np.mean(np.abs(r) / B_norm)), float(np.max(np.abs(r) / B_norm))],
               with_sheet=[float(after.mean()), float(after.max())],
               potential_max_over_encircling=float(np.abs(potential).max() / mean_current),
               potential_rms_over_encircling=float(np.sqrt(np.sum(potential**2 * sheet.area) / sheet.area.sum()) / mean_current),
               curvature_a=[float(jnp.min(kappa) * minor), float(jnp.max(kappa) * minor)],
               encircling_encircling_a=float(minimum_distance(gamma, gamma, same=True) / minor),
               encircling_plasma_a=float(minimum_distance(gamma, surface_points) / minor),
               currents_over_mean=np.round(np.asarray(encircling.currents(p)) / mean_current, 2).tolist(),
               length_a=np.round(np.asarray(encircling.length(p)) / minor, 1).tolist())
    print(f"{label}: " + ", ".join(f"{k} {np.round(v, 4).tolist() if not isinstance(v, float) else round(v, 4)}" for k, v in row.items()),
          flush=True)
    return row, coefficients


results = dict(minor_radius=minor, mean_encircling_current_kA=mean_current / 1e3, regularization=REGULARIZATION)
results["circles"], _ = describe("circles", parameters)

""" Encircling coils that leave the sheet the least to do """
x0, unravel = ravel_pytree({k: parameters[k] for k in FREE})


def objective(x):
    p = parameters | unravel(x)
    r = left(p)
    gamma, _, _ = encircling.expanded(p)
    base = encircling.gamma(p)
    limits = (convexity_loss(encircling.signed_curvature(p), CURVATURE_LIMIT / minor)
              + separation_loss(base, gamma[len(encircling):], ENCIRCLING_DISTANCE * minor)
              + separation_loss(base, base, ENCIRCLING_DISTANCE * minor, same=True)
              + separation_loss(base, surface_points, (PLASMA_DISTANCE + ENCIRCLING_SHAPING_DISTANCE) * minor)
              + jnp.mean(jnp.maximum(encircling.length(p) / (LENGTH_LIMIT * minor) - 1.0, 0.0) ** 2)
              + ((jnp.sum(p["current"]) - total_current) / total_current) ** 2)
    return r @ Q @ r / field_scale + PENALTY_WEIGHT * limits


value_and_grad = jax.jit(jax.value_and_grad(objective))
t_start = time()
result = minimize(lambda x: tuple(np.asarray(v, dtype=float) for v in value_and_grad(jnp.asarray(x))), np.asarray(x0),
                  jac=True, method="L-BFGS-B", options=dict(maxiter=MAX_ITERATIONS, maxcor=30, ftol=1e-15, gtol=1e-12))
parameters = parameters | unravel(jnp.asarray(result.x))
print(f"optimization took {time() - t_start:.1f} s, {result.nit} iterations: {result.message}")
results["optimized"], coefficients = describe("optimized", parameters)

""" Discrete shaping coils for what is left """
shaping = shaping_coils(boundary, NFP, PLASMA_DISTANCE * minor, np.array(SHAPING_RADII) * minor, SHAPING_DISTANCE * minor, mean_current)
relative_size = shaping.radius / shaping.radius.max() if LIMIT_SCALES_WITH_RADIUS else np.ones(len(shaping))
p_shaping = shaping.initial_parameters()
normal_of_coil = jax.jit(lambda c: jnp.sum(shaping.B(p_shaping | dict(current=c), points) * normals, axis=1))
coil_matrix = np.stack([np.asarray(normal_of_coil(jnp.zeros(len(shaping)).at[i].set(1.0))) for i in range(len(shaping))], axis=1)
r = np.asarray(left(parameters))
handed = sheet.coil_currents(coefficients, shaping) / mean_current
after = np.abs(r - coil_matrix @ handed) / B_norm
sizes = ", ".join(f"{int(np.sum(np.isclose(shaping.radius, r * minor)))} of radius {r} a" for r in sorted(SHAPING_RADII, reverse=True))
print(f"{len(shaping)} shaping coils per half period ({2 * NFP * len(shaping)} in all): {sizes}; "
      f"current limits {'scale with the radius' if LIMIT_SCALES_WITH_RADIUS else 'are equal'}")
print(f"  carrying the sheet: mean B.n {after.mean():.3%}, max {after.max():.3%}, max current {np.abs(handed).max():.2f}, rms {np.sqrt(np.mean(handed**2)):.2f} (of the mean encircling current)")
results["shaping"] = dict(count=len(shaping), from_sheet=[float(after.mean()), float(after.max()), float(np.abs(handed).max())], bounded=[])
for limit in CURRENT_LIMITS:
    solution = lsq_linear(coil_matrix, r, bounds=(-limit * relative_size, limit * relative_size), max_iter=5000)
    after = np.abs(r - coil_matrix @ solution.x) / B_norm
    results["shaping"]["bounded"].append(dict(limit=float(limit), mean=float(after.mean()), max=float(after.max()),
                                              at_limit=int(np.sum(np.abs(solution.x) > 0.999 * limit * relative_size)),
                                              rms=float(np.sqrt(np.mean(solution.x**2))), max_current=float(np.abs(solution.x).max())))
    print(f"  bounded fit, limit {limit:5.2f}: mean B.n {after.mean():.3%}, max {after.max():.3%}, "
          f"{results['shaping']['bounded'][-1]['at_limit']} at the limit, rms {results['shaping']['bounded'][-1]['rms']:.2f}, "
          f"max {results['shaping']['bounded'][-1]['max_current']:.2f}")

""" Shortest coil set: drop the shaping coils that do the least per metre of conductor, and refit the rest """
n_copies = 2 * NFP                                                     # each base coil stands for this many coils
encircling_length = n_copies * float(jnp.sum(encircling.length(parameters)))
encircling_ampere_metres = n_copies * float(jnp.sum(jnp.abs(encircling.currents(parameters)) * encircling.length(parameters)))
circumference = 2 * np.pi * shaping.radius
active = np.arange(len(shaping))
budget_size = min(len(shaping), SHAPING_BUDGET // n_copies)             # base coils within the budget
results["pruning"] = dict(limit=PRUNE_LIMIT, encircling_length_m=encircling_length, rows=[])
print(f"\nencircling coils: {n_copies * len(encircling)} coils, {encircling_length:.1f} m ({encircling_length / minor:.0f} a), "
      f"{encircling_ampere_metres / 1e6:.2f} MA m. Pruning the shaping coils at current limit {PRUNE_LIMIT}:")
print(" shaping coils (all) | shaping length [m] | total length [m] ([a]) | total MA m | mean B.n | max B.n | at limit")
while True:
    bound = PRUNE_LIMIT * relative_size[active]
    solution = lsq_linear(coil_matrix[:, active], r, bounds=(-bound, bound), max_iter=5000)
    after = np.abs(r - coil_matrix[:, active] @ solution.x) / B_norm
    shaping_length = n_copies * float(np.sum(circumference[active]))
    ampere_metres = encircling_ampere_metres + n_copies * float(np.sum(np.abs(solution.x) * mean_current * circumference[active]))
    row = dict(coils=int(n_copies * active.size), shaping_length_m=shaping_length, total_length_m=encircling_length + shaping_length,
               total_length_a=(encircling_length + shaping_length) / minor, total_MA_m=ampere_metres / 1e6,
               mean=float(after.mean()), max=float(after.max()), at_limit=int(np.sum(np.abs(solution.x) > 0.999 * bound)))
    results["pruning"]["rows"].append(row)
    print(f"   {row['coils']:6d}            |   {shaping_length:8.1f}         |  {row['total_length_m']:7.1f} ({row['total_length_a']:5.0f})     |  {row['total_MA_m']:6.2f}   "
          f"| {row['mean']:7.3%} | {row['max']:7.3%} | {row['at_limit']}")
    if n_copies * active.size == budget_size * n_copies:
        budget_set, budget_currents = active.copy(), solution.x.copy()
    if active.size <= 4:
        break
    worth = np.abs(solution.x) * shaping.radius[active]                # moment per metre of conductor
    keep = active.size - max(1, int(PRUNE_FRACTION * active.size))
    if active.size > budget_size > keep:
        keep = budget_size                                             # land exactly on the budget
    active = active[np.sort(np.argsort(worth)[active.size - keep:])]
after = np.abs(r) / B_norm
print(f"        0            |        0.0         |  {encircling_length:7.1f} ({encircling_length / minor:5.0f})     |  {encircling_ampere_metres / 1e6:6.2f}   | {after.mean():7.3%} | {after.max():7.3%} |")

""" The budget set, refined """
from essos.planar_coils import PlanarCoils  # noqa: E402

chosen = PlanarCoils(shaping.centres[budget_set], shaping.frames[budget_set, 2], shaping.radius[budget_set], mean_current,
                     nfp=NFP, stellsym=True, n_segments=shaping.n_segments)
p_chosen = chosen.initial_parameters() | dict(current=jnp.asarray(budget_currents))
chosen_size = relative_size[budget_set]
C = coil_matrix[:, budget_set]


def summary(label, p_enc, p_shp):
    left_over = np.asarray(left(p_enc) - jnp.sum(chosen.B(p_shp, points) * normals, axis=1))
    g_enc, _, _ = encircling.expanded(p_enc)
    g_shp, _, _ = chosen.expanded(p_shp)
    kappa = encircling.signed_curvature(p_enc)
    length = n_copies * (float(jnp.sum(encircling.length(p_enc))) + float(np.sum(2 * np.pi * chosen.radius)))
    row = dict(mean=float(np.mean(np.abs(left_over) / B_norm)), max=float(np.max(np.abs(left_over) / B_norm)),
               total_length_m=length, shaping_current_max=float(jnp.max(jnp.abs(p_shp["current"]))),
               curvature_a=[float(jnp.min(kappa) * minor), float(jnp.max(kappa) * minor)],
               encircling_length_a=np.round(np.asarray(encircling.length(p_enc)) / minor, 1).tolist(),
               encircling_currents=np.round(np.asarray(encircling.currents(p_enc)) / mean_current, 2).tolist(),
               encircling_encircling_a=float(minimum_distance(g_enc, g_enc, same=True) / minor),
               encircling_shaping_a=float(minimum_distance(g_enc, g_shp) / minor),
               shaping_shaping_a=float(minimum_distance(g_shp, g_shp, same=True) / minor),
               shaping_plasma_a=float(minimum_distance(g_shp, surface_points) / minor),
               encircling_plasma_a=float(minimum_distance(g_enc, surface_points) / minor))
    print(f"{label}: " + ", ".join(f"{k} {np.round(v, 4).tolist() if not isinstance(v, float) else round(v, 4)}" for k, v in row.items()),
          flush=True)
    return row


print(f"\nBudget: {n_copies * len(chosen)} shaping coils ({len(chosen)} per half period), current limit {PRUNE_LIMIT}")
results["budget"] = dict(count=n_copies * len(chosen), limit=PRUNE_LIMIT, pruned=summary("pruned", parameters, p_chosen))

# Encircling coils re-optimized for these shaping coils: their currents are eliminated like the sheet was,
# with the ridge that keeps the largest current near the limit at the start.
for ridge in 10.0 ** np.arange(-8.0, 0.5, 0.5):
    solve_coils = np.linalg.solve(C.T @ C / n_points + ridge * field_scale * np.eye(C.shape[1]), C.T / n_points)
    if np.max(np.abs(solve_coils @ np.asarray(left(parameters))) / chosen_size) <= PRUNE_LIMIT:
        break
Q_coils = jnp.asarray((np.eye(n_points) - C @ solve_coils) / n_points)
Q_sheet, Q = Q, Q_coils                                               # the objective above now sees the coils
value_and_grad = jax.jit(jax.value_and_grad(objective))
x0, unravel = ravel_pytree({k: parameters[k] for k in FREE})
result = minimize(lambda x: tuple(np.asarray(v, dtype=float) for v in value_and_grad(jnp.asarray(x))), np.asarray(x0),
                  jac=True, method="L-BFGS-B", options=dict(maxiter=MAX_ITERATIONS, maxcor=30, ftol=1e-15, gtol=1e-12))
parameters = parameters | unravel(jnp.asarray(result.x))
bound = PRUNE_LIMIT * chosen_size
p_chosen = p_chosen | dict(current=jnp.asarray(lsq_linear(C, np.asarray(left(parameters)), bounds=(-bound, bound), max_iter=5000).x))
results["budget"]["encircling_refit"] = summary(f"encircling coils refit (ridge {ridge:.0e})", parameters, p_chosen)

# Everything together: encircling coils, and each shaping coil's position, tilt and current.
free_all = dict(encircling={k: parameters[k] for k in FREE}, shaping={k: p_chosen[k] for k in ("shift", "tilt", "current")})
x0, unravel = ravel_pytree(free_all)


def joint(x):
    moved = unravel(x)
    p_enc, p_shp = parameters | moved["encircling"], p_chosen | moved["shaping"]
    error = (left(p_enc) - jnp.sum(chosen.B(p_shp, points) * normals, axis=1))
    g_enc, _, _ = encircling.expanded(p_enc)
    g_shp, _, _ = chosen.expanded(p_shp)
    base_enc, base_shp = encircling.gamma(p_enc), chosen.gamma(p_shp)
    over = jnp.maximum(jnp.abs(p_shp["current"]) / jnp.asarray(bound) - 1.0, 0.0)
    limits = (convexity_loss(encircling.signed_curvature(p_enc), CURVATURE_LIMIT / minor)
              + separation_loss(base_enc, g_enc[len(encircling):], ENCIRCLING_DISTANCE * minor)
              + separation_loss(base_enc, base_enc, ENCIRCLING_DISTANCE * minor, same=True)
              + separation_loss(base_enc, surface_points, (PLASMA_DISTANCE + ENCIRCLING_SHAPING_DISTANCE) * minor)
              + separation_loss(base_enc, g_shp, ENCIRCLING_SHAPING_DISTANCE * minor)
              + separation_loss(base_shp, g_shp[len(chosen):], SHAPING_DISTANCE * minor)
              + separation_loss(base_shp, base_shp, SHAPING_DISTANCE * minor, same=True)
              + separation_loss(base_shp, surface_points, PLASMA_DISTANCE * minor)
              + jnp.mean(jnp.maximum(encircling.length(p_enc) / (LENGTH_LIMIT * minor) - 1.0, 0.0) ** 2)
              + jnp.mean(over * over)
              + ((jnp.sum(p_enc["current"]) - total_current) / total_current) ** 2)
    return jnp.mean(error * error) / field_scale + PENALTY_WEIGHT * limits


joint_value_and_grad = jax.jit(jax.value_and_grad(joint))
t_start = time()
result = minimize(lambda x: tuple(np.asarray(v, dtype=float) for v in joint_value_and_grad(jnp.asarray(x))), np.asarray(x0),
                  jac=True, method="L-BFGS-B", options=dict(maxiter=2 * MAX_ITERATIONS, maxcor=30, ftol=1e-15, gtol=1e-12))
moved = unravel(jnp.asarray(result.x))
parameters, p_chosen = parameters | moved["encircling"], p_chosen | moved["shaping"]
print(f"joint refinement took {time() - t_start:.1f} s, {result.nit} iterations")
results["budget"]["joint"] = summary("joint", parameters, p_chosen)
shaping_gamma, shaping_gamma_dash, shaping_currents = chosen.expanded(p_chosen)

gamma, gamma_dash, currents = encircling.expanded(parameters)
np.savez(f"{OUTPUT}.npz", encircling_gamma=gamma, encircling_gamma_dash=gamma_dash, encircling_currents=currents,
         potential=sheet.potential(coefficients), sheet_points=sheet.points, coefficients=coefficients, modes=sheet.modes,
         shaping_gamma=shaping_gamma, shaping_gamma_dash=shaping_gamma_dash, shaping_currents=shaping_currents,
         points=points, normals=normals,
         **{f"encircling_{k}": np.asarray(v) for k, v in parameters.items()})
with open(f"{OUTPUT}.json", "w") as stream:
    json.dump(results, stream, indent=1)
print(f"Wrote {OUTPUT}.npz and {OUTPUT}.json")
