#  Planar coils for a finite-beta VMEC/VMEX equilibrium: convex encircling (toroidal-field) coils plus an
#  array of small circular shaping (dipole) coils, as in planar-coil stellarator designs (Kruger et al.,
#  Fusion Eng. Des. 230 (2026) 115893).
#  The coils must reproduce B_total - B_plasma on the boundary; virtual_casing_jax gives that field.
#  Step 1 fits the currents alone (a bounded linear least-squares problem: the shapes are the seed circles).
#  Step 2 frees the encircling-coil planes and shapes, and optionally the shaping-coil positions, under the
#  engineering limits: convex encircling coils of limited curvature, and minimum distances between
#  encircling coils, between shaping coils, between the two families and to the plasma.
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

from essos.planar_coils import (FourierBoundary, convexity_loss, encircling_coils, minimum_distance,
                                separation_loss, shaping_coils)

jax.config.update("jax_enable_x64", True)

vmec_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "input_files", "wout_QH_simple_finite_scaled.nc")
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "planar_coils"

""" Virtual casing grid per field period and accuracy (see optimize_coils_virtual_casing.py) """
NPHI, NTHETA, VC_DIGITS = 32, 32, 4

""" Coil set. Lengths are in units of the minor radius a of the equilibrium, as scaled from the Helios
    design (a = 1.78 m): coil-plasma and encircling-encircling 1.2 m, encircling-shaping 0.6 m,
    shaping-shaping 0.3 m, shaping radius 0.67 m, encircling curvature <= 1.5 /m """
N_ENCIRCLING = int(os.environ.get("N_ENCIRCLING", 3))      # per half field period
ENCIRCLING_ORDER = 4
PLASMA_DISTANCE = 0.67
ENCIRCLING_DISTANCE = 0.67
ENCIRCLING_SHAPING_DISTANCE = 0.34
SHAPING_DISTANCE = 0.17
SHAPING_RADIUS = 0.376
CURVATURE_LIMIT = 2.67                                      # times 1/a
SHAPING_CURRENT_LIMIT = float(os.environ.get("SHAPING_CURRENT_LIMIT", 0.22))  # of the mean encircling current
FREE_SHAPING_GEOMETRY = os.environ.get("FREE_SHAPING_GEOMETRY", "0") == "1"   # also move and tilt the shaping coils
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 600))

""" Weights of step 2 (the field term is the mean squared |dB|/|B|) """
CONVEXITY_WEIGHT, DISTANCE_WEIGHT, CURRENT_WEIGHT = 1e-2, 1e-2, 1e-2

""" Field the coils must produce on the boundary: B_external = B_total - B_plasma """
wout = read_wout(vmec_file)
NFP = int(wout.nfp)
minor = float(wout.Aminor_p)
t_start = time()
data = surface_field_data_from_wout(wout, nphi=NPHI, ntheta=NTHETA)      # one field period, (3, nphi, ntheta)
vc = VirtualCasingJAX()
vc.setup(VC_DIGITS, NFP, False, NPHI, NTHETA, data.gamma, NPHI, NTHETA, NPHI, NTHETA)
B_external = vc.compute_external_B(data.B_total.reshape(3, -1), digits=VC_DIGITS)
half = slice(0, NPHI // 2)                                               # stellarator symmetry: half a period
points = jnp.moveaxis(data.gamma, 0, -1)[half].reshape(-1, 3)
B_total = jnp.moveaxis(data.B_total, 0, -1)[half].reshape(-1, 3)
B_target = jnp.moveaxis(B_external.reshape(3, NPHI, NTHETA), 0, -1)[half].reshape(-1, 3)
normals = jnp.moveaxis(data.normal, 0, -1)[half].reshape(-1, 3)
normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
B_norm = jnp.linalg.norm(B_total, axis=1)
print(f"NFP {NFP}, minor radius {minor:.4f} m, aspect {float(wout.aspect):.2f}, beta {float(wout.betatotal):.3%}; "
      f"virtual casing {time() - t_start:.1f} s; plasma-current share of the boundary field "
      f"{float(jnp.mean(jnp.linalg.norm(B_total - B_target, axis=1) / B_norm)):.3f}")

""" Seed coils """
boundary = FourierBoundary(np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1], np.asarray(wout.xm), np.asarray(wout.xn))
shaping_offset = PLASMA_DISTANCE * minor
encircling_clearance = (PLASMA_DISTANCE + ENCIRCLING_SHAPING_DISTANCE + SHAPING_RADIUS) * minor
poloidal_current = 2 * np.pi * float(jnp.mean(jnp.linalg.norm(points[:, :2], axis=1) * B_norm)) / (4e-7 * np.pi)
encircling_current = poloidal_current / (2 * NFP * N_ENCIRCLING)
encircling = encircling_coils(boundary, NFP, N_ENCIRCLING, encircling_clearance, encircling_current, order=ENCIRCLING_ORDER)
shaping = shaping_coils(boundary, NFP, shaping_offset, SHAPING_RADIUS * minor, SHAPING_DISTANCE * minor, encircling_current)
parameters = dict(encircling=encircling.initial_parameters(), shaping=shaping.initial_parameters())
# The encircling current may come out with either sign: fit it once to the target.
unit = encircling.B(parameters["encircling"], points)
sign = float(jnp.vdot(unit, B_target) / jnp.vdot(unit, unit))
parameters["encircling"]["current"] = parameters["encircling"]["current"] * sign
print(f"{len(encircling)} encircling + {len(shaping)} shaping coils per half period "
      f"({2 * NFP * len(encircling)} + {2 * NFP * len(shaping)} in all); encircling current {sign * encircling_current / 1e3:.1f} kA, "
      f"radius {np.round(encircling.radius, 3)} m; shaping radius {SHAPING_RADIUS * minor:.3f} m at {shaping_offset:.3f} m")

surface_points = points[None]                                            # as one "coil" for the distance terms


def field(p):
    return encircling.B(p["encircling"], points) + shaping.B(p["shaping"], points)


def field_error(p):
    return (field(p) - B_target) / B_norm[:, None]


def report(label, p):
    error = field_error(p)
    normal_error = jnp.abs(jnp.sum(error * normals, axis=1))
    g_enc, _, _ = encircling.expanded(p["encircling"])
    g_shp, _, _ = shaping.expanded(p["shaping"])
    kappa = encircling.signed_curvature(p["encircling"])
    row = dict(
        bn_mean=float(jnp.mean(normal_error)), bn_max=float(jnp.max(normal_error)),
        dB_mean=float(jnp.mean(jnp.linalg.norm(error, axis=1))), dB_max=float(jnp.max(jnp.linalg.norm(error, axis=1))),
        curvature_min_a=float(jnp.min(kappa) * minor), curvature_max_a=float(jnp.max(kappa) * minor),
        encircling_encircling_a=float(minimum_distance(g_enc, g_enc, same=True) / minor),
        encircling_shaping_a=float(minimum_distance(g_enc, g_shp) / minor),
        shaping_shaping_a=float(minimum_distance(g_shp, g_shp, same=True) / minor),
        encircling_plasma_a=float(minimum_distance(g_enc, surface_points) / minor),
        shaping_plasma_a=float(minimum_distance(g_shp, surface_points) / minor),
        encircling_current_kA=np.round(np.asarray(encircling.currents(p["encircling"])) / 1e3, 1).tolist(),
        shaping_current_max_kA=float(jnp.max(jnp.abs(shaping.currents(p["shaping"]))) / 1e3),
        shaping_current_rms_kA=float(jnp.sqrt(jnp.mean(shaping.currents(p["shaping"]) ** 2)) / 1e3),
        encircling_length_a=np.round(np.asarray(encircling.length(p["encircling"])) / minor, 2).tolist())
    print(f"{label}: " + ", ".join(f"{k} {v:.3e}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    return row


results = {"seed": report("seed", parameters)}

""" Step 1: currents alone, bounded linear least squares """
t_start = time()


def currents_of(p):
    return jnp.concatenate((p["encircling"]["current"], p["shaping"]["current"]))


def with_currents(c):
    n = len(encircling)
    return dict(encircling=parameters["encircling"] | dict(current=c[:n]), shaping=parameters["shaping"] | dict(current=c[n:]))


# The field is linear in the currents: one evaluation per unit current gives the matrix. (Forward-mode
# differentiation of the same map returned a wrong Jacobian on a GPU; reverse mode and this agree.)
unit_field = jax.jit(lambda c: (field(with_currents(c)) / B_norm[:, None]).ravel())
n_currents = len(encircling) + len(shaping)
matrix = np.stack([np.asarray(unit_field(jnp.zeros(n_currents).at[i].set(1.0))) for i in range(n_currents)], axis=1)
target = np.asarray((B_target / B_norm[:, None]).ravel())
limit = SHAPING_CURRENT_LIMIT * abs(sign)
lower = np.concatenate((np.full(len(encircling), -np.inf), np.full(len(shaping), -limit)))
solution = lsq_linear(matrix, target, bounds=(lower, -lower), max_iter=2000)
parameters = with_currents(jnp.asarray(solution.x))
print(f"step 1 took {time() - t_start:.1f} s; {int(np.sum(np.abs(solution.x[len(encircling):]) > 0.999 * limit))} "
      f"of {len(shaping)} shaping currents at the limit")
results["currents"] = report("currents", parameters)

""" Step 2: encircling planes and shapes (and shaping positions) under the engineering limits """
free = dict(encircling=("shift", "tilt", "size", "shape", "current"),
            shaping=("shift", "tilt", "current") if FREE_SHAPING_GEOMETRY else ("current",))
x0, unravel = ravel_pytree({family: {k: parameters[family][k] for k in keys} for family, keys in free.items()})


def assemble(x):
    moved = unravel(x)
    return {family: parameters[family] | moved[family] for family in parameters}


def objective(x):
    p = assemble(x)
    error = field_error(p)
    g_enc, _, _ = encircling.expanded(p["encircling"])
    g_shp, _, _ = shaping.expanded(p["shaping"])
    base_shaping = shaping.gamma(p["shaping"])
    distances = (separation_loss(encircling.gamma(p["encircling"]), g_enc[len(encircling):], ENCIRCLING_DISTANCE * minor)
                 + separation_loss(encircling.gamma(p["encircling"]), encircling.gamma(p["encircling"]),
                                   ENCIRCLING_DISTANCE * minor, same=True)
                 + separation_loss(encircling.gamma(p["encircling"]), g_shp, ENCIRCLING_SHAPING_DISTANCE * minor)
                 + separation_loss(encircling.gamma(p["encircling"]), surface_points, PLASMA_DISTANCE * minor))
    if FREE_SHAPING_GEOMETRY:
        distances = distances + (separation_loss(base_shaping, g_shp[len(shaping):], SHAPING_DISTANCE * minor)
                                 + separation_loss(base_shaping, base_shaping, SHAPING_DISTANCE * minor, same=True)
                                 + separation_loss(base_shaping, surface_points, PLASMA_DISTANCE * minor))
    over = jnp.maximum(jnp.abs(p["shaping"]["current"]) - limit, 0.0) / limit
    return (jnp.mean(jnp.sum(error * error, axis=1))
            + CONVEXITY_WEIGHT * convexity_loss(encircling.signed_curvature(p["encircling"]), CURVATURE_LIMIT / minor)
            + DISTANCE_WEIGHT * distances + CURRENT_WEIGHT * jnp.mean(over * over))


value_and_grad = jax.jit(jax.value_and_grad(objective))
t_start = time()
result = minimize(lambda x: tuple(np.asarray(v, dtype=float) for v in value_and_grad(jnp.asarray(x))), np.asarray(x0),
                  jac=True, method="L-BFGS-B", options=dict(maxiter=MAX_ITERATIONS, maxcor=30, ftol=1e-15, gtol=1e-12))
parameters = assemble(jnp.asarray(result.x))
print(f"step 2 took {time() - t_start:.1f} s, {result.nit} iterations: {result.message}")
results["shapes"] = report("shapes", parameters)

""" Save: every coil after the symmetries, and the variables """
g_enc, d_enc, c_enc = encircling.expanded(parameters["encircling"])
g_shp, d_shp, c_shp = shaping.expanded(parameters["shaping"])
np.savez(f"{OUTPUT}.npz", encircling_gamma=g_enc, encircling_gamma_dash=d_enc, encircling_currents=c_enc,
         shaping_gamma=g_shp, shaping_gamma_dash=d_shp, shaping_currents=c_shp, points=points, B_target=B_target,
         B_coils=field(parameters), normals=normals,
         **{f"{family}_{k}": np.asarray(v) for family, p in parameters.items() for k, v in p.items()})
with open(f"{OUTPUT}.json", "w") as stream:
    json.dump(dict(wout=os.path.abspath(vmec_file), nfp=NFP, minor_radius=minor, results=results), stream, indent=1)
print(f"Wrote {OUTPUT}.npz and {OUTPUT}.json")
