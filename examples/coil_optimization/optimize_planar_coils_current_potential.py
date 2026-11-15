#  How much can an array of shaping (dipole) coils do, and how should its currents be chosen?
#  The encircling coils are fixed circles carrying the poloidal current. The rest of the normal field the
#  plasma needs (virtual casing) is then fitted in three ways and compared:
#    1. a current potential on the shaping-coil surface (the continuous limit of the array, one linear solve
#       per regularization): the best any array on that surface can do, and the current it takes;
#    2. the same potential handed to the discrete circular coils (each coil takes the dipole moment of the
#       surface cells nearest to it);
#    3. a direct bounded least-squares fit of the coil currents, for several current limits.
#  Requires the optional packages vmex (wout reader) and virtual-casing-jax >= 0.0.10.
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear
from virtual_casing_jax import VirtualCasingJAX
from vmex.core.virtual_casing import surface_field_data_from_wout
from vmex.core.wout import read_wout

from essos.planar_coils import CurrentPotential, FourierBoundary, encircling_coils, offset_surface, shaping_coils

jax.config.update("jax_enable_x64", True)

vmec_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "input_files", "wout_pwO_nfp4_aspect7_finite_beta.nc")
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "planar_coils_current_potential"

NPHI, NTHETA, VC_DIGITS = 32, 32, 4

""" Coil set, lengths in units of the minor radius (Helios values: 0.67, 0.34, 0.17, 0.376) """
N_ENCIRCLING = int(os.environ.get("N_ENCIRCLING", 3))                  # per half field period
PLASMA_DISTANCE = float(os.environ.get("PLASMA_DISTANCE", 0.67))       # shaping-coil surface to plasma
ENCIRCLING_SHAPING_DISTANCE = 0.34
SHAPING_DISTANCE = float(os.environ.get("SHAPING_DISTANCE", 0.17))
SHAPING_RADIUS = float(os.environ.get("SHAPING_RADIUS", 0.376))
POTENTIAL_MPOL, POTENTIAL_NTOR = int(os.environ.get("POTENTIAL_MPOL", 10)), int(os.environ.get("POTENTIAL_NTOR", 10))
REGULARIZATIONS = 10.0 ** np.arange(-17.0, -9.5, 1.0)                 # T^2 / A^2
CURRENT_LIMITS = (0.22, 0.5, 1.0, 2.0, np.inf)                         # of the encircling-coil current

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
print(f"NFP {NFP}, minor radius {minor:.4f} m, aspect {float(wout.aspect):.2f}, beta {float(wout.betatotal):.3%}")

""" Encircling coils: circles with the poloidal current """
boundary = FourierBoundary(np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1], np.asarray(wout.xm), np.asarray(wout.xn))
poloidal_current = 2 * np.pi * float(np.mean(np.linalg.norm(np.asarray(points)[:, :2], axis=1) * B_norm)) / (4e-7 * np.pi)
encircling_current = poloidal_current / (2 * NFP * N_ENCIRCLING)
clearance = (PLASMA_DISTANCE + ENCIRCLING_SHAPING_DISTANCE + SHAPING_RADIUS) * minor
encircling = encircling_coils(boundary, NFP, N_ENCIRCLING, clearance, encircling_current, order=0)
p_encircling = encircling.initial_parameters()
unit = encircling.B(p_encircling, points)
p_encircling["current"] = p_encircling["current"] * float(jnp.vdot(unit, B_target) / jnp.vdot(unit, unit))
encircling_current = float(jnp.abs(encircling.currents(p_encircling))[0])
normal_target = np.asarray(jnp.sum((B_target - encircling.B(p_encircling, points)) * normals, axis=1))


def error(normal_field):
    """Mean and maximum of |B.n| / |B| left on the boundary."""
    relative = np.abs(normal_target - np.asarray(normal_field)) / B_norm
    return float(relative.mean()), float(relative.max())


results = dict(minor_radius=minor, encircling_current_kA=encircling_current / 1e3, encircling_only=error(0.0))
print(f"{2 * NFP * N_ENCIRCLING} circular encircling coils of {encircling_current / 1e3:.1f} kA: "
      f"B.n/B mean {results['encircling_only'][0]:.3%}, max {results['encircling_only'][1]:.3%}")

""" Shaping coils on the surface PLASMA_DISTANCE outside the plasma """
shaping = shaping_coils(boundary, NFP, PLASMA_DISTANCE * minor, SHAPING_RADIUS * minor, SHAPING_DISTANCE * minor,
                        encircling_current)
p_shaping = shaping.initial_parameters()
normal_of_coil = jax.jit(lambda c: jnp.sum(shaping.B(p_shaping | dict(current=c), points) * normals, axis=1))
coil_matrix = np.stack([np.asarray(normal_of_coil(jnp.zeros(len(shaping)).at[i].set(1.0))) for i in range(len(shaping))], axis=1)
print(f"{len(shaping)} shaping coils per half period ({2 * NFP * len(shaping)} in all), radius {SHAPING_RADIUS * minor:.3f} m, "
      f"{PLASMA_DISTANCE * minor:.3f} m from the plasma; currents below in units of the encircling current")

""" 1 and 2: current potential, and the coils that carry it """
sheet = CurrentPotential(offset_surface(boundary, PLASMA_DISTANCE * minor), NFP, POTENTIAL_MPOL, POTENTIAL_NTOR)
sheet_matrix = sheet.normal_field_matrix(points, normals)
results["potential"] = []
print("\n regularization | sheet: mean B.n  max B.n  max Phi[kA]  rms Phi[kA] | coils from the sheet: mean B.n  max B.n  max I  rms I")
for regularization in REGULARIZATIONS:
    coefficients = sheet.solve(sheet_matrix, normal_target, regularization)
    potential = sheet.potential(coefficients)
    currents = sheet.coil_currents(coefficients, shaping) / encircling_current
    row = dict(regularization=float(regularization), sheet=error(sheet_matrix @ coefficients),
               potential_max_kA=float(np.abs(potential).max() / 1e3),
               potential_rms_kA=float(np.sqrt(np.sum(potential**2 * sheet.area) / sheet.area.sum()) / 1e3),
               coils=error(coil_matrix @ currents), current_max=float(np.abs(currents).max()),
               current_rms=float(np.sqrt(np.mean(currents**2))))
    results["potential"].append(row)
    print(f"   {regularization:9.0e}   |  {row['sheet'][0]:8.3%} {row['sheet'][1]:8.3%}  {row['potential_max_kA']:9.1f}  "
          f"{row['potential_rms_kA']:9.1f}   |   {row['coils'][0]:8.3%} {row['coils'][1]:8.3%}  {row['current_max']:6.2f} {row['current_rms']:6.2f}")

""" 3: direct bounded fit of the coil currents """
results["bounded"] = []
print("\n current limit | coils, bounded least squares: mean B.n  max B.n  coils at the limit  rms I")
for limit in CURRENT_LIMITS:
    solution = lsq_linear(coil_matrix, normal_target, bounds=(-limit, limit), max_iter=5000)
    row = dict(limit=float(limit), coils=error(coil_matrix @ solution.x),
               at_limit=int(np.sum(np.abs(solution.x) > 0.999 * limit)), current_rms=float(np.sqrt(np.mean(solution.x**2))),
               current_max=float(np.abs(solution.x).max()), currents=solution.x.tolist())
    results["bounded"].append(row)
    print(f"   {limit:8.2f}    |   {row['coils'][0]:8.3%} {row['coils'][1]:8.3%}   {row['at_limit']:3d} of {len(shaping)}   "
          f"{row['current_rms']:6.2f}  (max {row['current_max']:.2f})")

with open(f"{OUTPUT}.json", "w") as stream:
    json.dump(results, stream, indent=1)
print(f"\nWrote {OUTPUT}.json")
