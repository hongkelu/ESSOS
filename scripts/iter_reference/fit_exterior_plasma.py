"""Compact exterior-equivalent plasma for the TokaMaker ITER reference equilibrium.

~200 filaments on contours inside the plasma, with currents fitted so that their poloidal
flux matches the full TokaMaker current distribution on and outside the LCFS. Outside the
plasma the field is harmonic, so matching psi there reproduces B where field lines are traced.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from essos.fields import CircularLoopsField  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
GEOM = json.load(open("/pscratch/sd/h/hongkelu/manifold-optimization/.dependencies/external/OpenFUSIONToolkit/"
                      "src/examples/TokaMaker/ITER/ITER_geom.json"))
WALL = np.array(GEOM["limiter"])
ref = json.load(open(os.path.join(OUT, "iter_reference.json")))
coil_names = [n for n in GEOM["coils"] if not n.startswith("VS")]
coils = np.array([[GEOM["coils"][n]["rc"], GEOM["coils"][n]["zc"]] for n in coil_names])
coil_At = np.array([ref["coil_currents_At"][n] for n in coil_names])
f = ref["filaments"]
Rf, Zf, If = map(np.asarray, (f["R"], f["Z"], f["I"]))
keep = np.abs(If) > 1e-6 * np.abs(If).max()
Rf, Zf, If = Rf[keep], Zf[keep], If[keep]
plasma = CircularLoopsField(Rf, Zf, If)
total = CircularLoopsField(np.r_[Rf, coils[:, 0]], np.r_[Zf, coils[:, 1]], np.r_[If, coil_At])

# Reference X-point and LCFS (psi = psi_X) of the full model.
psi_total = lambda p: total.psi(p[0], p[1])  # noqa: E731
x = jnp.array([5.125, -3.4])
for _ in range(20):
    x = x - jnp.linalg.solve(jax.hessian(psi_total)(x), jax.grad(psi_total)(x))
psi_x = float(psi_total(x))
center = np.array([6.3, 0.5])


def boundary_point(angle):
    """LCFS point on the ray from ``center`` at ``angle`` (bisection on psi - psi_X)."""
    direction = np.array([np.cos(angle), np.sin(angle)])
    lo, hi = 0.5, 4.5
    inside = np.sign(float(psi_total(jnp.asarray(center + 0.5 * direction))) - psi_x)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sign(float(psi_total(jnp.asarray(center + mid * direction))) - psi_x) == inside:
            lo = mid
        else:
            hi = mid
    return center + 0.5 * (lo + hi) * direction


angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
lcfs = np.array([boundary_point(a) for a in angles])
# Ignore rays that leave through the X-point region (the separatrix legs are not closed there).
radius = np.linalg.norm(lcfs - center, axis=1)
lcfs = lcfs[radius < 4.4]

# Fit points: the LCFS, outward offsets, and a divertor-region grid below the X-point.
normals = lcfs - center
normals /= np.linalg.norm(normals, axis=1)[:, None]
fit_points = np.vstack([lcfs + d * normals for d in (0.0, 0.05, 0.15, 0.4, 1.0)])
Rd, Zd = np.meshgrid(np.linspace(4.0, 6.4, 25), np.linspace(-4.6, -3.2, 15))
fit_points = np.vstack([fit_points, np.c_[Rd.ravel(), Zd.ravel()]])

# Equivalent filaments on three contours at 0.35, 0.55, 0.75 of the way from the centre.
equivalent = np.vstack([center + s * (lcfs[::3] - center) for s in (0.35, 0.55, 0.75)] + [center[None]])

def psi_unit(point):
    """psi at ``point`` of each equivalent filament carrying 1 A."""
    field = CircularLoopsField(equivalent[:, 0], equivalent[:, 1], jnp.ones(len(equivalent)))
    _, plus, m, K, E = field._geometry(point[0], point[1])
    return point[0] * 4e-7 * jnp.sqrt(field.radii / point[0]) / jnp.sqrt(m) * ((1.0 - 0.5 * m) * K - E)


A = np.asarray(jax.jit(jax.vmap(psi_unit))(jnp.asarray(fit_points)))
b = np.asarray(jax.jit(jax.vmap(lambda p: plasma.psi(p[0], p[1])))(jnp.asarray(fit_points)))
lam = 1e-10 * np.linalg.norm(A, 2) ** 2
I_eq = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
print(f"{len(equivalent)} equivalent filaments; total current {I_eq.sum() / 1e6:.3f} MA (plasma {If.sum() / 1e6:.3f})",
      flush=True)
print("psi fit residual max [Wb]", np.max(np.abs(A @ I_eq - b)), "of range", np.ptp(b), flush=True)

eq = CircularLoopsField(equivalent[:, 0], equivalent[:, 1], I_eq)
probe = np.vstack([lcfs[::10] + 0.02 * normals[::10], lcfs[::10] + 0.2 * normals[::10],
                   [[5.565, -4.3], [4.25, -3.7], [5.0, -3.8], [5.3, -3.3], [8.3, 0.41]]])
B_full = np.array([plasma.B_cylindrical(R, Z) for R, Z in probe])
B_eq = np.array([eq.B_cylindrical(R, Z) for R, Z in probe])
B_coil_tf = np.array([CircularLoopsField(coils[:, 0], coils[:, 1], coil_At).B_cylindrical(R, Z) for R, Z in probe])
err_plasma = np.linalg.norm(B_eq - B_full, axis=1) / np.linalg.norm(B_full, axis=1)
err_poloidal = np.linalg.norm(B_eq - B_full, axis=1) / np.linalg.norm(B_full + B_coil_tf, axis=1)
print(f"plasma-field error: max {err_plasma.max():.2e}; relative to total poloidal field max {err_poloidal.max():.2e}",
      flush=True)
json.dump(dict(coil_names=coil_names, coil_At=coil_At.tolist(), coil_RZ=coils.tolist(),
               plasma_R=equivalent[:, 0].tolist(), plasma_Z=equivalent[:, 1].tolist(), plasma_I=I_eq.tolist(),
               xpoint=np.asarray(x).tolist(), psi_x=psi_x, lcfs=lcfs.tolist(), F0=5.3 * 6.2, Ip=float(If.sum()),
               source="OFT TokaMaker v26.9 ITER baseline with ITER reference coil currents (Ip 15.6 MA)",
               max_poloidal_field_error=float(err_poloidal.max())),
          open(os.path.join(OUT, "iter_exterior_plasma.json"), "w"), indent=1)
