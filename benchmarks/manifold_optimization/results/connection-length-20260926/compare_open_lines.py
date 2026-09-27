"""Accuracy/cost of pyna open_lines (fixed-step RK4 + trapezoid) vs ESSOS connection_length
(adaptive Dopri8 + event) against the closed-form circular-tokamak plate solution."""
import json, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from essos.dynamics import connection_length
from essos.fields import CircularTokamakField
from pyna.topo.open_lines import LaunchBundle, trace_open_bundle
from tests.test_connection_length import (MAJOR_RADIUS as R0, B_TOROIDAL as BT, B_POLOIDAL as BP,
                                          PLATE_DEPTH as H, circular_tokamak_plate_solution)

r = jnp.array([0.5, 0.7, 0.9]); theta = jnp.array([0.0, 1.2, 2.9])
R, Z = R0 + r * jnp.cos(theta), r * jnp.sin(theta)
seeds = jnp.stack([R, jnp.zeros(3), Z], axis=1)
field_fn = lambda x, p: CircularTokamakField(R0, BT, p[0]).B(x)
plate = lambda rz, phi, w: rz[1] + w[0]
def exact(p):
    L, pts = circular_tokamak_plate_solution(CircularTokamakField(R0, BT, p[0]), H, seeds)
    return L[:, 0], jnp.arctan2(pts[:, 0, 1], pts[:, 0, 0])
L_exact, _ = exact(jnp.array([BP]))
dL_exact = jax.jacfwd(lambda p: exact(p)[0])(jnp.array([BP]))[:, 0]
# forward-hit toroidal angle (unwrapped) from the closed form, perturbed as a cyna-like guess
from tests.test_connection_length import _toroidal_angle_advance
fld = CircularTokamakField(R0, BT, BP)
phi_hit = _toroidal_angle_advance(fld, r, jnp.pi + jnp.arcsin(H / r)) - _toroidal_angle_advance(fld, r, theta)
bundle = LaunchBundle(("a", "b", "c"), jnp.stack([R, Z], 1).tolist(), [1., 1., 1.])
rows = []
for n in (64, 256, 1024, 4096):
    def lengths(p):
        s = trace_open_bundle(field_fn, p, bundle, plate, jnp.array([H]), phi_start=0.,
                              phi_guesses=phi_hit + 0.01, maximum_phi_shift=0.2, n_steps=n)
        return s.connection_length
    f = jax.jit(lengths); g = jax.jit(jax.jacfwd(lengths))
    p = jnp.array([BP]); f(p).block_until_ready(); g(p).block_until_ready()
    t0 = time.perf_counter(); L = f(p).block_until_ready(); t1 = time.perf_counter(); dL = g(p)[:, 0].block_until_ready(); t2 = time.perf_counter()
    rows.append(dict(method="pyna open_lines", n_steps=n, max_rel_length_error=float(jnp.max(jnp.abs(L / L_exact - 1))),
                     max_rel_derivative_error=float(jnp.max(jnp.abs(dL / dL_exact - 1))), value_s=t1 - t0, jacfwd_s=t2 - t1))
for tol in (1e-6, 1e-8, 1e-10, 1e-12):
    def lengths(p, adjoint=None):
        return connection_length(CircularTokamakField(R0, BT, p[0]), seeds, lambda x: x[2] + H,
                                 max_length=200.0, tolerance=tol, adjoint=adjoint)["lengths"][:, 0]
    import diffrax
    f = jax.jit(lambda p: lengths(p)); g = jax.jit(jax.jacfwd(lambda p: lengths(p, diffrax.ForwardMode())))
    p = jnp.array([BP]); f(p).block_until_ready(); g(p).block_until_ready()
    t0 = time.perf_counter(); L = f(p).block_until_ready(); t1 = time.perf_counter(); dL = g(p)[:, 0].block_until_ready(); t2 = time.perf_counter()
    rows.append(dict(method="essos connection_length", tolerance=tol, max_rel_length_error=float(jnp.max(jnp.abs(L / L_exact - 1))),
                     max_rel_derivative_error=float(jnp.max(jnp.abs(dL / dL_exact - 1))), value_s=t1 - t0, jacfwd_s=t2 - t1))
for row in rows: print(json.dumps(row))
json.dump(rows, open(__file__.replace("compare_open_lines.py", "open_lines_vs_connection_length.json"), "w"), indent=1)
