"""Forces and torques on a planar TF + shaping (dipole) coil set saved by optimize_planar_coils_with_potential.py.

usage: planar_coil_loads.py cp.npz wout.nc

Each coil feels the field of all the other coils (the plasma's own field at the coils is left out; it is small next to
the coil field). Printed in the units of the file and scaled to a reactor of major radius REACTOR_R [m] and
volume-averaged field REACTOR_B [T]: force per length ~ B^2 L, net force ~ B^2 L^2, torque ~ B^2 L^3.
"""
import os
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from netCDF4 import Dataset  # noqa: E402
from essos.planar_coils import coil_loads  # noqa: E402

data = np.load(sys.argv[1])
with Dataset(sys.argv[2]) as wout:
    R0, B0 = float(wout["Rmajor_p"][()]), float(wout["volavgB"][()])
length, field = float(os.environ.get("REACTOR_R", 8.0)) / R0, float(os.environ.get("REACTOR_B", 6.0)) / B0
groups = [tuple(jnp.asarray(data[f"{k}_{q}"]) for q in ("gamma", "gamma_dash", "currents")) for k in ("encircling", "shaping")]
print(f"R0 {R0:.3f} m, <B> {B0:.3f} T; scaled by {length:.2f} in length and {field:.2f} in field")
print(" coils     | count | current [MA] | max force per length [MN/m] | max net force [MN] | max torque [MN m]   (scaled)")
for name, group, load in zip(("encircling", "shaping"), groups, jax.jit(coil_loads)(groups)):
    per_length = float(jnp.max(jnp.linalg.norm(load["force_per_length"], axis=-1))) * field**2 * length
    net = float(jnp.max(jnp.linalg.norm(load["force"], axis=-1))) * field**2 * length**2
    torque = float(jnp.max(jnp.linalg.norm(load["torque"], axis=-1))) * field**2 * length**3
    current = float(jnp.max(jnp.abs(group[2]))) * field * length
    print(f" {name:10s}| {len(group[2]):5d} | {current / 1e6:10.2f}   | {per_length / 1e6:14.2f}              | {net / 1e6:10.2f}         | {torque / 1e6:8.2f}")
