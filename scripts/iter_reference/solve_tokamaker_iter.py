"""Solve the OFT TokaMaker ITER baseline and export coil currents, gEQDSK and plasma current filaments."""
import json
import os
import sys

import numpy as np

W = "/pscratch/sd/h/hongkelu/manifold-optimization"
OFT = f"{W}/.tools/OpenFUSIONToolkit_v26.9/OpenFUSIONToolkit_v26.9-Linux-GNU-x86_64"
EXAMPLE = f"{W}/.dependencies/external/OpenFUSIONToolkit/src/examples/TokaMaker/ITER"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(OFT, "python"))
from OpenFUSIONToolkit import OFT_env  # noqa: E402
from OpenFUSIONToolkit.TokaMaker import TokaMaker  # noqa: E402
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh  # noqa: E402
from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun  # noqa: E402

myOFT = OFT_env(nthreads=8)
mygs = TokaMaker(myOFT)
mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(os.path.join(EXAMPLE, "ITER_mesh.h5"))
mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
mygs.setup(order=2, F0=5.3 * 6.2)
mygs.set_coil_vsc({"VS": 1.0})
mygs.set_coil_bounds({key: [-50.E6, 50.E6] for key in mygs.coil_sets})
Ip_target, P0_target = 15.6E6, 6.2E5
mygs.set_targets(Ip=Ip_target, pax=P0_target)
isoflux_pts = np.array([[8.20, 0.41], [8.06, 1.46], [7.51, 2.62], [6.14, 3.78], [4.51, 3.02], [4.26, 1.33],
                        [4.28, 0.08], [4.49, -1.34], [7.28, -1.89], [8.00, -0.68]])
x_point = np.array([[5.125, -3.4]])
mygs.set_isoflux_constraints(np.vstack((isoflux_pts, x_point)))
mygs.set_saddle_constraints(x_point)
reg = []
for name in mygs.coil_sets:
    weight = 2.E-2 if name.startswith("CS1") else 1.E-2
    reg.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=weight))
reg.append(mygs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=1.E2))
mygs.set_coil_reg(reg_terms=reg)
mygs.set_profiles(ffp_prof=create_power_flux_fun(40, 1.5, 2.0), pp_prof=create_power_flux_fun(40, 4.0, 1.0))
mygs.init_psi(6.3, 0.5, 2.0, 1.4, 0.0)


def export(tag):
    currents, region_currents = mygs.get_coil_currents()
    mygs.print_info()
    r, lc = np.asarray(mygs.r), np.asarray(mygs.lc)
    jtor = np.asarray(mygs.calc_jtor_plasma())
    print(tag, "jtor shape", jtor.shape, "nodes", r.shape, "cells", lc.shape, flush=True)
    tri = r[lc][:, :, :2]
    area = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    j_cell = jtor[lc].mean(axis=1) if jtor.shape[0] == r.shape[0] else None
    filaments = None
    if j_cell is not None:
        current = j_cell * area
        keep = np.abs(current) > 0
        centroid = tri.mean(axis=1)
        filaments = dict(R=centroid[keep, 0].tolist(), Z=centroid[keep, 1].tolist(), I=current[keep].tolist())
        print(tag, f"{keep.sum()} filaments carrying {current[keep].sum() / 1e6:.3f} MA", flush=True)
    mygs.save_eqdsk(os.path.join(OUT, f"iter_{tag}.geqdsk"), nr=257, nz=513, lcfs_pad=0.001)
    json.dump(dict(coil_currents_At={k: float(v) for k, v in currents.items()}, filaments=filaments),
              open(os.path.join(OUT, f"iter_{tag}.json"), "w"))


err = mygs.solve()
print("isoflux solve returned", err, flush=True)
export("isoflux")

# ITER reference coil currents (kA x turns) from the OFT example, with Ip/R0/Z0 targets.
mygs.set_isoflux_constraints(None)
mygs.set_saddle_constraints(None)
mygs.set_coil_currents({"CS3U": 5180.432355 * 553, "CS2U": -16660.61401 * 553, "CS1U": -36367.32465 * 553,
                        "CS1L": -36367.32465 * 553, "CS2L": -16472.05814 * 553, "CS3L": 8671.71887 * 553,
                        "PF1": 21725.94264 * 248.6, "PF2": -22213.70001 * 115.2, "PF3": -31877.98793 * 185.9,
                        "PF4": -26721.1652 * 169.9, "PF5": -38450.07502 * 216.8, "PF6": 39603.47595 * 459.4,
                        "VS": 0.0})
mygs.set_targets(Ip=Ip_target, R0=6.37, Z0=0.51)
err = mygs.solve()
print("reference-current solve returned", err, flush=True)
export("reference")
