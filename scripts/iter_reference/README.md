# ITER reference equilibrium for `examples/manifold_optimization/optimize_iter_divertor.py`

1. `solve_tokamaker_iter.py` runs the Open FUSION Toolkit (v26.9 binary release) TokaMaker ITER
   baseline (mesh, profiles and constraints of its `ITER_baseline_ex` notebook), then re-solves
   with the ITER reference coil currents of that notebook at Ip = 15.6 MA, and exports coil
   currents, gEQDSK files and the plasma toroidal current integrated over mesh triangles.
2. `fit_exterior_plasma.py` replaces the ~15,600 current-carrying cells by 181 exterior-equivalent
   circular filaments whose flux matches the full distribution on and outside the LCFS
   (poloidal-field error at most 0.25 %), and writes
   `examples/input_files/iter_tokamaker_reference.json`.

Both scripts expect the OFT source checkout (for `ITER_geom.json` and `ITER_mesh.h5`) and binary
release under the workspace paths set at their top, and must run on a compute node. The ITER wall
and coil geometry come from OFT (LGPL-3.0) and are read from that checkout, not copied here.
