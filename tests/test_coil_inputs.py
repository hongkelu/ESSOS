import json
import numpy as np
from essos.coils import load_simsopt_xyz_coils
from essos.coils import Curves,Coils


def test_imported_ncsx_graph_matches_native_symmetry_expansion():
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization/ncsx_coils.json'
    imported=load_simsopt_xyz_coils(source,n_segments=64)
    data=json.loads(source.read_text());objects=data['simsopt_objs'];base=[];currents=[]
    for curve_ref,current_ref in zip(data['graph'][0],data['graph'][1]):
        curve=objects[curve_ref['value']];dofs=objects[curve['dofs']['value']]['x']['data']
        base.append(np.asarray(dofs).reshape(3,-1));currents.append(objects[current_ref['value']]['current'])
    native=Coils(Curves(np.asarray(base),n_segments=64,nfp=3,stellsym=True),np.asarray(currents))
    np.testing.assert_allclose(imported.gamma,native.gamma,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(imported.currents,native.currents,rtol=0,atol=0)
