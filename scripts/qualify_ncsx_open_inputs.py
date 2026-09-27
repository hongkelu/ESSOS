#!/usr/bin/env python3
"""Explore native first hits for the accessible NCSX coil and vessel inputs.

This establishes numerical access/geometry, not an engineering specification
or correspondence with the finite-beta LI383 equilibrium.
"""
import argparse
import json
from pathlib import Path


def load_ncsx_vessel(source):
    """Read and symmetry-expand the supplied single-field-period wall mesh."""
    import numpy as np
    from pyna.topo.clearance3d import TriangleWall
    from pyna.toroidal.geometry import ToroidalWall
    lines=(source/'ncsx_wall.dat').read_text().splitlines();nv,nf=map(int,lines[2].split())
    vertices=np.array([list(map(float,line.split())) for line in lines[3:3+nv]])
    faces=np.array([list(map(int,line.split())) for line in lines[3+nv:3+nv+nf]])-1
    # The file covers one 120-degree field period with both end sections.
    if (nv,nf)!=(6100,12000):raise ValueError('Unexpected NCSX vessel layout')
    rows=vertices.reshape(61,100,3);r=np.linalg.norm(rows[:,:,:2],axis=-1);z=rows[:,:,2]
    angles=np.unwrap(np.arctan2(rows[:,0,1],rows[:,0,0]));span=2*np.pi/3
    if not np.allclose(r[0],r[-1],atol=1e-5) or not np.allclose(z[0],z[-1],atol=1e-5):raise ValueError('Field-period seam is not closed')
    if not np.allclose(angles,np.linspace(0,span,61),atol=1e-5):raise ValueError('Unexpected vessel toroidal grid')
    expanded=[]
    for angle in (0.,span,2*span):
        rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
        expanded.append(vertices@rotation.T)
    vertices3=np.concatenate(expanded);faces3=np.concatenate([faces+k*nv for k in range(3)])
    from scipy.spatial import cKDTree
    parent=np.arange(len(vertices3))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in cKDTree(vertices3).query_pairs(3e-6):parent[find(b)]=find(a)
    roots=np.array([find(i) for i in range(len(parent))]);unique_ids,inverse=np.unique(roots,return_inverse=True)
    unique=vertices3[unique_ids]
    mesh=TriangleWall(unique,inverse[faces3],('vessel',)*len(faces3))
    wall=ToroidalWall(np.linspace(0,span,60,endpoint=False),r[:-1],z[:-1],nfp=3)
    return wall,mesh,dict(merged_vertices=len(unique),faces=len(faces3),original_vertices=nv,
                          seam_merge_tolerance_m=3e-6,field_period_rad=span)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from run_manifold_baseline import require_compute_allocation
    require_compute_allocation(parser)
    args.output.mkdir(parents=True,exist_ok=False)
    from manifold_run_manifest import capture_source_manifest
    capture_source_manifest(args.output)
    import numpy as np
    from essos.coil_inputs import load_simsopt_xyz_coils
    from essos.fields import BiotSavart
    from essos.manifold import essos_field_to_pyna_cylindrical_grid
    from pyna.topo.clearance3d import TriangleWall
    from pyna.toroidal.geometry import ToroidalWall
    from pyna.toroidal.control.strike_heat import StrikeSeedBundle,trace_wall_strikes_field
    source=Path(__file__).resolve().parents[1]/'essos/data/manifold_optimization'
    wall,mesh,mesh_metadata=load_ncsx_vessel(source)
    coils=load_simsopt_xyz_coils(source/'ncsx_coils.json',n_segments=128);field=BiotSavart(coils)
    seeds=np.array([[1.95,0.],[2.1,0.],[2.2,0.],[2.3,0.],[1.3,.4],[1.6,.5],[1.7,.65]])
    inside=mesh.distance(np.column_stack((seeds[:,0],np.zeros(len(seeds)),seeds[:,1])),signed=True)>0
    seeds=seeds[inside]
    print(f'{len(seeds)} launch points inside the closed mesh',flush=True)
    grid=essos_field_to_pyna_cylindrical_grid(field,np.linspace(.65,2.65,81),np.linspace(-1.1,1.1,81),
           np.linspace(0,2*np.pi/3,96,endpoint=False),nfp=3,batch_size=256)
    bundle=StrikeSeedBundle('ncsx-open-inputs','general_open',seeds[:,0],seeds[:,1],np.zeros(len(seeds)),'+',np.ones(len(seeds)),'relative',np.arange(len(seeds)))
    strikes=trace_wall_strikes_field(grid,(bundle,),wall,max_turns=20,DPhi=.005)[0]
    report=dict(case='accessible NCSX coil/vessel input exploration',closed_triangle_mesh=True,
        mesh_metadata=mesh_metadata,
        launches_RZ_m=seeds.tolist(),hit_seed_indices=np.asarray(strikes.seed_index).tolist(),
        hits_RZPhi=np.column_stack((strikes.R,strikes.Z,strikes.phi)).tolist(),
        connection_lengths_m=np.asarray(strikes.connection_length).tolist(),
        limitation='No engineering limits, LI383 association, or whole-leg clearance qualification is implied')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
