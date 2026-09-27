"""Strict, non-executable readers for supported external coil input subsets."""
import json
import numpy as np
from essos.coils import Coils,Curves


def load_simsopt_xyz_coils(path,*,n_segments=128):
    """Read a SIMSON graph of XYZ Fourier, rotated curves, and scaled currents.

    Only explicit supported data records are interpreted. No classes named in
    the JSON are imported or instantiated. All coils in graph[2] are retained;
    the result has no implicit symmetry expansion, so it also covers explicit
    circular/PF coils when represented by the supported Fourier curves.
    """
    with open(path) as stream:data=json.load(stream)
    if data.get('@class')!='SIMSON':raise ValueError('Expected SIMSON coil graph')
    objects=data['simsopt_objs']
    def resolve(ref):
        if ref.get('$type')!='ref':raise ValueError('Expected explicit object reference')
        return objects[ref['value']]
    def curve(ref,seen=()):
        key=ref['value']
        if key in seen:raise ValueError('Cyclic curve reference')
        obj=resolve(ref);kind=obj['@class']
        if kind=='CurveXYZFourier':
            order=int(obj['order']);values=np.asarray(resolve(obj['dofs'])['x']['data'],dtype=float)
            if values.shape!=(3*(2*order+1),) or not np.all(np.isfinite(values)):raise ValueError('Invalid Fourier curve DOFs')
            return values.reshape(3,2*order+1)
        if kind=='RotatedCurve':
            coefficients=curve(obj['curve'],seen+(key,));phi=float(obj['phi'])
            rotation=np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1.]])
            coefficients=rotation@coefficients
            if obj['flip']:coefficients=np.diag([1.,-1.,-1.])@coefficients
            return coefficients
        raise ValueError('Unsupported coil curve: '+kind)
    def current(ref,seen=()):
        key=ref['value']
        if key in seen:raise ValueError('Cyclic current reference')
        obj=resolve(ref);kind=obj['@class']
        if kind=='Current':return float(obj['current'])
        if kind=='ScaledCurrent':return float(obj['scale'])*current(obj['current_to_scale'],seen+(key,))
        raise ValueError('Unsupported coil current: '+kind)
    curves=[];currents=[]
    for ref in data['graph'][2]:
        obj=resolve(ref)
        if obj['@class']!='Coil':raise ValueError('Expected coil reference')
        curves.append(curve(obj['curve']));currents.append(current(obj['current']))
    if not curves or not np.all(np.isfinite(currents)):raise ValueError('No finite coils')
    n=max(c.shape[1] for c in curves)
    padded=np.zeros((len(curves),3,n))
    for i,c in enumerate(curves):padded[i,:,:c.shape[1]]=c
    return Coils(Curves(padded,n_segments=n_segments,nfp=1,stellsym=False),np.asarray(currents))
