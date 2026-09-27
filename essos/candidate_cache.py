"""Explicit content identities for physical evaluations and current-only grids."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import jax.numpy as jnp


def content_identity(*,currents,geometry,wall,numerics,topology,target):
    """Hash all semantic inputs; callers must provide actual content, not IDs."""
    digest=hashlib.sha256()
    def add(value):
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:raise TypeError('Object arrays cannot identify numerical caches')
            digest.update(b'array');digest.update(str((value.dtype.str,value.shape)).encode());digest.update(value.tobytes())
        elif isinstance(value,dict):
            digest.update(b'mapping');digest.update(str(len(value)).encode()+b':')
            if not all(isinstance(k,str) for k in value):raise TypeError('Cache mapping keys must be strings')
            for key in sorted(value):add(key);add(value[key])
        elif isinstance(value,(tuple,list)):
            digest.update(b'list');digest.update(str(len(value)).encode()+b':')
            for item in value:add(item)
        elif isinstance(value,np.generic):add(value.item())
        else:
            encoded=json.dumps(value,allow_nan=False,sort_keys=True).encode()
            digest.update(str(len(encoded)).encode()+b':'+encoded)
    for name,value in locals().copy().items():
        if name in ('currents','geometry','wall','numerics','topology','target'):add(name);add(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class CurrentGridBasis:
    """Fields per ampere on one fixed grid with fixed coil geometry.

    Coils may be symmetry-linked current groups. Grid coordinates, coil
    quadrature, units, and symmetry conventions belong in geometry_grid_key.
    Shape changes require rebuilding the basis; wall changes invalidate the
    candidate evaluation but do not alter magnetic field values on this grid.
    """
    fields_per_ampere: np.ndarray
    geometry_grid_key: str

    def __post_init__(self):
        a=np.asarray(self.fields_per_ampere,dtype=float)
        if not self.geometry_grid_key or a.ndim<3 or a.shape[-1]!=3 or not np.all(np.isfinite(a)):raise ValueError('Finite vector-field basis and identity required')
        object.__setattr__(self,'fields_per_ampere',np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape))

    def evaluate(self,currents,*,geometry_grid_key):
        if geometry_grid_key!=self.geometry_grid_key:raise ValueError('Coil geometry or sampling grid changed; rebuild current basis')
        currents=jnp.asarray(currents)
        if currents.shape!=(self.fields_per_ampere.shape[0],):raise ValueError('Current groups must match field basis')
        return jnp.tensordot(currents,jnp.asarray(self.fields_per_ampere),axes=1)
