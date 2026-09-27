"""Exterior classification for a simple Fourier LCFS in cylindrical sections."""
import numpy as np


class FourierLCFSExteriorDomain:
    """Reject interior points and unresolved boundary proximity.

    Modes use the VMEC wout convention ``m*theta - xn*phi`` (xn includes nfp).
    Every constant-phi boundary must be a simple closed RZ curve. The polygon's
    chord error is bounded by max|d²(R,Z)/dtheta²| * dtheta² / 8. Clearance here
    is in a cylindrical section, NOT shortest three-dimensional wall distance.
    """
    def __init__(self, xm, xn, rmnc, zmns, *, rmns=None, zmnc=None,
                 ntheta=1024, minimum_section_clearance=0.):
        arrays = [np.asarray(a, float) for a in (xm,xn,rmnc,zmns)]
        if (arrays[0].ndim != 1 or not len(arrays[0])
                or any(a.shape != arrays[0].shape or not np.all(np.isfinite(a)) for a in arrays)):
            raise ValueError('Expected matching finite Fourier mode vectors')
        if np.any(arrays[0] != np.round(arrays[0])) or np.any(arrays[1] != np.round(arrays[1])):
            raise ValueError('Fourier modes must be integers for a closed periodic LCFS')
        if isinstance(ntheta, bool) or int(ntheta) != ntheta or ntheta < 16:
            raise ValueError('ntheta must be an integer >= 16')
        if not np.isfinite(minimum_section_clearance) or minimum_section_clearance < 0:
            raise ValueError('Invalid minimum section clearance')
        for value in (rmns,zmnc):
            a = np.zeros_like(arrays[0]) if value is None else np.asarray(value,float)
            if a.shape != arrays[0].shape or not np.all(np.isfinite(a)):
                raise ValueError('Invalid asymmetric Fourier coefficients')
            arrays.append(a)
        self.xm,self.xn,self.rmnc,self.zmns,self.rmns,self.zmnc = (
            np.frombuffer(a.tobytes(),dtype=float) for a in arrays)
        self.ntheta = int(ntheta)
        self.minimum_section_clearance = float(minimum_section_clearance)
        r2 = np.sum(self.xm**2*(abs(self.rmnc)+abs(self.rmns)))
        z2 = np.sum(self.xm**2*(abs(self.zmnc)+abs(self.zmns)))
        self.chord_error_bound = float(np.hypot(r2,z2)*(2*np.pi/self.ntheta)**2/8)

    @classmethod
    def from_wout(cls, wout, **kwargs):
        asym = bool(wout.lasym)
        return cls(wout.xm,wout.xn,np.asarray(wout.rmnc)[-1],np.asarray(wout.zmns)[-1],
            rmns=np.asarray(wout.rmns)[-1] if asym else None,
            zmnc=np.asarray(wout.zmnc)[-1] if asym else None,**kwargs)

    def section(self, phi):
        theta = np.arange(self.ntheta)*2*np.pi/self.ntheta
        phase = theta[:,None]*self.xm-self.xn*phi
        cs,sn = np.cos(phase),np.sin(phase)
        return np.column_stack((cs@self.rmnc+sn@self.rmns,cs@self.zmnc+sn@self.zmns))

    def classify(self, xyz):
        points = np.asarray(xyz,float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError('Expected finite Cartesian points (n,3)')
        outside = np.zeros(len(points),bool)
        clearance = np.empty(len(points))
        for i,(x,y,z) in enumerate(points):
            r = np.hypot(x,y)
            polygon = self.section(np.arctan2(y,x))
            if np.min(polygon[:,0]) <= self.chord_error_bound:
                raise ValueError('Boundary section approaches the cylindrical coordinate axis')
            a,b = polygon,np.roll(polygon,-1,axis=0)
            edge = b-a
            lengths2 = np.sum(edge*edge,axis=1)
            if np.any(lengths2 == 0):raise ValueError('Degenerate boundary section')
            t = np.clip(np.sum((np.array([r,z])-a)*edge,axis=1)/lengths2,0,1)
            distance = np.min(np.linalg.norm(np.array([r,z])-a-t[:,None]*edge,axis=1))
            # Ray crossing uses only straddling edges, avoiding horizontal-edge division.
            crossing = (a[:,1]>z) != (b[:,1]>z)
            aa,bb = a[crossing],b[crossing]
            intersections = aa[:,0]+(z-aa[:,1])*(bb[:,0]-aa[:,0])/(bb[:,1]-aa[:,1])
            interior = np.count_nonzero(intersections>r)%2 == 1
            clearance[i] = (-distance if interior else distance)-self.chord_error_bound
            outside[i] = r>0 and clearance[i]>self.minimum_section_clearance
        return outside,clearance

    def __call__(self, xyz):
        return self.classify(xyz)[0]
