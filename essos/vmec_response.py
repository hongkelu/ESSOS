"""Optional VMEC-JAX response at fixed flux-grid labels.

This adapter is deliberately named for its material coordinate convention.
B is Cartesian, but its sampling positions move with the equilibrium. It must
not be passed to a fixed-Cartesian-position field-line tracer. The separate
ImplicitEquilibriumField contract requires Eulerian field evaluation, including
coordinate inversion/response. Boundary Fourier controls here do not represent
free-boundary coil controls or a self-consistent open-field-region model.
"""
from dataclasses import dataclass,replace
from pathlib import Path
import hashlib
import numpy as np


@dataclass(frozen=True)
class VmecMaterialEvaluation:
    field: np.ndarray
    jacobian: np.ndarray | None
    parameters: np.ndarray
    converged: bool
    force_residuals: tuple[float,float,float]
    coordinate_convention: str = 'fixed_vmec_flux_grid_labels'

    def __post_init__(self):
        from pyna.topo.snapshot import immutable_array
        for name in ('field','jacobian','parameters'):
            value=getattr(self,name)
            if value is not None:object.__setattr__(self,name,immutable_array(value))


class VmecJaxMaterialField:
    """Use actual VMEC solves and its discrete state-response implementation.

    The pinned solver's discrete response is treated as provisional until it
    agrees with independent complete equilibrium finite differences. Primal
    force residuals are checked separately from response qualification.
    """
    def __init__(self,input_path,workdir,*,ns=16,controls=(('rc',1,0),),
                 max_iterations=2500,force_tolerance=1e-13,s_index=None,response_method="discrete"):
        if response_method not in ("discrete","implicit_residual"):raise ValueError("Unknown VMEC response method")
        self.response_method=response_method
        import jax
        import jax.numpy as jnp
        import vmec_jax as vj
        from vmec_jax.optimization import BoundaryParamSpec,FixedBoundaryExactOptimizer
        from vmec_jax.field import b_cartesian_from_state
        if not jax.config.x64_enabled:raise ValueError('VMEC response requires float64')
        self.input_path=Path(input_path);self.workdir=Path(workdir);self.workdir.mkdir(parents=True,exist_ok=True)
        self.ns=ns;self.max_iterations=max_iterations;self.force_tolerance=force_tolerance
        cfg,indata=vj.load_config(self.input_path);cfg=replace(cfg,ns=ns)
        self.static=vj.build_static(cfg);self.indata=indata
        self.s_index=ns//2 if s_index is None else s_index
        boundary=vj.boundary_from_indata(indata,self.static.modes,apply_m1_constraint=False)
        boundary_input=vj.boundary_input_from_indata(indata,self.static.modes)
        self.boundary_input=boundary_input
        specs=[]
        for kind,m,n in controls:
            indices=np.flatnonzero((np.asarray(self.static.modes.m)==m)&(np.asarray(self.static.modes.n)==n))
            if len(indices)!=1:raise ValueError('Requested boundary Fourier mode is unavailable')
            specs.append(BoundaryParamSpec(f'{kind}({m},{n})',kind,int(indices[0]),m,n))
        # The field helper infers no changing profiles: all input profiles are
        # held fixed; only the selected physical boundary modes vary.
        initial=vj.run_fixed_boundary(self.input_path,ns_override=ns,max_iter=max_iterations,multigrid=False,verbose=False)
        self.signgs=initial.signgs
        self._field=lambda state:b_cartesian_from_state(state,self.static,indata=self.indata,signgs=self.signgs,s_index=self.s_index)
        self.optimizer=FixedBoundaryExactOptimizer(self.static,indata,boundary,specs,
                         lambda state:jnp.ravel(self._field(state)),boundary_input=boundary_input,
                         inner_max_iter=max_iterations,inner_ftol=force_tolerance,
                         trial_max_iter=max_iterations,trial_ftol=force_tolerance,exact_path='tape')
        self.specs=tuple(specs)
        self.parameter_names=tuple(spec.name for spec in specs)

    def solve_primal(self,parameters):
        import vmec_jax as vj
        parameters=np.asarray(parameters,dtype=float)
        if parameters.shape!=(len(self.parameter_names),) or not np.all(np.isfinite(parameters)):raise ValueError('Invalid VMEC boundary parameters')
        tag=hashlib.sha256(parameters.tobytes()).hexdigest()[:20]
        path=self.workdir/f'input-{tag}.vmec'
        self.optimizer.save_input(path,parameters)
        from vmec_jax.namelist import InData,write_indata
        _,candidate_indata=vj.load_config(path)
        scalars=dict(candidate_indata.scalars)
        scalars['FTOL_ARRAY']=[self.force_tolerance]
        scalars['NS_ARRAY']=[self.ns]
        write_indata(path,InData(scalars=scalars,indexed=dict(candidate_indata.indexed)))
        run=vj.run_fixed_boundary(path,ns_override=self.ns,max_iter=self.max_iterations,multigrid=False,verbose=False)
        self._last_primal_run=run
        residuals=tuple(float(np.asarray(getattr(run.result,k))[-1]) for k in ('fsqr2_history','fsqz2_history','fsql2_history'))
        valid=bool(run.result.diagnostics.get('converged',False) and np.all(np.isfinite(residuals)) and max(residuals)<=self.force_tolerance)
        return VmecMaterialEvaluation(np.asarray(self._field(run.state)),None,parameters,valid,residuals)

    def response(self,parameters):
        primal=self.solve_primal(parameters)
        if not primal.converged:raise ValueError('VMEC primal equilibrium did not converge')
        if self.response_method=="discrete":
            field,columns=self.optimizer.b_cartesian_tangent_columns_fun(parameters,s_index=self.s_index)
        else:
            field,columns=self._implicit_response(parameters)
        if not np.all(np.isfinite(field)) or not np.all(np.isfinite(columns)):raise ValueError('VMEC response is nonfinite')
        discrepancy=np.linalg.norm(field-primal.field)/max(np.linalg.norm(primal.field),1e-30)
        if discrepancy>1e-5:raise ValueError(f'VMEC response base state differs from independently solved equilibrium: {discrepancy}')
        return VmecMaterialEvaluation(np.asarray(field),np.asarray(columns),np.asarray(parameters),True,primal.force_residuals)

    def _implicit_response(self,parameters):
        import jax
        import jax.numpy as jnp
        import vmec_jax as vj
        from vmec_jax.boundary import boundary_from_input_convention
        from vmec_jax.implicit import ImplicitFixedBoundaryOptions,solve_fixed_boundary_state_implicit_vmec_residual
        state0=self._last_primal_run.state
        def edges(c):
            raw=vj.apply_boundary_params(self.boundary_input,self.specs,c)
            boundary=boundary_from_input_convention(raw,self.static.modes,lasym=bool(self.static.cfg.lasym),apply_m1_constraint=False)
            return boundary.R_cos,boundary.R_sin,boundary.Z_cos,boundary.Z_sin
        # The implicit backend constructs an initial guess from BoundaryCoeffs;
        # its edge arguments are Fourier boundary coefficients, not normalized
        # rows of the solver state (these differ for nonaxisymmetric modes).
        options=ImplicitFixedBoundaryOptions(damping=0.,cg_tol=1e-11,cg_max_iter=500,
                                            residual_tangent_mode='chunked',jac_chunk_size=16)
        def material(c):
            e=edges(c)
            state=solve_fixed_boundary_state_implicit_vmec_residual(state0,self.static,indata=self.indata,
                       signgs=self.signgs,state0_host=state0,max_iter=self.max_iterations,
                       ftol=self.force_tolerance,step_size=self.indata.get_float('DELT',1.),implicit=options,
                       edge_Rcos=e[0],edge_Rsin=e[1],edge_Zcos=e[2],edge_Zsin=e[3])
            return self._field(state)
        c=jnp.asarray(parameters);columns=[];field=None
        for direction in np.eye(len(parameters)):
            field,column=jax.jvp(material,(c,),(jnp.asarray(direction),))
            columns.append(np.asarray(column))
        return np.asarray(field),np.stack(columns,axis=-1)

    @staticmethod
    def jvp(evaluation,direction):
        if not evaluation.converged or evaluation.jacobian is None:raise ValueError('Qualified primal and response required')
        return np.tensordot(evaluation.jacobian,np.asarray(direction),axes=1)

    @staticmethod
    def vjp(evaluation,cotangent):
        if not evaluation.converged or evaluation.jacobian is None:raise ValueError('Qualified primal and response required')
        return evaluation.jacobian.reshape(-1,evaluation.parameters.size).T@np.asarray(cotangent).ravel()
