"""Qualification-only VMEX radial ladder and response binding.

Pinned to the recorded VMEX source snapshot. Private stage/linearization hooks
retain the mask and constraint anchors from the *same* edge-accepted solve;
no process-wide warm-start cache is read or written. This is not a public API.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from vmex.core import freeboundary as fb
from vmex.core import freeboundary_implicit as fi
from vmex.core.fourier import mode_table
from vmex.core.multigrid import interpolate_state


class RadialContinuation:
    def __init__(self, inp, field_from_current, ladder, tolerance, iterations, output):
        self.inp = inp
        self.field_from_current = field_from_current
        self.ladder = tuple(ladder)
        if (not self.ladder or any(n < 3 for n in self.ladder)
                or any(b <= a for a, b in zip(self.ladder, self.ladder[1:]))):
            raise ValueError('Radial ladder must be strictly increasing, with NS >= 3')
        self.tolerance = tolerance
        self.iterations = iterations
        self.output = Path(output)
        self.records = []
        self._accepted = {}

    def solve(self, current, *, fresh=False):
        values = np.asarray(current, dtype=np.float64)
        if values.shape != (1,) or not np.all(np.isfinite(values)):
            raise ValueError('Expected one finite common current factor')
        key = values.tobytes()
        if not fresh and key in self._accepted:
            return self._accepted[key]
        field = self.field_from_current(jnp.asarray(values))
        previous = None
        modes = mode_table(int(self.inp.mpol), int(self.inp.ntor))
        run_dir = self.output / f'ladder-{len(self.records):02d}'
        run_dir.mkdir(exist_ok=False)
        record = dict(current_factor=float(values[0]), stages=[])
        self.records.append(record)
        for ns in self.ladder:
            inp = replace(self.inp, ns_array=[ns])
            seed = None if previous is None else interpolate_state(
                previous.result.state, ns_fine=ns, modes=modes)
            print(f'Radial continuation current={values[0]:.7f}, NS={ns}', flush=True)
            stage = fb._solve_free_boundary_stage(
                inp, external_field=field,
                resolution=fb.free_boundary_resolution(inp, field, ns=ns),
                ftol=self.tolerance, max_iterations=self.iterations,
                initial_state=seed,
                vacuum_continuation=None if previous is None else previous.vacuum,
                residual_continuation=None if previous is None else (
                    previous.result.fsqr, previous.result.fsqz, previous.result.fsql),
                constraint_continuation=None, reuse_vacuum_cache=False,
                include_edge_in_convergence=True, edge_force_tolerance=self.tolerance,
                error_on_no_convergence=False, use_fft=False)
            result = stage.result
            residuals = np.array([result.fsqr, result.fsqz, result.fsql, result.fedge], float)
            accepted = bool(result.converged and np.all(np.isfinite(residuals))
                            and np.all(residuals >= 0) and np.max(residuals) <= self.tolerance)
            row = dict(ns=ns, iterations=int(result.iterations), accepted=accepted,
                       residuals=residuals.tolist() if np.all(np.isfinite(residuals)) else None)
            record['stages'].append(row)
            if accepted:
                checkpoint = run_dir / f'state-ns{ns}.npz'
                np.savez(checkpoint, **{name:np.asarray(getattr(result.state, name))
                    for name in ('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin')},
                    rcon0=np.asarray(stage.rcon0), zcon0=np.asarray(stage.zcon0))
                row['checkpoint'] = str(checkpoint.relative_to(self.output))
                row['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (self.output/'continuation.json').write_text(json.dumps(self.records, indent=2, allow_nan=False)+'\n')
            print(f'NS={ns}: accepted={accepted}, residuals={residuals}', flush=True)
            if not accepted:
                raise ValueError(f'Radial continuation rejected NS={ns}, current={values[0]}')
            previous = stage
        self._accepted[key] = stage
        return stage

    def implicit_state(self, params, cfg):
        """Bind native VMEX implicit VJP to deterministic accepted ladder solves.

        Repeated identical controls reuse one immutable accepted root. Different
        controls always run the complete ladder, independently of call history.
        The VJP retains VMEX's frozen inactive-mode convention; independent
        ordinary ladder re-solves remain the qualification gate.
        """
        def host(c):
            stage = self.solve(c)
            return fi._linearization_from_stage(cfg, params, stage, inp=self.inp)
        reference = host(np.ones(1))
        shapes = jax.tree.map(lambda a:jax.ShapeDtypeStruct(a.shape, jnp.float64), reference)
        def callback(c):
            return jax.pure_callback(host, shapes, c)
        @jax.custom_vjp
        def state(c):
            return callback(c)[0]
        def forward(c):
            z, mask, rcon, zcon = callback(c)
            return z, (params, c, z, mask, rcon, zcon)
        def backward(saved, cotangent):
            _, current_bar = fi._solve_bwd(cfg, saved, cotangent)
            return (current_bar,)
        state.defvjp(forward, backward)
        return state
