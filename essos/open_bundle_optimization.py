"""General open-line objectives with independent production acceptance.

The local model traces each labelled launch to the wall with
:func:`essos.dynamics.connection_length` and differentiates it in forward mode.
The production callback must return pyna's ``OpenBundleValidation`` after
first-hit and whole-leg validation. Observables receive ``(hit_RZPhi,
connection_lengths, weights, parameters)`` in both paths, with ``phi`` unwrapped.
Hit correspondence uses Cartesian distance in metres and a separate unwrapped
phase check in radians; these quantities are never combined into one norm.
"""
from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from essos.candidate_cache import content_identity
from essos.dynamics import connection_length
from essos.topology_optimizer import Evaluation

LAUNCH_CONVENTIONS = ("fixed_physical", "moving_physical", "flux_labelled", "manifold_generated")


@dataclass(frozen=True)
class LaunchBundle:
    """Labelled launch points ``(R, Z)`` on the plane ``phi = phi_start`` with quadrature weights.

    Moving, flux-labelled and manifold-generated launches are supplied by the
    objective's ``launch_fn`` as active arrays; the stored points are then only the
    declared reference geometry.
    """

    labels: tuple
    launch_RZ: np.ndarray
    weights: np.ndarray
    launch_convention: str = "fixed_physical"
    weights_move: bool = False

    def __post_init__(self):
        labels = tuple(self.labels)
        points = np.array(self.launch_RZ, dtype=float)
        weights = np.array(self.weights, dtype=float)
        if not labels or len(set(labels)) != len(labels) or not all(isinstance(s, str) and s for s in labels):
            raise ValueError("Nonempty unique launch labels required")
        if points.shape != (len(labels), 2) or weights.shape != (len(labels),):
            raise ValueError("One (R, Z) launch point and one weight per label required")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Launch points must be finite and weights positive")
        if self.launch_convention not in LAUNCH_CONVENTIONS:
            raise ValueError("Unknown launch convention")
        points.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "launch_RZ", points)
        object.__setattr__(self, "weights", weights)


class _ParametrizedField:
    """Adapts ``field_fn(xyz, parameters)`` to the field interface of ``connection_length``."""

    def __init__(self, field_fn, parameters):
        self.field_fn = field_fn
        self.parameters = parameters

    def B_contravariant(self, xyz):
        return self.field_fn(xyz, self.parameters)


@dataclass(frozen=True)
class OpenBundleObjective:
    """Residuals of observables of labelled open field lines that end on a wall.

    Args:
        bundle: :class:`LaunchBundle` of launch points on ``phi = phi_start``.
        field_fn: ``field_fn(xyz, parameters)`` returning Cartesian ``B``.
        wall_fn: ``wall_fn(xyz, wall_parameters)``, positive inside the wall and zero on it.
        wall_parameters: ``wall_parameters(parameters)`` giving the (possibly moving) wall.
        observables: ``observables(hit_RZPhi, lengths, weights, parameters)`` vector.
        production_validate: ``production_validate(parameters, snapshot)`` returning
            pyna's ``OpenBundleValidation``.
        target, scales: Observable target and positive residual scales.
        phi_guesses: Expected unwrapped hit angle of every launch; its sign relative to
            ``phi_start`` selects the direction along ``B`` that is traced.
        identity: Version string of the callables above, part of the fingerprint.
        max_length: Cap on the traced length [m].
        tolerance: Integration and event tolerance of ``connection_length``.
    """

    bundle: object
    field_fn: object
    wall_fn: object
    wall_parameters: object
    observables: object
    production_validate: object
    target: tuple
    scales: tuple
    phi_guesses: tuple
    identity: str
    phi_start: float = 0.
    maximum_phi_shift: float = .2
    maximum_hit_discrepancy: float = 1e-3
    max_length: float = 100.
    tolerance: float = 1e-10
    launch_fn: object = None
    weights_fn: object = None

    def __post_init__(self):
        for name in ("target", "scales", "phi_guesses"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError("Finite objective vectors required")
            object.__setattr__(self, name, tuple(map(float, values)))
        if not self.identity or not self.target or len(self.scales) != len(self.target) or min(self.scales) <= 0:
            raise ValueError("Invalid fixed objective definition")
        if len(self.phi_guesses) != len(self.bundle.labels):
            raise ValueError("One hit-angle guess per launch required")
        if any(guess == self.phi_start for guess in self.phi_guesses):
            raise ValueError("Hit-angle guesses must differ from phi_start")
        if self.bundle.launch_convention != "fixed_physical" and self.launch_fn is None:
            raise ValueError("Moving launches require a physical launch function")
        if self.bundle.weights_move and self.weights_fn is None:
            raise ValueError("Moving weights require a physical weight function")
        controls = [self.phi_start, self.maximum_phi_shift, self.maximum_hit_discrepancy,
                    self.max_length, self.tolerance]
        if not np.all(np.isfinite(controls)) or min(controls[1:]) <= 0:
            raise ValueError("Invalid trace controls")

    @property
    def fingerprint(self):
        # identity versions callable field, wall, launch and observable definitions.
        return content_identity(currents=None, geometry=None, wall=None, numerics=None, topology=None, target=dict(
            coordinate_contract="cartesian_hits_unwrapped_phi_v2", wall_contract="xyz_positive_inside",
            identity=self.identity, labels=self.bundle.labels, launches=self.bundle.launch_RZ,
            weights=self.bundle.weights, convention=self.bundle.launch_convention,
            weights_move=self.bundle.weights_move, target=self.target, scales=self.scales,
            guesses=self.phi_guesses, phi_start=self.phi_start, shift=self.maximum_phi_shift,
            discrepancy=self.maximum_hit_discrepancy, max_length=self.max_length, tolerance=self.tolerance))

    def _check_snapshot(self, snapshot):
        if snapshot and snapshot.get("definition") != self.fingerprint:
            raise ValueError("Open-bundle snapshot definition changed")

    def _weights(self, c):
        weights = jnp.asarray(self.bundle.weights) if self.weights_fn is None else self.weights_fn(c)
        return weights if self.bundle.weights_move else jax.lax.stop_gradient(weights)

    def _launches(self, c):
        launches = jnp.asarray(self.bundle.launch_RZ if self.launch_fn is None else self.launch_fn(c))
        return jax.lax.stop_gradient(launches) if self.bundle.launch_convention == "fixed_physical" else launches

    def _direction(self, c, reference_phi):
        """Sign along B that advances every launch towards its expected hit angle."""
        launches = np.asarray(self._launches(jnp.asarray(c)))
        seeds = np.column_stack((launches[:, 0] * np.cos(self.phi_start),
                                 launches[:, 0] * np.sin(self.phi_start), launches[:, 1]))
        B = np.asarray(jax.vmap(lambda x: self.field_fn(x, jnp.asarray(c)))(jnp.asarray(seeds)))
        B_phi = (seeds[:, 0] * B[:, 1] - seeds[:, 1] * B[:, 0]) / launches[:, 0]
        signs = np.sign(B_phi * (np.asarray(reference_phi) - self.phi_start))
        if np.any(signs == 0) or np.any(signs != signs[0]):
            raise ValueError("All launches must reach their hits along the same direction of B")
        return float(signs[0])

    def _trace(self, c, reference_phi, direction):
        """Local hits ``(n, 3)`` in ``(R, Z, phi)`` and ``(x, y, z)``, lengths and validity."""
        launches = self._launches(c)
        seeds = jnp.column_stack((launches[:, 0] * jnp.cos(self.phi_start),
                                  launches[:, 0] * jnp.sin(self.phi_start), launches[:, 1]))
        wall_parameters = self.wall_parameters(c)
        result = connection_length(_ParametrizedField(self.field_fn, c), seeds,
                                   lambda xyz: self.wall_fn(xyz, wall_parameters),
                                   max_length=self.max_length, tolerance=self.tolerance,
                                   adjoint=diffrax.ForwardMode(), directions=(direction,))
        xyz = result["strike_points"][:, 0]
        lengths = result["lengths"][:, 0]
        wrapped = jnp.arctan2(xyz[:, 1], xyz[:, 0])
        # Unwrap onto the branch of the accepted (production) hit angle.
        turns = jax.lax.stop_gradient(jnp.round((jnp.asarray(reference_phi) - wrapped) / (2 * jnp.pi)))
        hits = jnp.column_stack((jnp.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2], wrapped + 2 * jnp.pi * turns))
        valid = jnp.all(result["hit"][:, 0]) & jnp.all(jnp.isfinite(lengths))
        return hits, xyz, lengths, valid

    def _residual(self, hits, lengths, c):
        values = jnp.asarray(self.observables(hits, lengths, self._weights(c), c))
        return (values - jnp.asarray(self.target)) / jnp.asarray(self.scales)

    def refresh(self, c, snapshot):
        from pyna.topo.open_validation import OpenBundleValidation

        self._check_snapshot(snapshot)
        checked = self.production_validate(np.array(c, copy=True), dict(snapshot))
        if not isinstance(checked, OpenBundleValidation):
            raise TypeError("Production callback must return OpenBundleValidation")
        if not checked.valid:
            return Evaluation(np.zeros(len(self.target)), snapshot, valid=False, status=checked.status)
        if checked.hit_RZPhi.shape != (len(self.bundle.labels), 3) or checked.connection_lengths.shape != (len(self.bundle.labels),):
            raise ValueError("Production launch identity/shape changed")
        reference_phi = checked.hit_RZPhi[:, 2]
        hits, xyz, _, local_valid = self._trace(jnp.asarray(c), reference_phi, self._direction(c, reference_phi))
        R, Z, phi = checked.hit_RZPhi.T
        physical_hits = np.column_stack((R * np.cos(phi), R * np.sin(phi), Z))
        discrepancy = float(np.max(np.linalg.norm(np.asarray(xyz) - physical_hits, axis=1)))
        phase_error = float(np.max(np.abs(np.asarray(hits)[:, 2] - phi)))
        valid = (bool(local_valid) and np.isfinite(discrepancy) and discrepancy <= self.maximum_hit_discrepancy
                 and phase_error <= self.maximum_phi_shift)
        if snapshot and np.max(np.abs(phi - np.asarray(snapshot["hit_phi"]))) > self.maximum_phi_shift:
            valid = False
        updated = dict(definition=self.fingerprint, hit_phi=phi.tolist(), hits=checked.hit_RZPhi.tolist(),
                       lengths=checked.connection_lengths.tolist(), parameters=np.asarray(c).tolist())
        return Evaluation(np.asarray(self._residual(checked.hit_RZPhi, checked.connection_lengths, c)), updated,
                          valid=valid, status="valid" if valid else "local_production_correspondence_failed",
                          diagnostics={"hit_discrepancy_m": discrepancy, "unwrapped_phase_discrepancy_rad": phase_error})

    def model(self, c, snapshot):
        self._check_snapshot(snapshot)
        if not snapshot:
            raise ValueError("Production refresh required before local linearization")
        if not np.array_equal(c, np.asarray(snapshot["parameters"])):
            raise ValueError("Model parameters differ from accepted snapshot")
        reference_phi = np.asarray(snapshot["hit_phi"])
        direction = self._direction(c, reference_phi)

        def residual(parameters):
            hits, _, lengths, _ = self._trace(parameters, reference_phi, direction)
            return self._residual(hits, lengths, parameters)

        _, _, _, valid = self._trace(jnp.asarray(c), reference_phi, direction)
        # The model value uses accepted production observables; the smooth
        # model Jacobian is qualified separately against production differences.
        value = self._residual(jnp.asarray(snapshot["hits"]), jnp.asarray(snapshot["lengths"]), jnp.asarray(c))
        return Evaluation(np.asarray(value), snapshot, jacobian=np.asarray(jax.jacfwd(residual)(jnp.asarray(c))),
                          valid=bool(valid))
