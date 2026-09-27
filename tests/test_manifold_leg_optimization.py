"""Heat-independent leg objectives, continuation contracts and rollback."""

from dataclasses import fields, replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from essos.losses import custom_loss
from essos.topology_objectives import (
    ManifoldLegStage2Target, make_manifold_leg_stage2_loss,
    require_same_leg_configuration,
)
from essos.topology_validation import ManifoldLegValidationConfig
from essos.topology_objectives import (
    ManifoldContinuationSchedule, ManifoldContinuationStage,
    ManifoldContinuationState, compose_manifold_stage2_loss,
    accept_manifold_continuation_stage,
)
from essos.topology_validation import validated_manifold_backtracking_step
from tests.test_manifold_heat_optimization import (
    _heat_target, _sample_target, _leg_wall_signed_distance,
    _ShiftedHyperbolicField, PARAMETERS,
)
import tests.test_manifold_validation as integration


def _leg_target(source, wall_signed_distance, **overrides):
    kwargs = {item.name: getattr(source, item.name) for item in fields(ManifoldLegStage2Target)
              if hasattr(source, item.name)}
    kwargs.update(wall_signed_distance=wall_signed_distance, minimum_clearance_m=.001,
                  clearance_scale_m=.001, terminal_exclusion_fraction=.2)
    kwargs.update(overrides)
    return ManifoldLegStage2Target(**kwargs)


def test_leg_target_composes_without_heat_and_zero_weight_never_traces(monkeypatch):
    source = _heat_target()
    target = _leg_target(source, _leg_wall_signed_distance, terminal_exclusion_fraction=.05)
    active = ManifoldContinuationState(
        ManifoldContinuationSchedule((ManifoldContinuationStage(name="legs", leg_clearance_weight=.3),)),
        _sample_target(source), leg_target_state=target,
    )
    assert active.heat_target_state is None
    field = _ShiftedHyperbolicField(jnp.asarray(PARAMETERS))
    base = custom_loss(lambda field: field.parameters[0] ** 2, "field")
    total = compose_manifold_stage2_loss(base, active, dependencies={"field": field})
    standalone = make_manifold_leg_stage2_loss(target)
    standalone.dependencies = {"field": field}
    value, gradient = standalone.value_and_grad(standalone.starting_dofs)
    combined_value, combined_gradient = total(total.starting_dofs), total.grad(total.starting_dofs)
    np.testing.assert_allclose(combined_value, PARAMETERS[0] ** 2 + .3 * value)
    np.testing.assert_allclose(combined_gradient, np.array([2 * PARAMETERS[0], 0, 0]) + .3 * gradient)
    direction, delta = np.array([.2, -.3, .4]), 2e-7
    finite_difference = (standalone(standalone.starting_dofs + delta * direction)
                         - standalone(standalone.starting_dofs - delta * direction)) / (2 * delta)
    np.testing.assert_allclose(np.dot(gradient, direction), finite_difference, rtol=3e-5, atol=3e-7)

    def forbidden(*args, **kwargs):
        raise AssertionError("inactive leg objective traced")
    monkeypatch.setattr("essos.topology_objectives.trace_manifold_leg_reference", forbidden)
    inactive = replace(active, schedule=ManifoldContinuationSchedule((ManifoldContinuationStage(name="base"),)))
    ordinary = compose_manifold_stage2_loss(base, inactive, dependencies={"field": field})
    np.testing.assert_allclose(ordinary(ordinary.starting_dofs), PARAMETERS[0] ** 2)
    with pytest.raises(ValueError, match="leg_target_state"):
        compose_manifold_stage2_loss(base, replace(active, leg_target_state=None))


@pytest.mark.parametrize("override", [
    {"minimum_clearance_m": -.1}, {"clearance_scale_m": 0},
    {"terminal_exclusion_fraction": 1}, {"sample_stride": True},
    {"strike_weights": [0., 0.]}, {"strike_weights": [1.]},
])
def test_leg_target_rejects_invalid_configuration(override):
    with pytest.raises((ValueError, TypeError)):
        _leg_target(_heat_target(), _leg_wall_signed_distance, **override)


def test_leg_refresh_preserves_all_static_physics():
    target = _leg_target(_heat_target(), _leg_wall_signed_distance)
    weights = np.array([1., 2.])
    frozen = replace(target, strike_weights=weights)
    weights[0] = 99
    assert frozen.strike_weights == (1., 2.)
    for change in ({"minimum_clearance_m": .002}, {"terminal_exclusion_fraction": .3},
                   {"strike_weights": (2., 1.)}, {"wall_n_steps": 64},
                   {"wall_signed_distance": lambda p: 1.}):
        with pytest.raises(ValueError, match="accepted leg refresh changed"):
            require_same_leg_configuration(target, replace(target, **change))


def _vessel_distance(point):
    R = jnp.hypot(point[0], point[1])
    return .02 - jnp.hypot(R - 1., point[2])


def test_production_leg_transaction_accepts_refresh_and_rolls_back_clearance_failure():
    integration._require_cyna()
    original = integration._continuation_state()
    source = integration._heat_target(original.target_state.branch_reference)
    target = _leg_target(source, _vessel_distance, strike_weights=(1., 0.))
    state = replace(original, leg_target_state=target)
    field = integration._ShiftedHyperbolicField(jnp.array([integration.RATE, 1., 0.]))
    config = ManifoldLegValidationConfig(
        wall=integration._strike_wall(), maximum_hit_displacement_m=.002,
        jax_cyna_tolerance_m=1e-5, maximum_endpoint_error_m=1e-6,
        maximum_projection_distance_m=1e-5, production_DPhi=.005, max_turns=3,
    )
    with pytest.raises(TypeError, match="ManifoldLegValidationConfig"):
        integration._validate(field, state)
    accepted = integration._validate(field, state, leg_validation_config=config)
    assert accepted.accepted, accepted.rejection_reason
    assert accepted.leg_validation.production.minimum_clearance_m > .005
    assert accepted.accepted_state.leg_target_state.labels == target.labels
    assert accepted.accepted_state.leg_target_state.wall_signed_distance is target.wall_signed_distance
    assert state.stage_index == 0
    assert accepted.accepted_state.stage_index == 1
    with pytest.raises(ValueError, match="requires accepted refresh state"):
        accept_manifold_continuation_stage(state, accepted.candidate_branch,
                                          accepted.sample_refresh, accepted.correspondence)

    # A zero-weight second leg still has a mandatory production margin.
    failing = replace(state, leg_target_state=replace(target, minimum_clearance_m=.008))
    result = validated_manifold_backtracking_step(
        np.array([1.]), np.array([1.000001]),
        lambda dofs: integration._ShiftedHyperbolicField(jnp.array([integration.RATE, dofs[0], 0.])),
        failing,
        lambda candidate, snapshot: integration._validate(candidate, snapshot, leg_validation_config=config),
        maximum_attempts=2,
    )
    assert not result.accepted
    assert result.continuation_state is failing
    np.testing.assert_array_equal(result.dofs, [1.])
    assert all(attempt.validation.rejection_reason == "leg_validation:production_leg_clearance_below_limit"
               for attempt in result.attempts), [a.validation.rejection_reason for a in result.attempts]
    assert all(attempt.validation.accepted_state is None for attempt in result.attempts)
    assert all(a.validation.leg_validation.production.legs[0].accepted for a in result.attempts)
    assert all(not a.validation.leg_validation.production.legs[1].accepted for a in result.attempts)

    # The independent inner gate must also reject a bad smooth wall model.
    bad_inner = replace(state, leg_target_state=replace(target, wall_signed_distance=lambda point: jnp.asarray(0.)))
    rejected = integration._validate(field, bad_inner, leg_validation_config=config)
    assert rejected.leg_validation.production.accepted
    assert rejected.rejection_reason == "leg_validation:inner_leg_clearance_below_limit"
