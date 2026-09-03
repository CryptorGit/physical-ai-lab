from __future__ import annotations

from qmini_population_bwm.hidden_physics import HiddenPhysics, HiddenPhysicsRanges


def make_hidden(seed: int) -> HiddenPhysics:
    return HiddenPhysics(
        ranges=HiddenPhysicsRanges(
            friction=(0.4, 0.9),
            wind_x=(-0.2, 0.2),
            wind_y=(-0.3, 0.3),
        ),
        fatigue_alpha=0.1,
        fatigue_beta=0.01,
        fatigue_effectiveness_coefficient=0.5,
        seed=seed,
    )


def test_hidden_sampling_is_seed_deterministic() -> None:
    first = make_hidden(15015)
    second = make_hidden(15015)
    assert first.sample() == second.sample()
    assert first.sample() == second.sample()


def test_hidden_state_is_not_appended_to_policy_observation() -> None:
    hidden = make_hidden(1)
    observation = (1.0, 2.0, 3.0)
    assert hidden.policy_visible_observation(observation) == observation
    assert "friction" not in hidden.policy_visible_observation.__doc__.lower()
