from __future__ import annotations

from qmini_population_bwm.data_schema import (
    CanonicalBodilyObservation,
    TransitionRecord,
    validate_reward_separation,
)


def test_reward_vector_is_separate_from_canonical_observation() -> None:
    observation = CanonicalBodilyObservation(
        base_linear_velocity_b=(0.0, 0.0, 0.0),
        base_angular_velocity_b=(0.0, 0.0, 0.0),
        projected_gravity_b=(0.0, 0.0, -1.0),
        joint_position=(0.0,) * 10,
        joint_velocity=(0.0,) * 10,
        previous_actually_applied_action=(0.0,) * 10,
        left_foot_contact=1.0,
        right_foot_contact=1.0,
    )
    record = TransitionRecord(
        episode_id="e",
        source_snapshot_id="s",
        t=0,
        observation=observation,
        next_observation=observation,
        action_proposed=(0.0,) * 10,
        action_applied=(0.0,) * 10,
        command=(0.1, 0.0, 0.0),
        reward_vector={"upright": 1.0, "velocity": 0.5},
    )
    assert validate_reward_separation(record) == []
    assert record.reward_vector["upright"] == 1.0
