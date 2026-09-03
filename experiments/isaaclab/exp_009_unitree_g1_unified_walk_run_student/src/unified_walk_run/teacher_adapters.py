"""Teacher labels projected onto the canonical 123D student interface."""

from __future__ import annotations


def assert_teacher_action_contract(*actions):
    for action in actions:
        if action.shape[-1] != 37:
            raise ValueError(f"teacher action is not 37D: {action.shape}")


def canonical_student_observation(canonical_state, command, to_walk_observation):
    observation = to_walk_observation(canonical_state, command)
    if observation.shape[-1] != 123:
        raise ValueError(f"canonical student observation is not 123D: {observation.shape}")
    return observation
