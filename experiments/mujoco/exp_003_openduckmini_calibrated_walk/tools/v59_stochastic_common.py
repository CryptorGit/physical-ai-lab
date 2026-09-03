"""Independent stochastic-input primitives for the v59 parity audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class StochasticInputs:
    initial_qpos_offset: np.ndarray
    initial_qvel_offset: np.ndarray
    initial_base_velocity: np.ndarray
    observation_noise: np.ndarray
    delay_length: int
    delay_buffer_initial_state: np.ndarray
    backlash_initial_state: np.ndarray
    phase_initialization: float


def delay_buffer_step(
    action_history: np.ndarray, action: np.ndarray, delay: int
) -> tuple[np.ndarray, np.ndarray]:
    """Matches joystick.py: roll by one 14-joint control step, prepend, index."""
    action = np.asarray(action)
    history = np.asarray(action_history)
    joints = action.shape[-1]
    if history.size % joints:
        raise ValueError("history length must be a multiple of action size")
    rows = history.reshape(-1, joints)
    if not 0 <= int(delay) < rows.shape[0]:
        raise ValueError("delay is outside the allocated buffer")
    updated = np.roll(history, joints)
    updated[:joints] = action
    return updated, updated.reshape(-1, joints)[int(delay)].copy()


def inject_observation_noise(
    raw_before_noise: np.ndarray, observation_noise: np.ndarray
) -> np.ndarray:
    raw = np.asarray(raw_before_noise)
    noise = np.asarray(observation_noise)
    if raw.shape != (101,) or noise.shape != (101,):
        raise ValueError("v59 actor observations must have shape (101,)")
    return raw + noise


def normalized_observation(
    raw_after_noise: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (np.asarray(raw_after_noise) - np.asarray(mean)) / np.asarray(std)


def stochastic_actor_from_logits(
    logits: np.ndarray, standard_normal_sample: np.ndarray
) -> np.ndarray:
    """Matches Brax NormalTanhDistribution with injected N(0,1) samples."""
    loc, raw_scale = np.split(np.asarray(logits), 2, axis=-1)
    scale = np.logaddexp(0.0, raw_scale) + 0.001
    return np.tanh(loc + scale * np.asarray(standard_normal_sample))


def backlash_observation(
    actuator_qpos: np.ndarray, backlash_qpos_with_head_zeros: np.ndarray
) -> np.ndarray:
    """Backlash is a passive joint displacement added before observation noise."""
    return np.asarray(actuator_qpos) + np.asarray(backlash_qpos_with_head_zeros)


def json_value(value: Any):
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()
