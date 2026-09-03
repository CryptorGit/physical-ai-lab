"""Pure NumPy helpers for the v59 controller parity audit.

This module deliberately does not import the training environment.  It is the
independent side of the teacher-forced controller comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


COMMANDS = (
    {"command_id": "C0_stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0},
    {"command_id": "C1_forward", "vx": 0.10, "vy": 0.0, "yaw_rate": 0.0},
    {"command_id": "C2_backward", "vx": -0.10, "vy": 0.0, "yaw_rate": 0.0},
    {"command_id": "C3_yaw_left", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.60},
    {
        "command_id": "C4_backward_right_max",
        "vx": -0.07,
        "vy": 0.0,
        "yaw_rate": -0.30,
    },
)


@dataclass(frozen=True)
class CompositionResult:
    teacher_active: bool
    teacher_action: np.ndarray
    residual_scaled: np.ndarray
    combined_pre_limit: np.ndarray
    motor_target: np.ndarray


def swish(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def numpy_actor(raw_observation, mean, std, layers):
    """Runs the exported v59 deterministic actor using checkpoint arrays."""
    x = (np.asarray(raw_observation, np.float32) - mean) / std
    for index, (kernel, bias) in enumerate(layers):
        x = x @ kernel + bias
        if index + 1 != len(layers):
            x = swish(x)
    return np.tanh(x[..., : x.shape[-1] // 2])


def compose_motor_target(
    action,
    *,
    command,
    default,
    lower,
    upper,
    action_scale,
    previous_target,
    max_motor_velocity,
    dt,
    backward_reference,
    backward_actuator_indices,
    backward_joint_indices,
    backward_residual_scale,
    coupled_slope,
    coupled_intercept,
):
    """Reproduces joystick.py's deterministic v59 action composition."""
    bounded = np.clip(np.asarray(action), -1.0, 1.0)
    positive_span = 0.9 * (upper - default)
    negative_span = 0.9 * (default - lower)
    directional_span = np.where(bounded >= 0.0, positive_span, negative_span)
    base_span = np.minimum(action_scale, directional_span)
    magnitude = np.abs(bounded)
    direct_delta = np.sign(bounded) * (
        base_span * magnitude + (directional_span - base_span) * magnitude**5
    )
    direct_target = default + direct_delta

    teacher = default.copy()
    teacher[np.asarray(backward_actuator_indices)] = backward_reference[
        np.asarray(backward_joint_indices)
    ]
    turn_blend = np.clip(abs(float(command[2])) / 0.20, 0.0, 1.0)
    residual_scales = np.full(
        default.shape,
        backward_residual_scale * max(0.50, 1.0 - 0.50 * turn_blend),
    )
    residual_scales[5:9] = backward_residual_scale * (1.0 - 0.5 * turn_blend)
    reverse = float(command[0]) < -0.02
    residual_scaled = residual_scales * bounded if reverse else direct_delta
    combined = teacher + residual_scaled if reverse else direct_target

    speed_limited = np.clip(
        combined,
        previous_target - max_motor_velocity * dt,
        previous_target + max_motor_velocity * dt,
    )
    coupled_upper = coupled_intercept - coupled_slope * speed_limited[5]
    speed_limited[6] = min(speed_limited[6], coupled_upper)
    final = np.clip(speed_limited, lower, upper)
    return CompositionResult(
        teacher_active=reverse,
        teacher_action=teacher if reverse else np.zeros_like(default),
        residual_scaled=residual_scaled,
        combined_pre_limit=combined,
        motor_target=final,
    )


def error_metrics(expected, actual):
    expected = np.asarray(expected, np.float64)
    actual = np.asarray(actual, np.float64)
    delta = actual - expected
    flat_expected = expected.reshape(-1)
    flat_actual = actual.reshape(-1)
    denom = np.linalg.norm(flat_expected) * np.linalg.norm(flat_actual)
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "cosine_similarity": (
            float(np.dot(flat_expected, flat_actual) / denom)
            if denom
            else (1.0 if np.array_equal(flat_expected, flat_actual) else 0.0)
        ),
    }
