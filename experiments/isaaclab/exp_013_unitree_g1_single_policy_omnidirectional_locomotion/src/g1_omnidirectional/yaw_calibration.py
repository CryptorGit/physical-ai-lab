"""Command-only yaw calibration proposed by EXP013 Phase W1B-C1."""

from __future__ import annotations

import torch


NAME = "MonotonicPositiveYawCalibrationV1"
POSITIVE_GAIN = 1.5
NEGATIVE_GAIN = 1.0
ACTOR_INPUT_RANGE = (-1.0, 1.0)


def calibrate_yaw(yaw_target):
    """Map a physical yaw-rate target to the frozen actor's command input."""
    if isinstance(yaw_target, torch.Tensor):
        calibrated = torch.where(yaw_target > 0, yaw_target * POSITIVE_GAIN, yaw_target)
        return calibrated.clamp(*ACTOR_INPUT_RANGE)
    value = float(yaw_target)
    calibrated = value * POSITIVE_GAIN if value > 0 else value
    return min(max(calibrated, ACTOR_INPUT_RANGE[0]), ACTOR_INPUT_RANGE[1])

