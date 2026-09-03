"""Continuous minimum-jerk command profiles."""

import torch


def minimum_jerk(progress):
    progress = torch.as_tensor(progress).clamp(0, 1)
    return 10 * progress**3 - 15 * progress**4 + 6 * progress**5


def ramp(source, target, elapsed, duration):
    return source + (target - source) * minimum_jerk(elapsed / duration)
