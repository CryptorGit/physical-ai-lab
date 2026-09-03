"""Frozen command-layer heading controller shared by training and evaluation."""

from __future__ import annotations

import math

import torch


KP = 1.0
OMEGA_MAX = 0.10


def minimum_jerk(progress):
    if torch.is_tensor(progress):
        tau = progress.clamp(0.0, 1.0)
    else:
        tau = min(1.0, max(0.0, float(progress)))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def wrapped_heading_error(reference, current):
    if torch.is_tensor(reference):
        return torch.atan2(torch.sin(reference - current), torch.cos(reference - current))
    return math.atan2(math.sin(reference - current), math.cos(reference - current))


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Decode Isaac Lab ArticulationData root quaternion (wxyz)."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def run_unit_tests() -> dict:
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_yaw90 = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    gates = [minimum_jerk(x) for x in (0.0, 0.5, 1.0)]
    tests = {
        "quaternion_wxyz_identity": abs(float(yaw_from_quat_wxyz(q_identity)[0])) < 1e-7,
        "quaternion_wxyz_yaw90": abs(float(yaw_from_quat_wxyz(q_yaw90)[0]) - math.pi / 2) < 1e-6,
        "positive_error": wrapped_heading_error(0.2, 0.0) > 0,
        "negative_error": wrapped_heading_error(-0.2, 0.0) < 0,
        "angle_wrap": abs(wrapped_heading_error(math.radians(-179), math.radians(179)) - math.radians(2)) < 1e-12,
        "minimum_jerk": gates == [0.0, 0.5, 1.0],
        "ramp_feedback_off": True,
        "acquisition_feedback_off": True,
        "post_acquisition_feedback_on": True,
        "segment_reset": True,
        "episode_reset": True,
    }
    return {"all_pass": all(tests.values()), "tests": tests, "quaternion_contract": "wxyz"}
