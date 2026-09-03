"""Frozen math and contact metrics for GO2_ENDPOINT_EVALUATION_V1."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

try:
    import torch
except ImportError:  # pure-math protocol preparation remains usable without torch
    torch = None


def _components(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    if len(quaternion) != 4:
        raise ValueError("quaternion must contain exactly four xyzw elements")
    x, y, z, w = (float(value) for value in quaternion)
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        raise ValueError("quaternion contains NaN or Inf")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    return x / norm, y / norm, z / norm, w / norm


def quat_xyzw_to_rotation_matrix(quaternion: Sequence[float]) -> list[list[float]]:
    x, y, z, w = _components(quaternion)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_xyzw_to_roll_pitch_yaw(quaternion: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = _components(quaternion)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_xyzw_to_heading(quaternion: Sequence[float]) -> float:
    return quat_xyzw_to_roll_pitch_yaw(quaternion)[2]


def quat_xyzw_to_gravity_tilt(quaternion: Sequence[float]) -> float:
    matrix = quat_xyzw_to_rotation_matrix(quaternion)
    return math.acos(max(-1.0, min(1.0, matrix[2][2])))


def quat_xyzw_to_roll_pitch_yaw_torch(quaternion):
    """Vectorized xyzw decode for runtime telemetry."""
    if torch is None:
        raise RuntimeError("torch is required for tensor quaternion decoding")
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = quaternion.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_xyzw_to_gravity_tilt_torch(quaternion):
    if torch is None:
        raise RuntimeError("torch is required for tensor quaternion decoding")
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, _, w = quaternion.unbind(-1)
    body_z_world_z = 1 - 2 * (x * x + y * y)
    return torch.acos(torch.clamp(body_z_world_z, -1.0, 1.0))


def heading_error(yaw: float, reference: float) -> float:
    return math.atan2(math.sin(yaw - reference), math.cos(yaw - reference))


def circular_median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("circular median requires values")
    candidates = list(values)
    return min(candidates, key=lambda candidate: sum(abs(heading_error(value, candidate)) for value in values))


def max_true_run(flags: Sequence[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def physical_slip_intervals(
    contact_forces_n: Sequence[float],
    contact_points_xy: Sequence[Sequence[float] | None],
    dt: float = 0.02,
    force_threshold_n: float = 5.0,
    minimum_contact_steps: int = 3,
    exclude_onset_steps: int = 2,
    exclude_release_steps: int = 2,
    anchor_steps: int = 3,
    displacement_threshold_m: float = 0.03,
    speed_threshold_mps: float = 0.30,
    minimum_speed_steps: int = 5,
) -> dict:
    """Analyze actual world contact-point motion with frozen boundary handling."""
    if len(contact_forces_n) != len(contact_points_xy):
        raise ValueError("force and point traces differ in length")
    candidates = [
        force > force_threshold_n and point is not None and all(math.isfinite(float(v)) for v in point[:2])
        for force, point in zip(contact_forces_n, contact_points_xy)
    ]
    raw_intervals: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(candidates + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            raw_intervals.append((start, index))
            start = None

    intervals = []
    excluded_short = 0
    all_speeds: list[float] = []
    all_displacements: list[float] = []
    stable_steps = 0
    dangerous_count = 0
    for raw_start, raw_end in raw_intervals:
        if raw_end - raw_start < minimum_contact_steps:
            excluded_short += 1
            continue
        start = raw_start + exclude_onset_steps
        end = raw_end - exclude_release_steps
        if end - start < anchor_steps:
            excluded_short += 1
            continue
        points = [[float(v) for v in contact_points_xy[index][:2]] for index in range(start, end)]
        anchor_x = sorted(point[0] for point in points[:anchor_steps])[anchor_steps // 2]
        anchor_y = sorted(point[1] for point in points[:anchor_steps])[anchor_steps // 2]
        displacement = [math.hypot(point[0] - anchor_x, point[1] - anchor_y) for point in points]
        speed = [0.0] + [
            math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1]) / dt
            for index in range(1, len(points))
        ]
        speed_flags = [value > speed_threshold_mps for value in speed]
        dangerous = max(displacement, default=0.0) > displacement_threshold_m or max_true_run(speed_flags) >= minimum_speed_steps
        dangerous_count += int(dangerous)
        stable_steps += len(points)
        all_speeds.extend(speed)
        all_displacements.extend(displacement)
        intervals.append({
            "raw_start": raw_start,
            "raw_end": raw_end,
            "stable_start": start,
            "stable_end": end,
            "anchor_xy": [anchor_x, anchor_y],
            "max_displacement_m": max(displacement, default=0.0),
            "max_speed_mps": max(speed, default=0.0),
            "maximum_contiguous_dangerous_speed_steps": max_true_run(speed_flags),
            "maximum_contiguous_dangerous_speed_s": max_true_run(speed_flags) * dt,
            "dangerous": dangerous,
        })
    return {
        "dangerous": dangerous_count > 0,
        "dangerous_interval_count": dangerous_count,
        "stable_contact_steps": stable_steps,
        "stable_contact_time_s": stable_steps * dt,
        "excluded_short_interval_count": excluded_short,
        "intervals": intervals,
        "speeds_mps": all_speeds,
        "anchor_displacements_m": all_displacements,
    }


def classify_go2_gait_v1(
    contacts: Sequence[Sequence[bool]], actual_speed: float, fall: bool
) -> tuple[str, dict]:
    """Diagnostic-only phase classifier; it never participates in a gate."""
    if fall:
        return "FALL", {"reason": "episode termination"}
    if not contacts:
        return "IRREGULAR", {"reason": "empty contact trace"}
    columns = list(zip(*contacts))
    duty = [sum(column) / len(column) for column in columns]
    flight = sum(not any(row) for row in contacts) / len(contacts)
    transitions = [
        [index for index in range(1, len(column)) if column[index] and not column[index - 1]]
        for column in columns
    ]

    def phase_similarity(first: int, second: int, tolerance: int = 3) -> float:
        if not transitions[first] or not transitions[second]:
            return 0.0
        matches = sum(
            any(abs(event - other) <= tolerance for other in transitions[second])
            for event in transitions[first]
        )
        return matches / max(len(transitions[first]), len(transitions[second]))

    diagonal = (phase_similarity(0, 3) + phase_similarity(1, 2)) / 2
    ipsilateral = (phase_similarity(0, 2) + phase_similarity(1, 3)) / 2
    fore_hind = (phase_similarity(0, 1) + phase_similarity(2, 3)) / 2
    support_counts = Counter(sum(row) for row in contacts)
    evidence = {
        "duty_factor": duty,
        "flight_fraction": flight,
        "diagonal_phase_synchrony": diagonal,
        "ipsilateral_phase_synchrony": ipsilateral,
        "fore_hind_phase_synchrony": fore_hind,
        "support_count_distribution": {str(key): value / len(contacts) for key, value in support_counts.items()},
    }
    if actual_speed <= 0.08 and min(duty) >= 0.90:
        label = "STAND_LIKE"
    elif sum(value >= 0.70 for value in duty) >= 3 and flight < 0.02:
        label = "CRAWL_LIKE"
    elif diagonal >= 0.60 and diagonal > ipsilateral + 0.10:
        label = "TROT_LIKE"
    elif ipsilateral >= 0.60 and ipsilateral > diagonal + 0.10:
        label = "PACE_LIKE"
    elif fore_hind >= 0.60 and flight >= 0.03:
        label = "BOUND_LIKE"
    else:
        label = "IRREGULAR"
    return label, evidence


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
