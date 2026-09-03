"""Pure tensor reward contract for the frozen RUN_TO_WALK Pilot 1 protocol."""
from __future__ import annotations

import torch


def reward_terms(x: dict[str, torch.Tensor], weights: dict[str, float], thresholds: dict[str, float]):
    zero = torch.zeros_like(x["speed"])
    speed_error = x["speed"] - x["target_speed"]
    raw = {
        "speed_reduction_progress": (x["previous_speed"] - x["speed"]).clamp(-1.0, 1.0),
        "walk_speed_tracking": torch.exp(-((speed_error) / thresholds["speed_tracking_scale_mps"]) ** 2),
        "speed_overshoot": torch.relu(x["target_speed"] - x["speed"] - thresholds["overshoot_below_target_mps"]).square(),
        "reverse_velocity": (x["speed"] < thresholds["reverse_velocity_mps"]).float(),
        "heading_tracking": torch.exp(-(x["heading_error"] / thresholds["heading_tracking_scale_rad"]) ** 2),
        "lateral_velocity": x["lateral_velocity"].square(),
        "upright": x["tilt"].square(),
        "excessive_tilt": (x["tilt"] > thresholds["excessive_tilt_rad"]).float(),
        "excessive_flight_reduction": x["flight_reduction_event"].float(),
        "valid_landing": x["valid_landing"].float(),
        "run_cycle_termination": x["run_cycle_terminated"].float(),
        "flight_frequency_reduction": x["flight_frequency_reduced"].float(),
        "vertical_velocity": x["vertical_velocity"].square(),
        "walk_compatible_contact": x["walk_compatible_contact"].float(),
        "stable_support": x["stable_support"].float(),
        "walk_contract_progress": x["walk_contract_progress"].clamp(0.0, 1.0),
        "walk_acceptance": x["walk_acceptance_first"].float(),
        "fall": x["fall"].float(),
        "torso_contact": x["torso_contact"].float(),
        "dangerous_slip": x["dangerous_slip"].float(),
        "impact": x["impact_failure"].float(),
        "ankle_effort_dwell": x["ankle_saturation"].float(),
        "knee_velocity_dwell": x["knee_saturation"].float(),
        "joint_limit": x["joint_limit"].float(),
        "action_rate": x["action_rate"].square(),
        "entry_run_action_alignment": x["entry_action_error"].square() * x["entry_alignment_gate"].float(),
        "exit_walk_action_alignment": x["exit_action_error"].square() * x["exit_alignment_gate"].float(),
        "walk_acceptance_bonus": x["completion_first"].float(),
    }
    weighted = {name: raw[name] * float(weights[name]) for name in raw}
    return raw, weighted, sum(weighted.values(), zero)
