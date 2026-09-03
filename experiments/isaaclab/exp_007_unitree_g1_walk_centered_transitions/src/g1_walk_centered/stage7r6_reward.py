"""Pure tensor reward contract frozen for the Stage 7R Pilot 1 protocol."""
from __future__ import annotations
import torch

def reward_terms(x: dict[str, torch.Tensor], weights: dict[str, float], thresholds: dict[str, float]):
    z=torch.zeros_like(x["speed"])
    terms={
      "speed_progress": (x["speed"]-x["previous_speed"]).clamp(-1,1),
      "speed_tracking": torch.exp(-((x["speed"]-x["target_speed"])/thresholds["speed_tracking_scale_mps"])**2),
      "heading_tracking": torch.exp(-(x["heading_error"]/thresholds["heading_tracking_scale_rad"])**2),
      "lateral_velocity": x["lateral_velocity"].square(),
      "upright": x["tilt"].square(),
      "safe_liftoff": x["safe_liftoff"].float(),
      "safe_flight": x["safe_flight"].float(),
      "valid_landing": x["valid_landing"].float(),
      "alternating_landing": x["alternating_landing"].float(),
      "consecutive_periodic_cycle": x["consecutive_cycle"].float(),
      "dangerous_slip": x["dangerous_slip"].float(),
      "impact": x["impact_failure"].float(),
      "ankle_effort_dwell": x["ankle_saturation"].float(),
      "knee_velocity_dwell": x["knee_saturation"].float(),
      "excessive_flight": x["excessive_flight"].float(),
      "fall": x["fall"].float(),
      "torso_contact": x["torso_contact"].float(),
      "joint_limit": x["joint_limit"].float(),
      "action_rate": x["action_rate"].square(),
      "source_action_alignment": x["source_action_error"].square()*x["source_alignment_gate"].float(),
      "target_action_alignment": x["target_action_error"].square()*x["target_alignment_gate"].float(),
      "run_acceptance_bonus": x["acceptance_first"].float(),
    }
    weighted={name:terms[name]*float(weights[name]) for name in terms}
    return weighted, sum(weighted.values(),z)
