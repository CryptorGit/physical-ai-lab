"""Bounded, analysis-only action candidates for replay audits."""

from __future__ import annotations

import torch


def candidate_actions(baseline, walk, run, joint_groups, delta=0.05):
    result = {
        "baseline": baseline,
        "walk_expert": walk,
        "run_expert": run,
        "target_walk_alignment": baseline + (walk - baseline).clamp(-delta, delta),
    }
    for name, indices in joint_groups.items():
        for sign in (-1.0, 1.0):
            action = baseline.clone()
            action[:, indices] += sign * delta
            result[f"bounded_{name}_{'minus' if sign < 0 else 'plus'}"] = action
    return result


def state_match(reference, candidate, tolerances):
    errors = {
        key: float((reference[key] - candidate[key]).abs().max())
        for key in reference
    }
    return errors, all(errors[key] <= tolerances[key] for key in errors)
