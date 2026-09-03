"""Aggregate same-seed counterfactual branch replays and local trade-offs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage12_tangential_slip_reward_directionality"
RAW = OUT / "raw"
SPEEDS = (0.2, 0.4, 0.6, 1.2, 2.0)
JOINTS = (
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


standard_files = [
    RAW / f"counterfactual_{str(speed).replace('.', 'p')}_standard.pt" for speed in SPEEDS
]
linearity_files = [
    RAW / f"counterfactual_{str(speed).replace('.', 'p')}_linearity.pt" for speed in SPEEDS
]
if not all(path.exists() for path in standard_files + linearity_files):
    missing = [str(path) for path in standard_files + linearity_files if not path.exists()]
    raise SystemExit(f"counterfactual files missing: {missing}")

manifests, matching_rows, all_rows = [], [], []
for path in standard_files + linearity_files:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifests.append({
        "path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size,
        "speed": payload["speed"], "mode": payload["mode"],
        "episodes": payload["episodes"], "variants": len(payload["matching"]),
    })
    for item in payload["matching"]:
        matching_rows.append({"speed": payload["speed"], "mode": payload["mode"], **item})
    if payload["mode"] == "standard":
        all_rows.extend(payload["rows"])

all_matched = all(row["matched"] for row in matching_rows)
max_matching_error = max(
    value for row in matching_rows
    for key, value in row.items()
    if key not in ("speed", "mode", "dimension", "delta", "matched")
)
dump("counterfactual_branch_manifest.json", {
    "seed_root": 20272901, "speeds": list(SPEEDS),
    "states_per_speed": 100, "total_primary_branch_states": 500,
    "normalized_action_perturbation": 0.02,
    "joint_target_equivalent_rad": 0.005,
    "linearity_states_per_speed": 20,
    "linearity_perturbations": [0.01, -0.01, 0.04, -0.04],
    "horizons_steps": [1, 2, 4, 8], "primary_horizon_steps": 8,
    "files": manifests, "state_setter": False, "teleport": False,
    "cross_environment_state_injection": False,
})
dump("prebranch_matching_audit.json", {
    "all_replays_matched": all_matched,
    "tolerance": 1.0e-5, "maximum_absolute_error": max_matching_error,
    "fields": ["root", "joint", "previous_action", "contact_age", "heading_state"],
    "rows": matching_rows,
})

by_state = defaultdict(list)
for row in all_rows:
    score_reduction = (
        (row["baseline_cumulative_score_8"] - row["perturbed_cumulative_score_8"])
        / max(row["baseline_cumulative_score_8"], 1.0e-12)
    )
    slip_lower = row["perturbed_cumulative_score_8"] < row["baseline_cumulative_score_8"]
    speed_worsening = row["perturbed_speed_error_8"] - row["baseline_speed_error_8"]
    heading_worsening = row["perturbed_heading_8"] - row["baseline_heading_8"]
    improving = (
        row["prebranch_matched"] and score_reduction >= 0.20
        and speed_worsening <= 0.03 and heading_worsening <= 0.02
        and not row["new_contact_loss"] and not row["flight"] and not row["fall"]
        and not row["saturation_increase"]
    )
    row.update({
        "score_reduction_fraction": score_reduction, "slip_lower": slip_lower,
        "speed_worsening": speed_worsening, "heading_worsening": heading_worsening,
        "locally_improving": improving,
    })
    by_state[(row["speed"], row["episode"])].append(row)

state_rows = []
for (speed, episode), rows in by_state.items():
    candidates = [row for row in rows if row["locally_improving"]]
    best = max(rows, key=lambda row: row["score_reduction_fraction"])
    state_rows.append({
        "speed": speed, "episode": episode, "improving_exists": bool(candidates),
        "improving_count": len(candidates),
        "best_reduction": best["score_reduction_fraction"],
        "best_dimension": best["dimension"], "best_delta": best["delta"],
        "baseline_score": best["baseline_cumulative_score_8"],
        "finite_difference_gradient_magnitude": abs(
            best["perturbed_cumulative_score_8"] - best["baseline_cumulative_score_8"]
        ) / 0.02,
    })

speed_summary = {}
rates = []
for speed in SPEEDS:
    states = [row for row in state_rows if row["speed"] == speed]
    rate = np.mean([row["improving_exists"] for row in states])
    rates.append(rate)
    speed_summary[f"{speed:g}"] = {
        "branch_states": len(states), "locally_improving_state_rate": float(rate),
        "median_best_reduction": float(np.median([row["best_reduction"] for row in states])),
        "median_local_gradient_magnitude": float(np.median([
            row["finite_difference_gradient_magnitude"] for row in states
        ])),
    }
overall_rate = float(np.mean([row["improving_exists"] for row in state_rows]))
if all(rate >= 0.30 for rate in rates):
    controllability = "SLIP_LOCALLY_CONTROLLABLE"
elif overall_rate >= 0.10 or any(rate >= 0.30 for rate in rates):
    controllability = "SLIP_PARTIALLY_CONTROLLABLE"
else:
    controllability = "SLIP_NOT_LOCALLY_CONTROLLABLE"

dimension_summary = {}
for dimension, name in enumerate(JOINTS):
    rows = [row for row in all_rows if row["dimension"] == dimension]
    dimension_summary[name] = {
        "improving_variant_rate": float(np.mean([row["locally_improving"] for row in rows])),
        "slip_lower_variant_rate": float(np.mean([row["slip_lower"] for row in rows])),
        "mean_score_reduction": float(np.mean([row["score_reduction_fraction"] for row in rows])),
    }
leg_summary = {}
for leg in ("FL", "FR", "RL", "RR"):
    names = [index for index, name in enumerate(JOINTS) if name.startswith(leg)]
    rows = [row for row in all_rows if row["dimension"] in names]
    leg_summary[leg] = {
        "improving_variant_rate": float(np.mean([row["locally_improving"] for row in rows])),
        "slip_lower_variant_rate": float(np.mean([row["slip_lower"] for row in rows])),
    }
dump("local_slip_controllability.json", {
    "classification": controllability, "overall_improving_state_rate": overall_rate,
    "by_speed": speed_summary, "by_action_dimension": dimension_summary,
    "by_leg": leg_summary,
    "criterion": {
        "cumulative_raw_score_reduction": 0.20, "speed_worsening_m_s_lte": 0.03,
        "heading_worsening_rad_lte": 0.02, "new_contact_loss": False,
        "fall": False, "flight": False, "saturation_increase": False,
    },
})

# Trade-offs among all perturbations that lower slip, before hard feasibility filtering.
lowering = [row for row in all_rows if row["slip_lower"]]
conflict_speed = np.mean([row["speed_worsening"] > 0.03 for row in lowering])
conflict_heading = np.mean([row["heading_worsening"] > 0.02 for row in lowering])
conflict_contact = np.mean([
    row["new_contact_loss"] or row["flight"] for row in lowering
])
conflict_other = np.mean([row["fall"] or row["saturation_increase"] for row in lowering])
conflicts = {
    "speed": float(conflict_speed), "heading": float(conflict_heading),
    "contact": float(conflict_contact), "fall_or_saturation": float(conflict_other),
}
large = [name for name, rate in conflicts.items() if rate >= 0.50]
if not large:
    tradeoff = "SLIP_REDUCTION_WITHOUT_CAPABILITY_TRADEOFF"
elif large == ["speed"]:
    tradeoff = "SLIP_REDUCTION_CONFLICTS_WITH_SPEED"
elif large == ["heading"]:
    tradeoff = "SLIP_REDUCTION_CONFLICTS_WITH_HEADING"
elif large == ["contact"]:
    tradeoff = "SLIP_REDUCTION_CONFLICTS_WITH_CONTACT"
else:
    tradeoff = "MULTIPLE_LOCAL_TRADEOFFS"
dump("local_tradeoff_analysis.json", {
    "classification": tradeoff, "slip_lowering_variants": len(lowering),
    "conflict_rates": conflicts,
    "safe_improving_variants": sum(row["locally_improving"] for row in all_rows),
})

# Reward severity vs existence/magnitude of a local improving direction.
baseline_score = np.asarray([row["baseline_score"] for row in state_rows])
improving = np.asarray([row["improving_exists"] for row in state_rows], float)
gradient_magnitude = np.asarray([
    row["finite_difference_gradient_magnitude"] for row in state_rows
])


def corr(left, right):
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


dump("reward_controllability_agreement.json", {
    "reward_score_vs_improving_action_exists_point_biserial": corr(baseline_score, improving),
    "reward_score_vs_local_finite_difference_gradient": corr(baseline_score, gradient_magnitude),
    "high_reward_state_threshold_p80": float(np.percentile(baseline_score, 80)),
    "high_reward_improving_state_rate": float(improving[
        baseline_score >= np.percentile(baseline_score, 80)
    ].mean()),
    "interpretation": (
        "reward identifies severe states and finite-difference action direction is assessed "
        "without using a state setter"
    ),
})

# Local linearity audit uses paired 0.01 and 0.04 files.
linearity = []
for path in linearity_files:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = payload["rows"]
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["episode"], row["dimension"])][row["delta"]] = row
    slopes_small, slopes_large = [], []
    for variants in grouped.values():
        if not all(delta in variants for delta in (0.01, -0.01, 0.04, -0.04)):
            continue
        baseline = variants[0.01]["baseline_cumulative_score_8"]
        slopes_small.append(
            (variants[0.01]["perturbed_cumulative_score_8"]
             - variants[-0.01]["perturbed_cumulative_score_8"]) / 0.02
        )
        slopes_large.append(
            (variants[0.04]["perturbed_cumulative_score_8"]
             - variants[-0.04]["perturbed_cumulative_score_8"]) / 0.08
        )
    linearity.append({
        "speed": payload["speed"], "pairs": len(slopes_small),
        "small_large_slope_correlation": corr(slopes_small, slopes_large),
        "median_absolute_slope_ratio": float(np.median(
            np.abs(np.asarray(slopes_large)) / np.maximum(np.abs(slopes_small), 1e-12)
        )),
    })

dump("counterfactual_action_results.json", {
    "primary_variant_rows": len(all_rows), "state_summary": state_rows,
    "by_speed": speed_summary, "by_action_dimension": dimension_summary,
    "linearity": linearity,
    "raw_rows": "stored in Git-excluded raw counterfactual tensors",
})
print(json.dumps({
    "all_matched": all_matched, "max_error": max_matching_error,
    "controllability": controllability, "overall_rate": overall_rate,
    "tradeoff": tradeoff,
}, indent=2))
