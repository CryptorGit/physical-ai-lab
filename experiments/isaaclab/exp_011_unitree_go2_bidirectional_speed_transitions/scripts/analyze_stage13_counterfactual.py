"""Analyze valid fresh-process variants and classify local slip controllability."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage13_fresh_process_counterfactual_replay"
RAW = OUT / "raw"
JOINTS = (
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
)
LEGS = ("front-left", "front-right", "rear-left", "rear-right")


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_trace(run_id):
    return torch.load(RAW / f"{run_id}.pt", map_location="cpu", weights_only=False)


def prefix_match(baseline, variant, branch_step):
    fields = ("state_hash", "action_hash", "observation_hash", "contact_hash", "controller_state_hash")
    for step in range(branch_step + 1):
        for field in fields:
            if baseline["hash_rows"][step][field] != variant["hash_rows"][step][field]:
                return False, field, step
    return True, None, None


def outcome(payload, branch_step):
    trace = payload["trace"]
    indices = slice(branch_step, branch_step + 8)
    slip = trace["raw_slip_score"][indices, 0]
    tangent = trace["tangential_speed"][indices, 0]
    contact_age = trace["contact_age"][indices, 0]
    stable = contact_age >= 3
    dangerous = (tangent > 0.30) & stable
    friction = trace["friction_utilization"][indices, 0]
    speed_error = trace["speed_error"][indices, 0]
    heading = trace["heading_error"][indices, 0].abs()
    tilt = trace["gravity_tilt"][indices, 0]
    contact = trace["foot_contact"][branch_step:branch_step + 9, 0]
    branch_contact = contact[0]
    contact_loss = bool((branch_contact[None] & ~contact[1:]).any())
    flight = (contact[1:].sum(1) == 0).float()
    return {
        "cumulative_slip": float(slip.sum()),
        "slip_reward": float(-0.00559195994498 * slip.sum()),
        "tangential_p95": float(tangent.quantile(0.95)),
        "dangerous_fraction": float(dangerous.float().mean()),
        "friction_p95": float(friction.quantile(0.95)),
        "speed_error": float(speed_error.mean()),
        "heading_error": float(heading.max()),
        "gravity_tilt": float(tilt.max()),
        "contact_loss": contact_loss,
        "flight_fraction": float(flight.mean()),
        "fall": bool(trace["fall"][indices, 0].any()),
        "saturation": bool(trace["saturation"][indices, 0].any()),
        "normal_force_mean": float(trace["normal_force"][indices, 0].mean()),
    }


branches = load_json("counterfactual_branch_manifest.json")["branches"]
rows = []
valid = 0
expected = len(branches) * 24
baseline_cache = {}
for branch in branches:
    b0_id = branch["baseline_b0_run_id"]
    baseline = baseline_cache.setdefault(b0_id, load_trace(b0_id))
    base = outcome(baseline, branch["branch_step"])
    for dimension in range(12):
        for delta in (-0.02, 0.02):
            sign = "plus" if delta > 0 else "minus"
            magnitude = str(abs(delta)).replace(".", "p")
            run_id = (
                f"{branch['branch_id']}_action_{dimension:02d}_{sign}_{magnitude}"
            )
            path = RAW / f"{run_id}.pt"
            if not path.exists():
                rows.append({
                    "branch_id": branch["branch_id"], "speed": branch["speed"],
                    "dimension": dimension, "joint": JOINTS[dimension],
                    "leg": LEGS[dimension % 4], "delta": delta,
                    "valid": False, "invalid_reason": "TRACE_MISSING",
                })
                continue
            variant = load_trace(run_id)
            matched, mismatch_field, mismatch_step = prefix_match(
                baseline, variant, branch["branch_step"]
            )
            if not matched:
                rows.append({
                    "branch_id": branch["branch_id"], "speed": branch["speed"],
                    "dimension": dimension, "joint": JOINTS[dimension],
                    "leg": LEGS[dimension % 4], "delta": delta,
                    "valid": False,
                    "invalid_reason": f"PREFIX_{mismatch_field}_STEP_{mismatch_step}",
                })
                continue
            valid += 1
            result = outcome(variant, branch["branch_step"])
            new_contact_loss = result["contact_loss"] and not base["contact_loss"]
            slip_reduction = (
                (base["cumulative_slip"] - result["cumulative_slip"])
                / max(base["cumulative_slip"], 1.0e-12)
            )
            improving = (
                slip_reduction >= 0.20
                and result["speed_error"] - base["speed_error"] <= 0.03
                and result["heading_error"] - base["heading_error"] <= 0.02
                and not new_contact_loss
                and not result["fall"]
                and not (result["saturation"] and not base["saturation"])
            )
            rows.append({
                "branch_id": branch["branch_id"], "speed": branch["speed"],
                "seed": branch["seed"], "branch_step": branch["branch_step"],
                "support_category": branch["support_category"],
                "dimension": dimension, "joint": JOINTS[dimension],
                "leg": LEGS[dimension % 4], "joint_group": JOINTS[dimension].split("_")[1],
                "delta": delta, "valid": True, "invalid_reason": "",
                "baseline_cumulative_slip": base["cumulative_slip"],
                "variant_cumulative_slip": result["cumulative_slip"],
                "absolute_slip_difference": result["cumulative_slip"] - base["cumulative_slip"],
                "relative_slip_reduction": slip_reduction,
                "baseline_speed_error": base["speed_error"],
                "variant_speed_error": result["speed_error"],
                "speed_error_difference": result["speed_error"] - base["speed_error"],
                "baseline_heading_error": base["heading_error"],
                "variant_heading_error": result["heading_error"],
                "heading_error_difference": result["heading_error"] - base["heading_error"],
                "gravity_tilt_difference": result["gravity_tilt"] - base["gravity_tilt"],
                "flight_fraction_difference": result["flight_fraction"] - base["flight_fraction"],
                "normal_force_difference": result["normal_force_mean"] - base["normal_force_mean"],
                "baseline_contact_loss": base["contact_loss"],
                "contact_loss": result["contact_loss"],
                "new_contact_loss": new_contact_loss,
                "fall": result["fall"],
                "saturation_increase": result["saturation"] and not base["saturation"],
                "tangential_p95_difference": result["tangential_p95"] - base["tangential_p95"],
                "dangerous_fraction_difference": result["dangerous_fraction"] - base["dangerous_fraction"],
                "friction_p95_difference": result["friction_p95"] - base["friction_p95"],
                "locally_improving": improving,
            })

fieldnames = sorted({key for row in rows for key in row})
with (OUT / "counterfactual_action_results.csv").open(
    "w", newline="", encoding="utf-8"
) as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
valid_rows = [row for row in rows if row["valid"]]
valid_rate = valid / expected
if valid_rate < 0.95:
    dump("counterfactual_action_results.json", {
        "expected_variants": expected, "valid_variants": valid,
        "valid_variant_rate": valid_rate,
        "gate": "FRESH_PROCESS_BRANCH_REPLAY_INSUFFICIENT",
    })
    raise SystemExit(2)

by_branch = defaultdict(list)
for row in valid_rows:
    by_branch[row["branch_id"]].append(row)
branch_results = []
for branch in branches:
    variants = by_branch[branch["branch_id"]]
    improving = [row for row in variants if row["locally_improving"]]
    branch_results.append({
        "branch_id": branch["branch_id"], "speed": branch["speed"],
        "support_category": branch["support_category"],
        "improving": bool(improving),
        "improving_variant_count": len(improving),
        "best_relative_slip_reduction": max(
            (row["relative_slip_reduction"] for row in variants), default=0.0
        ),
    })

speed_results = {}
for speed in sorted({item["speed"] for item in branches}):
    subset = [item for item in branch_results if item["speed"] == speed]
    rate = sum(item["improving"] for item in subset) / len(subset)
    speed_results[str(speed)] = {
        "branches": len(subset),
        "improving_branches": sum(item["improving"] for item in subset),
        "improving_branch_rate": rate,
    }
overall_rate = sum(item["improving"] for item in branch_results) / len(branch_results)
if all(item["improving_branch_rate"] >= 0.30 for item in speed_results.values()):
    controllability = "SLIP_LOCALLY_CONTROLLABLE"
elif overall_rate < 0.10:
    controllability = "SLIP_NOT_LOCALLY_CONTROLLABLE"
else:
    controllability = "SLIP_PARTIALLY_CONTROLLABLE"
dump("counterfactual_action_results.json", {
    "expected_variants": expected, "valid_variants": valid,
    "valid_variant_rate": valid_rate, "gate": "PASS",
    "branch_results": branch_results,
})
dump("local_slip_controllability.json", {
    "classification": controllability,
    "overall_improving_branch_rate": overall_rate,
    "by_speed": speed_results,
    "by_support_category": {
        category: {
            "branches": len(subset := [item for item in branch_results if item["support_category"] == category]),
            "improving_rate": sum(item["improving"] for item in subset) / len(subset),
        }
        for category in sorted({item["support_category"] for item in branch_results})
    },
    "criterion": load_json("protocol.json")["locally_improving"],
})

attribution = []
for dimension, joint in enumerate(JOINTS):
    subset = [row for row in valid_rows if row["dimension"] == dimension]
    improving = [row for row in subset if row["locally_improving"]]
    attribution.append({
        "dimension": dimension, "joint": joint, "leg": LEGS[dimension % 4],
        "joint_group": joint.split("_")[1],
        "improving_variant_rate": len(improving) / len(subset),
        "mean_slip_reduction": float(np.mean([row["relative_slip_reduction"] for row in subset])),
        "mean_speed_tradeoff": float(np.mean([row["speed_error_difference"] for row in subset])),
        "mean_heading_tradeoff": float(np.mean([row["heading_error_difference"] for row in subset])),
        "contact_loss_rate": float(np.mean([row["new_contact_loss"] for row in subset])),
    })
dump("joint_leg_attribution.json", {"joints": attribution})

slip_candidates = [row for row in valid_rows if row["relative_slip_reduction"] >= 0.20]
speed_conflict = np.mean([row["speed_error_difference"] > 0.03 for row in slip_candidates])
heading_conflict = np.mean([row["heading_error_difference"] > 0.02 for row in slip_candidates])
contact_conflict = np.mean([
    row["new_contact_loss"] or row["fall"] or row["saturation_increase"]
    for row in slip_candidates
])
conflicts = {
    "speed": float(speed_conflict), "heading": float(heading_conflict),
    "contact": float(contact_conflict),
}
dominant = [key for key, value in conflicts.items() if value > 0.50]
if not dominant:
    tradeoff = "SLIP_REDUCTION_WITHOUT_CAPABILITY_TRADEOFF"
elif len(dominant) > 1:
    tradeoff = "MULTIPLE_LOCAL_TRADEOFFS"
else:
    tradeoff = {
        "speed": "SLIP_REDUCTION_CONFLICTS_WITH_SPEED",
        "heading": "SLIP_REDUCTION_CONFLICTS_WITH_HEADING",
        "contact": "SLIP_REDUCTION_CONFLICTS_WITH_CONTACT",
    }[dominant[0]]
dump("local_tradeoff_analysis.json", {
    "classification": tradeoff,
    "slip_reducing_variants": len(slip_candidates),
    "conflict_fraction": conflicts,
})

# Central finite differences for later comparison with reward-derived preference.
empirical = []
for branch in branches:
    for dimension in range(12):
        pair = [
            row for row in by_branch[branch["branch_id"]]
            if row["dimension"] == dimension
        ]
        minus = next(row for row in pair if row["delta"] < 0)
        plus = next(row for row in pair if row["delta"] > 0)
        derivative = (
            plus["variant_cumulative_slip"] - minus["variant_cumulative_slip"]
        ) / 0.04
        empirical.append({
            "branch_id": branch["branch_id"], "speed": branch["speed"],
            "dimension": dimension, "joint": JOINTS[dimension],
            "empirical_slip_derivative": derivative,
            "preferred_action_sign": -1 if derivative > 0 else 1,
        })
dump("empirical_local_slip_gradient.json", {"rows": empirical})

# Twenty-percent local-linearity subset: branches were selected by fixed
# manifest order (index modulo five), independently of outcomes.
linearity_branches = [
    item for index, item in enumerate(branches) if index % 5 == 0
]
linearity_rows = []
for branch in linearity_branches:
    baseline = baseline_cache[branch["baseline_b0_run_id"]]
    base = outcome(baseline, branch["branch_step"])
    for dimension in range(12):
        primary_pair = [
            row for row in by_branch[branch["branch_id"]]
            if row["dimension"] == dimension
        ]
        primary_by_sign = {
            -1: next(row for row in primary_pair if row["delta"] < 0),
            1: next(row for row in primary_pair if row["delta"] > 0),
        }
        for sign in (-1, 1):
            effects = {0.02: (
                primary_by_sign[sign]["variant_cumulative_slip"]
                - base["cumulative_slip"]
            )}
            valid_scales = True
            for magnitude in (0.01, 0.04):
                label = str(magnitude).replace(".", "p")
                run_id = (
                    f"{branch['branch_id']}_action_{dimension:02d}_"
                    f"{'plus' if sign > 0 else 'minus'}_{label}"
                )
                path = RAW / f"{run_id}.pt"
                if not path.exists():
                    valid_scales = False
                    continue
                variant = load_trace(run_id)
                matched, _, _ = prefix_match(
                    baseline, variant, branch["branch_step"]
                )
                if not matched:
                    valid_scales = False
                    continue
                effects[magnitude] = (
                    outcome(variant, branch["branch_step"])["cumulative_slip"]
                    - base["cumulative_slip"]
                )
            sign_consistent = (
                valid_scales
                and len({np.sign(value) for value in effects.values() if abs(value) > 1e-12}) <= 1
            )
            monotonic = (
                valid_scales
                and abs(effects[0.01]) <= abs(effects[0.02]) <= abs(effects[0.04])
            )
            linearity_rows.append({
                "branch_id": branch["branch_id"], "speed": branch["speed"],
                "dimension": dimension, "joint": JOINTS[dimension], "sign": sign,
                "valid": valid_scales, "effects": effects,
                "response_sign_consistent": sign_consistent,
                "magnitude_monotonic": monotonic,
            })
valid_linearity = [row for row in linearity_rows if row["valid"]]
dump("local_linearity_audit.json", {
    "branches": len(linearity_branches),
    "branch_fraction": len(linearity_branches) / len(branches),
    "variants_expected": len(linearity_branches) * 12 * 4,
    "comparisons": len(linearity_rows),
    "valid_comparisons": len(valid_linearity),
    "valid_rate": len(valid_linearity) / max(len(linearity_rows), 1),
    "response_sign_consistency_rate": float(np.mean([
        row["response_sign_consistent"] for row in valid_linearity
    ])) if valid_linearity else 0.0,
    "magnitude_monotonic_rate": float(np.mean([
        row["magnitude_monotonic"] for row in valid_linearity
    ])) if valid_linearity else 0.0,
    "primary_delta_extreme_nonlinearity": (
        "NOT_OBSERVED"
        if valid_linearity and np.mean([
            row["response_sign_consistent"] for row in valid_linearity
        ]) >= 0.80 else "PRESENT_OR_INCONCLUSIVE"
    ),
    "rows": linearity_rows,
})
