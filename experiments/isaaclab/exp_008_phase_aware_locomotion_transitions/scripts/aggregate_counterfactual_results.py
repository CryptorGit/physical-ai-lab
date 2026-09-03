"""Aggregate fresh-app counterfactual replays and enforce state matching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
OUT = REPO / "results/exp_008_phase_aware_locomotion_transitions/stage0_observability_and_controllability"
CANDIDATES = ["baseline", "walk_expert", "run_expert", "bounded_joint_group", "target_walk_alignment"]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    config = yaml.safe_load((EXP / "configs/stage0_observability_probe.yaml").read_text(encoding="utf-8"))
    frames = [pd.read_csv(OUT / f"counterfactual_raw_{candidate}.csv") for candidate in CANDIDATES]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(OUT / "counterfactual_episode_results.csv", index=False)
    reference = np.load(OUT / "prebranch_baseline.npz")
    comparisons = []
    matched_ids_by_candidate = {}
    baseline_schedule = (
        frame[frame["candidate"].eq("baseline")]
        .set_index("physical_env_id")[["branch_age", "branch_step"]]
        .to_dict("index")
    )
    for candidate in CANDIDATES[1:]:
        current = np.load(OUT / f"prebranch_{candidate}.npz")
        reference_map = {int(value): index for index, value in enumerate(reference["physical_env_id"])}
        current_map = {int(value): index for index, value in enumerate(current["physical_env_id"])}
        candidate_schedule = (
            frame[frame["candidate"].eq(candidate)]
            .set_index("physical_env_id")[["branch_age", "branch_step"]]
            .to_dict("index")
        )
        common = sorted(
            key
            for key in set(reference_map) & set(current_map)
            if key in baseline_schedule
            and key in candidate_schedule
            and baseline_schedule[key] == candidate_schedule[key]
        )
        root_error = max(
            (
                float(
                    np.max(
                        np.abs(
                            reference["root_position"][reference_map[key]]
                            - current["root_position"][current_map[key]]
                        )
                    )
                )
                for key in common
            ),
            default=float("inf"),
        )
        joint_error = max(
            (
                float(
                    np.max(
                        np.abs(
                            reference["joint_position"][reference_map[key]]
                            - current["joint_position"][current_map[key]]
                        )
                    )
                )
                for key in common
            ),
            default=float("inf"),
        )
        velocity_error = max(
            (
                float(
                    np.max(
                        np.abs(
                            reference["velocity"][reference_map[key]]
                            - current["velocity"][current_map[key]]
                        )
                    )
                )
                for key in common
            ),
            default=float("inf"),
        )
        match_pass = (
            bool(common)
            and root_error <= config["counterfactual"]["state_match_root_position_tolerance_m"]
            and joint_error <= config["counterfactual"]["state_match_joint_position_tolerance_rad"]
            and velocity_error <= config["counterfactual"]["state_match_velocity_tolerance"]
        )
        matched_ids_by_candidate[candidate] = set(common) if match_pass else set()
        comparisons.append(
            {
                "candidate": candidate,
                "identical_selected_set": bool(np.array_equal(reference["physical_env_id"], current["physical_env_id"])),
                "matched_envs": len(common),
                "root_position_max_error_m": root_error,
                "joint_position_max_error_rad": joint_error,
                "velocity_max_error": velocity_error,
                "candidate_comparison_valid": match_pass,
            }
        )
    tolerance = config["counterfactual"]
    matching_pass = all(item["candidate_comparison_valid"] for item in comparisons)
    dump(
        "prebranch_state_matching.json",
        {
            "method": "fresh Isaac app per candidate, identical task/reset seed, source route, physical env IDs, and prebranch actions",
            "state_copy": False,
            "comparisons": comparisons,
            "tolerances": {
                "root_position_m": tolerance["state_match_root_position_tolerance_m"],
                "joint_position_rad": tolerance["state_match_joint_position_tolerance_rad"],
                "velocity": tolerance["state_match_velocity_tolerance"],
            },
            "all_within_tolerance": matching_pass,
        },
    )
    summary = {}
    for candidate, group in frame.groupby("candidate"):
        summary[candidate] = {
            "branch_states": int(len(group)),
            "safe_contract_successes": int(group["contract_20_step_success"].sum()),
            "safe_contract_success_rate": float(group["contract_20_step_success"].mean()),
            "unsafe_rate": float(group["unsafe"].mean()),
            "maximum_streak_mean": float(group["maximum_walk_valid_streak"].mean()),
            "maximum_streak_p95": float(group["maximum_walk_valid_streak"].quantile(0.95)),
        }
    dump(
        "counterfactual_results.json",
        {
            "summary": summary,
            "rollout_steps_after_branch": tolerance["rollout_steps_after_branch"],
            "state_copy": False,
            "prebranch_matching_pass": matching_pass,
            "production_capability_claim": False,
        },
    )
    phase_results = {}
    for (candidate, phase), group in frame.groupby(["candidate", "launch_phase"]):
        phase_results[f"{candidate}:{phase}"] = {
            "count": int(len(group)),
            "safe_success_rate": float(group["contract_20_step_success"].mean()),
            "unsafe_rate": float(group["unsafe"].mean()),
        }
    dump("per_phase_corrective_results.json", phase_results)
    valid_rows = []
    for candidate, ids in matched_ids_by_candidate.items():
        if ids:
            valid_rows.append(
                frame[
                    frame["candidate"].eq(candidate)
                    & frame["physical_env_id"].isin(ids)
                ]
            )
    correction = pd.concat(valid_rows, ignore_index=True) if valid_rows else frame.iloc[0:0]
    overall_best = max(
        float(group["contract_20_step_success"].mean())
        for _, group in correction.groupby("candidate")
    ) if len(correction) else 0.0
    phase_best = max(
        float(group["contract_20_step_success"].mean())
        for _, group in correction.groupby(["candidate", "launch_phase"])
    ) if len(correction) else 0.0
    threshold = config["classification"]["bounded_correction_success_min"]
    if overall_best >= threshold:
        classification = "BOUNDED_CORRECTION_EXISTS"
        qualifier = "safe bounded correction met the fixed overall threshold"
    elif phase_best >= threshold:
        classification = "PHASE_CONDITIONAL_CORRECTION_EXISTS"
        qualifier = "only a source-phase subset met the fixed threshold"
    else:
        classification = "NO_LOCAL_CORRECTION_FOUND"
        qualifier = (
            "no prebranch-matched safe correction met the threshold; "
            "unmatched frozen-expert branches were excluded from positive claims"
        )
    dump(
        "controllability_classification.json",
        {
            "classification": classification,
            "qualifier": qualifier,
            "overall_best_safe_success_rate": overall_best,
            "phase_best_safe_success_rate": phase_best,
            "threshold": threshold,
            "prebranch_matching_pass": matching_pass,
        },
    )
    observability = json.loads((OUT / "observability_classification.json").read_text(encoding="utf-8"))["classification"]
    if classification == "PHASE_CONDITIONAL_CORRECTION_EXISTS":
        decision = "PHASE_CONDITIONED_ACTION_ALIGNMENT"
    elif classification == "NO_LOCAL_CORRECTION_FOUND":
        decision = "UNIFIED_WALK_RUN_DISTILLATION"
    elif observability == "STATIC_152D_OBSERVABLE":
        decision = "STATIC_FEEDFORWARD_REDESIGN"
    elif observability == "HISTORY_REQUIRED":
        decision = "RECURRENT_TRANSITION_ACTOR"
    elif observability == "EXPLICIT_PHASE_FEATURES_REQUIRED":
        decision = "EXPLICIT_PHASE_AUGMENTATION"
    else:
        decision = "UNRESOLVED_CONTACT_DYNAMICS"
    dump(
        "final_stage0_decision.json",
        {
            "decision": decision,
            "observability": observability,
            "controllability": classification,
            "single_next_implementation": True,
            "rationale": qualifier,
        },
    )
    print(json.dumps({"matching": matching_pass, "classification": classification, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
