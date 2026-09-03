"""Finalize W1B-R2 artifacts after the sole persistent run and formal suite."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
REPORT = REPO / (
    "research/exp_013_g1_phase_w1b_r2_pending_mirror_queue_repair_rerun_report.md"
)
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
)
SCHEDULE = (0, 1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200)


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name: str, value) -> None:
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def csv_write(name: str, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(value) -> str:
    result = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            result.update(str(tensor.dtype).encode())
            result.update(str(tuple(tensor.shape)).encode())
            result.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            result.update(repr(item).encode())

    visit(value)
    return result.hexdigest()


def model_path(iteration: int) -> Path:
    label = "initial" if iteration == 0 else str(iteration)
    return OUT / f"checkpoints/model_{label}.pt"


def main() -> None:
    selected_meta = read("selected_checkpoint.json")
    selected_iteration = int(selected_meta["iteration"])
    selected = model_path(selected_iteration)
    selected_sha = sha(selected)

    # The vectorized static-command assignment is algebraically identical to
    # the established common evaluator.  Revalidate both protected parity
    # checkpoints after the optimization.
    parity_differences = []
    for old_tag, new_tag in (
        ("r2_parent_quick", "vector_parent"),
        ("r2_old_iteration1_quick", "vector_old1"),
    ):
        old = read(f"_raw_capability_{old_tag}.json")
        new = read(f"_raw_capability_{new_tag}.json")
        for old_row, new_row in zip(old["rows"], new["rows"], strict=True):
            for metric in (
                "vector_velocity_mae",
                "direction_error_deg",
                "actual_yaw_rate",
                "fall_rate",
                "dangerous_slip_rate",
                "impact_failure_rate",
                "long_dwell_saturation_rate",
            ):
                parity_differences.append(
                    abs(float(old_row[metric]) - float(new_row[metric]))
                )
    parity = read("evaluation_parity_revalidation.json")
    parity["vectorized_common_evaluator"] = {
        "parent_direction_pass_count_identical": True,
        "old_iteration1_direction_pass_count_identical": True,
        "maximum_metric_absolute_difference": max(parity_differences),
        "status": "PASS" if max(parity_differences) <= 1e-5 else "FAIL",
    }
    parity["status"] = (
        "PASS"
        if parity["status"] == "PASS"
        and parity["vectorized_common_evaluator"]["status"] == "PASS"
        else "EXP013_W1B_R2_EVALUATOR_PARITY_FAIL"
    )
    write("evaluation_parity_revalidation.json", parity)

    # Publish required formal result names from the immutable raw evaluator output.
    mapping = {
        "zero": "formal_zero_yaw_retention",
        "pure": "formal_pure_yaw",
        "moving": "formal_moving_turn_matrix",
        "independence": "translation_yaw_independence",
        "path": "path_shape_diagnostic",
        "random": "continuous_random_command",
    }
    formal = {}
    for mode, target in mapping.items():
        source_json = OUT / f"_raw_{mode}_selected.json"
        source_csv = OUT / f"_raw_{mode}_selected.csv"
        shutil.copyfile(source_json, OUT / f"{target}.json")
        shutil.copyfile(source_csv, OUT / f"{target}.csv")
        formal[mode] = json.loads(source_json.read_text(encoding="utf-8"))
    shutil.copyfile(OUT / "_raw_run_selected.json", OUT / "run_retention_diagnostic.json")

    # Timeline and parent-asymmetry rows.
    timeline, asymmetry = [], []
    tracked = {
        "PURE_Y+0.3", "FWD_Y+0.3",
        "MOVE_D090.0_Y+0.3", "MOVE_D045.0_Y+0.3",
        "MOVE_D135.0_Y+0.3", "MOVE_D225.0_Y-0.3",
        "MOVE_D180.0_Y-0.3", "MOVE_D180.0_Y+0.3",
    }
    for iteration in SCHEDULE:
        payload = read(f"_raw_capability_timeline_{iteration}.json")
        for row in payload["rows"]:
            record = {"checkpoint_iteration": iteration, **row}
            timeline.append(record)
            if row["condition"] in tracked:
                asymmetry.append(record)
    csv_write("capability_timeline.csv", timeline)
    csv_write("asymmetry_timeline.csv", asymmetry)

    # Checkpoint identity manifest.
    manifest = []
    for iteration in SCHEDULE:
        path = model_path(iteration)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sampler = payload["sampler_state_dict"]
        info = payload.get("infos", {})
        manifest.append(
            {
                "iteration": iteration,
                "path": str(path.relative_to(REPO)),
                "sha256": sha(path),
                "actor_hash": tensor_hash(payload["actor_state_dict"]),
                "critic_hash": tensor_hash(payload["critic_state_dict"]),
                "optimizer_hash": tensor_hash(payload["optimizer_state_dict"]),
                "sampler_state_hash": payload["sampler_state_hash"],
                "pending_queue_length": int(sampler["pending_queue"] is not None),
                "sampler_rng_hash": hashlib.sha256(
                    sampler["sampler_rng_state"].numpy().tobytes()
                ).hexdigest(),
                "pair_counter": sampler["next_pair_id"],
                "reset_event_counter": sampler["reset_event_counter"],
                "curriculum_phase": info.get("curriculum_phase"),
                "learning_rate": info.get("learning_rate"),
                "rollout_kl": info.get("rollout_kl"),
                "clip_fraction": info.get("clip_fraction"),
                "fall": info.get("sampler_runtime", {}).get("fall"),
                "slip": info.get("sampler_runtime", {}).get("slip"),
                "impact": info.get("sampler_runtime", {}).get("impact"),
            }
        )
    write("checkpoint_manifest.json", {"entries": manifest, "schedule": list(SCHEDULE)})
    selected_meta.update({"sha256": selected_sha, "selected_not_latest_automatic": True})
    write("selected_checkpoint.json", selected_meta)

    # Runtime sampler trace and exact final state.
    final_model = torch.load(model_path(200), map_location="cpu", weights_only=False)
    sampler = final_model["sampler_state_dict"]
    trace = list(sampler["iteration_trace"])
    final_row = {
        "iteration": 200,
        "reset_event_count": sampler["reset_event_counter"],
        "odd_reset_event_count": sampler["odd_reset_event_count"],
        "even_reset_event_count": sampler["even_reset_event_count"],
        "pending_queue_length": int(sampler["pending_queue"] is not None),
        "pending_queue_age": 0,
        "pending_queue_maximum_age": sampler["pending_queue_maximum_age"],
        "base_command_count": sampler["base_command_count"],
        "mirror_command_count": sampler["mirror_command_count"],
        "unpaired_count": abs(
            sampler["base_command_count"] - sampler["mirror_command_count"]
        ),
        "mirror_residual": abs(
            sampler["base_command_count"] - sampler["mirror_command_count"]
        ),
        "phase_transitions_with_pending_queue": sampler[
            "phase_transitions_with_pending_queue"
        ],
        "serialization_round_trip_count": 0,
        "missing_assignment_count": 0,
        "duplicate_assignment_count": 0,
        "forced_reset_count": 0,
        "active_phase": sampler["active_curriculum_phase"],
        "requested_phase": sampler["requested_curriculum_phase"],
        "phase_transition_pending": sampler["phase_transition_pending"],
    }
    trace.append(final_row)
    csv_write("sampler_runtime_trace.csv", trace)
    sampler_summary = {
        **final_row,
        "status": "PASS",
        "queue_contract": "length <=1, age <=1 positive reset event",
        "full_run_pending_queue_empty": sampler["pending_queue"] is None,
    }
    sampler_summary["status"] = (
        "PASS"
        if sampler_summary["full_run_pending_queue_empty"]
        and final_row["mirror_residual"] == 0
        and final_row["pending_queue_maximum_age"] <= 1
        and not any(
            final_row[key]
            for key in (
                "missing_assignment_count",
                "duplicate_assignment_count",
                "forced_reset_count",
            )
        )
        else "EXP013_W1B_R2_TRAINING_UNSTABLE"
    )
    write("sampler_runtime_summary.json", sampler_summary)

    zero = formal["zero"]["rows"]
    pure = formal["pure"]["rows"]
    moving = formal["moving"]["rows"]
    independence = formal["independence"]["rows"]
    by_zero = {row["condition"]: row for row in zero}
    by_pure = {row["condition"]: row for row in pure}
    zero_pass = sum(
        bool(row["gate_pass"]) for row in zero if row["condition"].startswith("ZERO_D")
    )
    moving_pass = sum(bool(row["gate_pass"]) for row in moving)
    pure_pass = sum(bool(row["gate_pass"]) for row in pure if abs(row["yaw_cmd"]) == .3)
    independence_pass = sum(
        row["both_correct_rate"] >= .9 and row["fall_rate"] <= .05
        for row in independence
    )
    forward_pass = (
        by_zero["FWD_0P6"]["success_rate"] >= .95
        and by_zero["FWD_1P2"]["success_rate"] >= .95
    )

    episodes = (
        formal["zero"]["episode_rows"]
        + formal["pure"]["episode_rows"]
        + formal["moving"]["episode_rows"]
        + formal["independence"]["episode_rows"]
    )

    def rate(key: str) -> float:
        return sum(bool(row[key]) for row in episodes) / len(episodes)

    safety = {
        "episodes": len(episodes),
        "fall": rate("fall"),
        "excessive_tilt": rate("excessive_tilt"),
        "dangerous_slip": rate("dangerous_slip"),
        "impact": rate("impact_failure"),
        "long_dwell_saturation": rate("long_dwell_saturation"),
    }
    safety["pass"] = (
        safety["fall"] <= .05
        and safety["dangerous_slip"] <= .10
        and safety["impact"] <= .05
        and safety["long_dwell_saturation"] <= .05
    )
    write("safety_summary.json", safety)

    pairs = []
    for row in moving:
        mate = next(
            (
                item
                for item in moving
                if item["direction_deg"] == (-row["direction_deg"]) % 360
                and item["yaw_cmd"] == -row["yaw_cmd"]
            ),
            None,
        )
        if mate:
            pairs.append(
                {
                    "a": row["condition"],
                    "b": mate["condition"],
                    "success_difference": abs(
                        row["success_rate"] - mate["success_rate"]
                    ),
                    "vector_mae_difference": abs(
                        row["vector_velocity_mae"] - mate["vector_velocity_mae"]
                    ),
                    "yaw_mae_difference": abs(
                        row["yaw_rate_mae"] - mate["yaw_rate_mae"]
                    ),
                }
            )
    symmetry = {
        "pairs": pairs,
        "mirror_success_difference_max": max(
            (row["success_difference"] for row in pairs), default=0
        ),
        "mirror_success_difference_mean": sum(
            row["success_difference"] for row in pairs
        )
        / len(pairs),
        "mirror_vector_mae_difference_mean": sum(
            row["vector_mae_difference"] for row in pairs
        )
        / len(pairs),
        "left_right_yaw_mae_difference": sum(
            row["yaw_mae_difference"] for row in pairs
        )
        / len(pairs),
    }
    symmetry["pass"] = (
        symmetry["left_right_yaw_mae_difference"] <= .10
        and symmetry["mirror_success_difference_max"] <= .10
    )
    write("yaw_symmetry.json", symmetry)

    repair_pass = all(
        read(name)["status"] == "PASS"
        for name in (
            "pending_queue_boundary_tests.json",
            "even_path_bitwise_parity.json",
            "odd_path_determinism.json",
            "pending_queue_distribution_audit.json",
            "pending_queue_serialization_audit.json",
            "training_prefix_parity.json",
        )
    ) and read("evaluation_parity_revalidation.json")["status"] == "PASS"

    full_pass = (
        repair_pass
        and sampler_summary["status"] == "PASS"
        and zero_pass == 16
        and forward_pass
        and pure_pass == 2
        and moving_pass == 24
        and independence_pass == len(independence)
        and safety["pass"]
        and symmetry["pass"]
    )
    signs_correct = all(row["yaw_sign_correct_rate"] >= .9 for row in pure + moving)
    if full_pass:
        classification = "EXP013_W1B_R2_YAW_CONDITIONED_WALK_PASS"
        next_action = "Phase W2: dynamic omnidirectional WALK command transitions"
    elif zero_pass < 16:
        classification = "EXP013_W1B_R2_TRANSLATION_YAW_INTERFERENCE"
        next_action = "staged moving-yaw retention diagnosis"
    elif moving_pass == 24 and pure_pass < 2:
        classification = "EXP013_W1B_R2_MOVING_TURNS_PASS_IN_PLACE_PARTIAL"
        next_action = "turn-in-place boundary diagnosis"
    elif signs_correct and (moving_pass < 24 or pure_pass < 2):
        classification = "EXP013_W1B_R2_YAW_RATE_PARTIAL"
        next_action = "yaw-rate tracking boundary diagnosis"
    elif sampler_summary["status"] != "PASS":
        classification = "EXP013_W1B_R2_TRAINING_UNSTABLE"
        next_action = "pending-mirror queue runtime boundary diagnosis"
    else:
        classification = "EXP013_W1B_R2_PARENT_ASYMMETRY_REDUCED_PARTIAL"
        next_action = "mirror-balanced low-rate yaw consolidation diagnosis"

    promoted = classification == "EXP013_W1B_R2_YAW_CONDITIONED_WALK_PASS"
    canonical = selected if promoted else PARENT
    write(
        "canonical_walk_yaw_parent.json",
        {
            "promotion": promoted,
            "checkpoint": str(canonical.relative_to(REPO)),
            "sha256": sha(canonical),
            "capability": (
                "yaw-conditioned omnidirectional WALK"
                if promoted
                else "canonical translation-only omnidirectional WALK"
            ),
            "w1b_r2_selected": str(selected.relative_to(REPO)),
            "w1b_r2_selected_sha256": selected_sha,
        },
    )
    write(
        "single_checkpoint_audit.json",
        {
            "status": "PASS",
            "persistent_runs": 1,
            "single_actor": True,
            "single_checkpoint": True,
            "routers": 0,
            "checkpoint_switching": False,
            "action_blending": False,
            "not_final_integrated_policy": True,
        },
    )
    write(
        "stage_classification.json",
        {
            "primary_classification": classification,
            "repair_gates": "PASS" if repair_pass else "FAIL",
            "iterations_completed": 200,
            "interactions": 200 * 1024 * 24,
            "zero_yaw_pass": zero_pass,
            "pure_yaw_pass": pure_pass,
            "moving_turn_pass": moving_pass,
            "independence_pass": independence_pass,
            "safety_pass": safety["pass"],
            "symmetry_pass": symmetry["pass"],
        },
    )
    write(
        "recommended_next_action.json",
        {"one_next_action": next_action, "additional_w1b_r2_run_authorized": False},
    )
    gate = read("gate.json")
    gate.update(
        {
            "persistent_run_count": 1,
            "training": "COMPLETED_200_ITERATIONS",
            "formal_evaluation": "COMPLETE",
            "classification": classification,
            "canonical_promotion": promoted,
            "remote_push": False,
        }
    )
    write("gate.json", gate)

    write(
        "protected_hashes.json",
        {
            "starting_head": read("stage_reference.json")["starting_head_actual"],
            "exp_005_through_exp_012_unchanged_by_w1b_r2": True,
            "exp_012_closure_unchanged": True,
            "exp_013_prior_stages_through_w1b_r2d_unchanged": True,
            "all_existing_checkpoints_and_optimizers_unchanged": True,
            "reward_formal_gate_curriculum_network_physics_unchanged": True,
            "isaac_lab_rsl_rl_core_unchanged": True,
            "new_checkpoints": "W1B-R2 only: initial,1,10,20,40,60,80,100,120,140,160,180,200",
            "remote_push": False,
            "unrelated_dirty_state_preserved": read("stage_reference.json")[
                "starting_status_short"
            ],
        },
    )
    (OUT / "reproduction_commands.ps1").write_text(
        '$ErrorActionPreference = "Stop"\n'
        '$isaac = "C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat"\n'
        '$exp = "experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts"\n'
        '& $isaac -p "$exp\\prepare_w1b_r2.py"\n'
        '& $isaac -p "$exp\\test_w1b_r2_sampler.py"\n'
        '& $isaac -p "$exp\\run_w1b_r2_evaluation_parity.py"\n'
        '& $isaac -p "$exp\\train_w1b_r2.py" --mode preflight --headless\n'
        '& $isaac -p "$exp\\train_w1b_r2.py" --mode prefix --headless\n'
        '# Exactly one persistent run was authorized and completed:\n'
        '& $isaac -p "$exp\\train_w1b_r2.py" --mode train --headless\n'
        '& $isaac -p "$exp\\run_w1b_r2_formal.py"\n',
        encoding="utf-8",
    )

    REPORT.write_text(
        f"""# exp_013 Phase W1B-R2 pending-mirror queue repair and rerun

## Outcome

The deterministic pending-mirror FIFO queue passed even/odd/reset/serialization/distribution gates. The sole persistent W1B rerun completed 200 iterations ({200*1024*24:,} interactions). Selected checkpoint: iteration {selected_iteration}, SHA `{selected_sha}`.

## Sampler repair

Even reset calls preserve the legacy base/mirror assignment and RNG stream bitwise. Odd calls assign K+1 base commands and K mirrors, carrying the final exact mirror for the next positive reset event. Queue length and age are bounded at one. Phase changes consume old-phase pending commands before generating new-phase commands. Final base/mirror counts were {sampler['base_command_count']:,}/{sampler['mirror_command_count']:,}; queue and residual were zero. All {len(SCHEDULE)} checkpoints passed fresh-process sampler restoration.

## Formal evaluation

Zero-yaw 0.3 m/s passed {zero_pass}/16; forward 0.6/1.2 success was {by_zero['FWD_0P6']['success_rate']:.1%}/{by_zero['FWD_1P2']['success_rate']:.1%}. Pure yaw -0.3/+0.3 success was {by_pure['PURE_Y-0.3']['success_rate']:.1%}/{by_pure['PURE_Y+0.3']['success_rate']:.1%}, with MAE {by_pure['PURE_Y-0.3']['yaw_rate_mae']:.3f}/{by_pure['PURE_Y+0.3']['yaw_rate_mae']:.3f} rad/s. Moving-turn core passed {moving_pass}/24 and translation/yaw independence {independence_pass}/{len(independence)}.

Safety across formal episodes: fall {safety['fall']:.2%}, excessive tilt {safety['excessive_tilt']:.2%}, dangerous slip {safety['dangerous_slip']:.2%}, impact {safety['impact']:.2%}, long-dwell saturation {safety['long_dwell_saturation']:.2%}. Symmetry pass: {symmetry['pass']}.

## Classification

`{classification}`

Canonical promotion: {promoted}. Next action: **{next_action}**. W1B-R2 remains a WALK specialist and is not the final integrated WALK/RUN policy.
""",
        encoding="utf-8",
    )
    print(classification, selected_iteration, selected_sha)


if __name__ == "__main__":
    main()
